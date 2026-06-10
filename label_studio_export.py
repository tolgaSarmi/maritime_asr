"""
label_studio_export.py
══════════════════════════════════════════════════════════════════════════════
Label Studio Dataset Exporter for Maritime VHF ASR Dissertation.

  • Maritime_ASR_Main (real)
    → Audio stays on Azure cloud. Manifest stores URLs for streaming.

  • sim_vhf_dataset (simulated)
    → Audio downloaded to disk. Public Azure container — no SAS token needed.

Usage:
    python label_studio_export.py --api-key YOUR_KEY --project real
    python label_studio_export.py --api-key YOUR_KEY --project simulated
    python label_studio_export.py --api-key YOUR_KEY --project both
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import requests
from tqdm import tqdm

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
LS_BASE_URL   = "https://app.heartex.com"
PROJECT_NAMES = {
    "real":      "Maritime_ASR_Main",
    "simulated": "sim_vhf_dataset",
}
OUTPUT_DIRS = {
    "real":      Path("data/real"),
    "simulated": Path("/content/drive/MyDrive/ASR_Dissertation/data/simulated"),
}
# cloud = keep audio on Azure, store URL in manifest
# local = download audio to disk
STORAGE_MODE = {
    "real":      "cloud",
    "simulated": "local",
}


# ─── API Client ──────────────────────────────────────────────────────────────
class LabelStudioClient:

    def __init__(self, api_key: str, base_url: str = LS_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.session  = requests.Session()
        self.session.headers.update({
            "Authorization": f"Token {api_key}",
            "Content-Type":  "application/json",
        })

    # ── Projects ─────────────────────────────────────────────────────────────
    def list_projects(self) -> list[dict]:
        projects, page = [], 1
        while True:
            resp    = self._get("/api/projects/", params={"page": page, "page_size": 50})
            results = resp.get("results", [])
            projects.extend(results)
            if not resp.get("next"):
                break
            page += 1
        return projects

    def find_project(self, name: str) -> dict | None:
        for p in self.list_projects():
            if p.get("title", "").strip().lower() == name.strip().lower():
                return p
        return None

    # ── Tasks ─────────────────────────────────────────────────────────────────
    def export_tasks(self, project_id: int, known_total: int = 0) -> list[dict]:
        """
        Fetch ALL tasks from a project using paginated /api/tasks.

        known_total: the task count from the project object — used as the
                     definitive stop condition since the API's own "count"
                     field returns a global total across all projects.
        """
        tasks     = []
        page      = 1
        page_size = 100   # Heartex cloud caps at 100 regardless of what you request

        log.info("   Fetching %d tasks (100 per page = %d pages)...",
                 known_total, -(-known_total // page_size) if known_total else "?")

        while True:
            resp = self._get(
                "/api/tasks",
                params={
                    "project":   project_id,
                    "page":      page,
                    "page_size": page_size,
                    "fields":    "all",
                },
            )

            # Bare list response — all tasks returned at once
            if isinstance(resp, list):
                tasks.extend(resp)
                log.info("   Got all %d tasks in one response", len(tasks))
                break

            # Standard paginated response
            if "results" in resp:
                results = resp["results"]
            else:
                results = resp.get("tasks", [])

            if not results:
                break

            tasks.extend(results)
            log.info("   Page %d — %d tasks fetched (%d / %d total)",
                     page, len(results), len(tasks), known_total or "?")

            # Stop when we have collected all tasks
            if known_total and len(tasks) >= known_total:
                break

            # Stop if this page had fewer tasks than requested (last page)
            if len(results) < page_size:
                break

            page += 1

        return tasks

    # ── File Download ─────────────────────────────────────────────────────────
    def download_file(self, url: str, dest: Path) -> bool:
        """
        Download a file to disk.
        Uses plain requests — NOT the LS session — to avoid sending the
        Label Studio auth header to Azure, which causes 403 errors.
        """
        if not url.startswith("http"):
            url = self.base_url + url
        try:
            import requests as _req
            r = _req.get(url, stream=True, timeout=60)
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as exc:
            log.warning("Failed to download %s: %s", url, exc)
            return False

    def _get(self, path: str, params: dict | None = None):
        url  = self.base_url + path
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()


# ─── Helpers ─────────────────────────────────────────────────────────────────
def strip_sas_token(url: str) -> str:
    """Remove SAS query string from Azure Blob URL (public container)."""
    parsed = urlparse(url)
    return urlunparse(parsed._replace(query=""))


def extract_transcription(annotations: list[dict]) -> str | None:
    for ann in annotations:
        for result in ann.get("result", []):
            val      = result.get("value", {})
            ann_type = result.get("type", "")
            if ann_type in ("textarea", "transcription"):
                text = val.get("text", [])
                if isinstance(text, list) and text:
                    return text[0].strip()
                if isinstance(text, str) and text:
                    return text.strip()
            if "text" in val:
                text = val["text"]
                if isinstance(text, list) and text:
                    return text[0].strip()
                if isinstance(text, str):
                    return text.strip()
    return None


def extract_audio_url(task_data: dict) -> str | None:
    for key in ("audio", "audio_url", "url", "file"):
        if key in task_data:
            return task_data[key]
    for v in task_data.values():
        if isinstance(v, str) and any(
            v.lower().endswith(ext) for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a")
        ):
            return v
    return None


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^\w\-_.]", "_", s)[:120]


def get_project_task_count(project: dict) -> int:
    """Extract the task count from a project object."""
    for field in ("task_count", "tasks_number", "num_tasks_with_annotations",
                  "total_annotations_number"):
        val = project.get(field, 0)
        if val:
            return int(val)
    return 0


# ─── Export Pipeline ─────────────────────────────────────────────────────────
def export_project(
    client: LabelStudioClient,
    project_key: str,
    skip_unannotated: bool = True,
) -> dict:

    mode         = STORAGE_MODE[project_key]
    project_name = PROJECT_NAMES[project_key]
    output_dir   = OUTPUT_DIRS[project_key]
    audio_dir    = output_dir / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "local":
        audio_dir.mkdir(parents=True, exist_ok=True)
        log.info("💾 Storage: LOCAL  → audio saved to %s", audio_dir)
    else:
        log.info("☁️  Storage: CLOUD  → audio URLs stored in manifest (no download)")

    # Find project
    log.info("🔍 Finding project: '%s'", project_name)
    project = client.find_project(project_name)
    if project is None:
        log.error("Project '%s' not found.", project_name)
        return {"exported": 0, "skipped": 0, "failed": 0}

    project_id    = project["id"]
    known_total   = get_project_task_count(project)
    log.info("✅ Found project ID=%d  |  Tasks: %d", project_id, known_total)

    # Fetch all tasks — pass known_total so pagination stops correctly
    tasks = client.export_tasks(project_id, known_total=known_total)
    log.info("   Received %d task records total", len(tasks))

    manifest: list[dict] = []
    stats = {"exported": 0, "skipped": 0, "failed": 0}

    for task in tqdm(tasks, desc=f"Processing {project_key}", unit="task"):
        task_id     = task.get("id", "unknown")
        annotations = task.get("annotations", [])

        if skip_unannotated and not annotations:
            stats["skipped"] += 1
            continue

        transcription = extract_transcription(annotations)
        if skip_unannotated and not transcription:
            stats["skipped"] += 1
            continue

        audio_url = extract_audio_url(task.get("data", {}))
        if not audio_url:
            stats["skipped"] += 1
            continue

        if mode == "cloud":
            # Real dataset: store Azure URL directly — no download
            record = {
                "id":            task_id,
                "audio_file":    audio_url,
                "transcription": transcription or "",
                "project":       project_name,
                "data_type":     project_key,
                "storage":       "cloud",
            }
            manifest.append(record)
            stats["exported"] += 1

        else:
            # Simulated dataset: download audio (public container, strip SAS)
            clean_url      = strip_sas_token(audio_url)
            parsed         = urlparse(clean_url)
            original_name  = Path(parsed.path).name or f"task_{task_id}.wav"
            audio_filename = f"{task_id}_{sanitize_filename(original_name)}"
            audio_path     = audio_dir / audio_filename

            if not audio_path.exists():
                if not client.download_file(clean_url, audio_path):
                    stats["failed"] += 1
                    continue
                time.sleep(0.03)

            record = {
                "id":            task_id,
                "audio_file":    str(audio_path.relative_to(output_dir)),
                "transcription": transcription or "",
                "project":       project_name,
                "data_type":     project_key,
                "storage":       "local",
            }
            manifest.append(record)
            stats["exported"] += 1

        for field in ("duration", "speaker", "channel", "scenario"):
            if field in task.get("data", {}):
                record[field] = task["data"][field]

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    try:
        import pandas as pd
        pd.DataFrame(manifest).to_csv(output_dir / "manifest.csv", index=False)
        log.info("   Saved CSV → %s", output_dir / "manifest.csv")
    except ImportError:
        pass

    log.info("✅ [%s] Exported=%d  Skipped=%d  Failed=%d → %s",
             project_key, stats["exported"], stats["skipped"],
             stats["failed"], manifest_path)
    return stats


# ─── CLI ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Label Studio datasets for ASR dissertation"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LABEL_STUDIO_API_KEY", ""),
        help="Label Studio API key (or set LABEL_STUDIO_API_KEY env var)",
    )
    parser.add_argument(
        "--project",
        choices=["real", "simulated", "both"],
        default="both",
        help="Which project to export (default: both)",
    )
    parser.add_argument(
        "--include-unannotated",
        action="store_true",
        help="Include tasks with no annotations",
    )
    parser.add_argument(
        "--base-url",
        default=LS_BASE_URL,
        help=f"Label Studio base URL (default: {LS_BASE_URL})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.api_key:
        log.error(
            "API key required.\n"
            "  Provide --api-key KEY  or  set LABEL_STUDIO_API_KEY env var.\n"
            "  Get your key: https://app.heartex.com/user/account"
        )
        sys.exit(1)

    client = LabelStudioClient(api_key=args.api_key, base_url=args.base_url)

    log.info("🌐 Connecting to %s ...", args.base_url)
    try:
        projects = client.list_projects()
        log.info("   Found %d accessible projects", len(projects))
        for p in projects:
            log.info("   • [%d] %s  (tasks: %s)",
                     p["id"], p["title"], get_project_task_count(p) or "?")
    except requests.HTTPError as exc:
        log.error("Connection failed: %s", exc)
        sys.exit(1)

    targets = ["real", "simulated"] if args.project == "both" else [args.project]
    all_stats: dict[str, Any] = {}

    for target in targets:
        log.info("\n" + "═" * 60)
        log.info("📦 Exporting: %s  [%s]",
                 PROJECT_NAMES[target],
                 "cloud — URLs only" if STORAGE_MODE[target] == "cloud"
                 else "local — downloading audio")
        log.info("═" * 60)
        all_stats[target] = export_project(
            client=client,
            project_key=target,
            skip_unannotated=not args.include_unannotated,
        )

    log.info("\n" + "═" * 60)
    log.info("📊 Export Summary")
    log.info("═" * 60)
    for key, s in all_stats.items():
        icon = "☁️ " if STORAGE_MODE[key] == "cloud" else "💾"
        log.info("  %-12s [%s] → exported: %4d | skipped: %4d | failed: %4d",
                 key, icon, s["exported"], s["skipped"], s["failed"])
    log.info("\nNext step:  python main.py --mode data")


if __name__ == "__main__":
    main()
