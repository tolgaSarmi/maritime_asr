"""
label_studio_export.py
══════════════════════════════════════════════════════════════════════════════
Label Studio Dataset Exporter
Connects to https://app.heartex.com and exports both:
  • sim_vhf_dataset  – Simulated VHF maritime speech
  • Maritime_ASR_Main – Real labelled maritime speech

Usage:
    python label_studio_export.py --api-key YOUR_KEY
    python label_studio_export.py --api-key YOUR_KEY --project simulated
    python label_studio_export.py --api-key YOUR_KEY --project real
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

from src.env import load_env_file
from typing import Any
from urllib.parse import urlparse

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
LS_BASE_URL = "https://app.heartex.com"
PROJECT_NAMES = {
    "simulated": "sim_vhf_dataset",
    "real": "Maritime_ASR_Main",
}
OUTPUT_DIRS = {
    "simulated": Path("data/simulated"),
    "real": Path("data/real"),
}


# ─── API Client ──────────────────────────────────────────────────────────────
class LabelStudioClient:
    """Thin REST client for Label Studio / Heartex cloud."""

    def __init__(self, api_key: str, base_url: str = LS_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Token {api_key}",
                "Content-Type": "application/json",
            }
        )

    # ── Projects ─────────────────────────────────────────────────────────────
    def list_projects(self) -> list[dict]:
        projects, page = [], 1
        while True:
            resp = self._get("/api/projects/", params={"page": page, "page_size": 50})
            results = resp.get("results", [])
            projects.extend(results)
            if not resp.get("next"):
                break
            page += 1
        return projects

    def find_project_by_name(self, name: str) -> dict | None:
        for p in self.list_projects():
            if p.get("title", "").strip().lower() == name.strip().lower():
                return p
        return None

    # ── Tasks ─────────────────────────────────────────────────────────────────
    def list_tasks(self, project_id: int) -> list[dict]:
        tasks, page = [], 1
        while True:
            resp = self._get(
                f"/api/tasks/",
                params={"project": project_id, "page": page, "page_size": 100},
            )
            results = resp.get("tasks", resp) if isinstance(resp, dict) else resp
            if isinstance(results, dict):
                results = results.get("results", [])
            tasks.extend(results)
            if isinstance(resp, dict) and not resp.get("next"):
                break
            elif isinstance(resp, list):
                break
            page += 1
        return tasks

    def export_tasks(self, project_id: int, export_type: str = "JSON") -> list[dict]:
        """
        Fetch all tasks with annotations using the paginated /api/tasks endpoint.
        Works on Heartex cloud (app.heartex.com) where /api/projects/{id}/export
        returns 404.
        """
        tasks = []
        page = 1
        while True:
            resp = self._get(
                "/api/tasks",
                params={
                    "project": project_id,
                    "page": page,
                    "page_size": 100,
                    "fields": "all",         # include annotations
                },
            )
            # Handle both response shapes the API returns
            if isinstance(resp, list):
                if not resp:
                    break
                tasks.extend(resp)
                break  # list response means all tasks returned at once
            
            results = resp.get("tasks", resp.get("results", []))
            if not results:
                break
            tasks.extend(results)
            log.info("   Page %d — fetched %d tasks (total so far: %d)",
                     page, len(results), len(tasks))
            if not resp.get("next"):
                break
            page += 1

        return tasks

    # ── Files ─────────────────────────────────────────────────────────────────
    def download_file(self, url: str, dest: Path) -> bool:
        """Download a file, handling both absolute and relative URLs."""
        if not url.startswith("http"):
            url = self.base_url + url
        try:
            r = self.session.get(url, stream=True, timeout=60)
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as exc:
            log.warning("Failed to download %s: %s", url, exc)
            return False

    # ── Internal ──────────────────────────────────────────────────────────────
    def _get(self, path: str, params: dict | None = None, raw: bool = False):
        url = self.base_url + path
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp if raw else resp.json()


# ─── Annotation Parsing ──────────────────────────────────────────────────────
def extract_transcription(annotations: list[dict]) -> str | None:
    """
    Pull transcription text from Label Studio annotation results.
    Handles multiple annotation schemas:
      • type = "textarea"   → value.text[0]
      • type = "transcription" → value.text
    """
    for ann in annotations:
        for result in ann.get("result", []):
            val = result.get("value", {})
            ann_type = result.get("type", "")

            if ann_type in ("textarea", "transcription"):
                text = val.get("text", [])
                if isinstance(text, list) and text:
                    return text[0].strip()
                if isinstance(text, str) and text:
                    return text.strip()

            # Fallback: any key named 'text'
            if "text" in val:
                text = val["text"]
                if isinstance(text, list) and text:
                    return text[0].strip()
                if isinstance(text, str):
                    return text.strip()

    return None


def extract_audio_url(task_data: dict) -> str | None:
    """Extract audio file URL from task data dict."""
    for key in ("audio", "audio_url", "url", "file"):
        if key in task_data:
            return task_data[key]

    # Search recursively one level
    for v in task_data.values():
        if isinstance(v, str) and any(
            v.lower().endswith(ext) for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a")
        ):
            return v

    return None


# ─── Export Pipeline ──────────────────────────────────────────────────────────
def sanitize_filename(s: str) -> str:
    """Create safe filename from string."""
    return re.sub(r"[^\w\-_.]", "_", s)[:120]


def export_project(
    client: LabelStudioClient,
    project_key: str,
    download_audio: bool = True,
    skip_unannotated: bool = True,
) -> dict:
    """
    Export a single Label Studio project.

    Returns a summary dict with counts.
    """
    project_name = PROJECT_NAMES[project_key]
    output_dir = OUTPUT_DIRS[project_key]
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    log.info("🔍 Finding project: '%s'", project_name)
    project = client.find_project_by_name(project_name)
    if project is None:
        log.error("Project '%s' not found. Check your API key and project name.", project_name)
        return {"exported": 0, "skipped": 0, "failed": 0}

    project_id = project["id"]
    total_tasks = project.get("task_count", project.get("num_tasks_with_annotations", "?"))
    log.info("✅ Found project ID=%d  |  Tasks: %s", project_id, total_tasks)

    log.info("📥 Exporting tasks from Label Studio...")
    tasks = client.export_tasks(project_id)
    log.info("   Received %d task records", len(tasks))

    manifest: list[dict] = []
    stats = {"exported": 0, "skipped": 0, "failed": 0}

    for task in tqdm(tasks, desc=f"Processing {project_key}", unit="task"):
        task_id = task.get("id", "unknown")
        annotations = task.get("annotations", [])

        # Skip unannotated tasks
        if skip_unannotated and not annotations:
            stats["skipped"] += 1
            continue

        transcription = extract_transcription(annotations)
        if skip_unannotated and not transcription:
            stats["skipped"] += 1
            continue

        audio_url = extract_audio_url(task.get("data", {}))
        if not audio_url:
            log.debug("Task %s has no audio URL", task_id)
            stats["skipped"] += 1
            continue

        # Derive local filename
        parsed = urlparse(audio_url)
        original_name = Path(parsed.path).name or f"task_{task_id}.wav"
        audio_filename = f"{task_id}_{sanitize_filename(original_name)}"
        audio_path = audio_dir / audio_filename

        # Download audio
        downloaded = False
        if download_audio:
            if audio_path.exists():
                downloaded = True  # already cached
            else:
                downloaded = client.download_file(audio_url, audio_path)
                if downloaded:
                    time.sleep(0.05)  # be polite to the server

        if download_audio and not downloaded:
            stats["failed"] += 1
            continue

        record = {
            "id": task_id,
            "audio_file": str(audio_path.relative_to(output_dir)) if downloaded else None,
            "audio_url": audio_url,
            "transcription": transcription or "",
            "project": project_name,
            "data_type": project_key,
        }

        # Add any extra meta from task data
        for field in ("duration", "speaker", "channel", "scenario"):
            if field in task.get("data", {}):
                record[field] = task["data"][field]

        manifest.append(record)
        stats["exported"] += 1

    # Save manifest JSON
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Save CSV for convenience
    try:
        import pandas as pd  # noqa: PLC0415

        df = pd.DataFrame(manifest)
        df.to_csv(output_dir / "manifest.csv", index=False)
        log.info("   Saved CSV manifest → %s", output_dir / "manifest.csv")
    except ImportError:
        pass

    log.info(
        "✅ [%s] Exported=%d  Skipped=%d  Failed=%d → %s",
        project_key,
        stats["exported"],
        stats["skipped"],
        stats["failed"],
        manifest_path,
    )
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
        choices=["simulated", "real", "both"],
        default="both",
        help="Which project to export",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Only export metadata, do not download audio files",
    )
    parser.add_argument(
        "--include-unannotated",
        action="store_true",
        help="Include tasks that have no completed annotations",
    )
    parser.add_argument(
        "--base-url",
        default=LS_BASE_URL,
        help=f"Label Studio base URL (default: {LS_BASE_URL})",
    )
    return parser.parse_args()


def main() -> None:
    load_env_file()
    args = parse_args()

    if not args.api_key:
        log.error(
            "API key required. Provide --api-key, set LABEL_STUDIO_API_KEY, or add it to .env.\n"
            "  Copy .env.example to .env and get your key from: https://app.heartex.com/user/account"
        )
        sys.exit(1)

    client = LabelStudioClient(api_key=args.api_key, base_url=args.base_url)

    # Verify connection
    log.info("🌐 Connecting to %s ...", args.base_url)
    try:
        projects = client.list_projects()
        log.info("   Found %d accessible projects", len(projects))
        for p in projects:
            log.info("   • [%d] %s  (tasks: %s)", p["id"], p["title"], p.get("task_count", "?"))
    except requests.HTTPError as exc:
        log.error("Authentication failed: %s", exc)
        sys.exit(1)

    targets = ["simulated", "real"] if args.project == "both" else [args.project]
    all_stats: dict[str, Any] = {}

    for target in targets:
        log.info("\n" + "═" * 60)
        log.info("📦 Exporting: %s", PROJECT_NAMES[target])
        log.info("═" * 60)
        stats = export_project(
            client=client,
            project_key=target,
            download_audio=not args.no_download,
            skip_unannotated=not args.include_unannotated,
        )
        all_stats[target] = stats

    log.info("\n" + "═" * 60)
    log.info("📊 Export Summary")
    log.info("═" * 60)
    for key, s in all_stats.items():
        log.info(
            "  %-12s → exported: %4d | skipped: %4d | failed: %4d",
            key,
            s["exported"],
            s["skipped"],
            s["failed"],
        )
    log.info("\nNext step:  python main.py --mode all")


if __name__ == "__main__":
    main()
