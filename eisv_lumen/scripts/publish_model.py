"""Publish EISV-Lumen models to HuggingFace Hub.

Supports uploading teacher (LoRA adapter) and student (decision-forest)
models with their associated artifacts.

Usage:
    python3 -m eisv_lumen.scripts.publish_model teacher [--dry-run]
    python3 -m eisv_lumen.scripts.publish_model student [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # eisv-lumen/

TEACHER_DIR = PROJECT_ROOT / "outputs" / "teacher_lora_v6" / "final_adapter"
TEACHER_REPO = "hikewa/eisv-lumen-teacher"
TEACHER_FILES = [
    "adapter_model.safetensors",
    "adapter_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "README.md",
]

STUDENT_DIR = PROJECT_ROOT / "outputs" / "student_small"
STUDENT_REPO = "hikewa/eisv-lumen-student"
STUDENT_FILES = [
    "exported/pattern_forest.json",
    "exported/token1_forest.json",
    "exported/token2_forest.json",
    "exported/scaler.json",
    "exported/mappings.json",
    "exported/student_inference.py",
    "eval_results.json",
    "training_metrics.json",
    "README.md",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _human_size(nbytes: int) -> str:
    """Return a human-readable file size string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024  # type: ignore[assignment]
    return f"{nbytes:.1f} TB"


def _resolve_files(
    base_dir: Path, file_list: list[str]
) -> list[tuple[Path, str]]:
    """Return list of (local_path, repo_path) for files that exist.

    Prints a warning and skips files that are missing.
    """
    resolved = []
    for rel in file_list:
        local = base_dir / rel
        if local.exists():
            resolved.append((local, rel))
        else:
            print(f"  [SKIP] {rel} (not found)", file=sys.stderr)
    return resolved


# ---------------------------------------------------------------------------
# Core upload logic
# ---------------------------------------------------------------------------


def upload_model(
    base_dir: Path,
    repo_id: str,
    file_list: list[str],
    dry_run: bool = False,
) -> None:
    """Upload model files to a HuggingFace model repo.

    Parameters
    ----------
    base_dir:
        Local directory containing the model files.
    repo_id:
        HuggingFace repo id, e.g. ``hikewa/eisv-lumen-teacher``.
    file_list:
        Relative paths (within *base_dir*) to upload.
    dry_run:
        If True, list files and sizes without uploading.
    """
    print(f"\nModel directory : {base_dir}")
    print(f"HF repo        : {repo_id}")
    print(f"Dry run        : {dry_run}\n")

    files = _resolve_files(base_dir, file_list)

    if not files:
        print("ERROR: No files found to upload.", file=sys.stderr)
        sys.exit(1)

    # --- Dry-run: just show what would be uploaded -------------------------
    total_bytes = 0
    print(f"{'File':<45} {'Size':>12}")
    print("-" * 59)
    for local_path, repo_path in files:
        size = local_path.stat().st_size
        total_bytes += size
        print(f"  {repo_path:<43} {_human_size(size):>12}")
    print("-" * 59)
    print(f"  {'TOTAL':<43} {_human_size(total_bytes):>12}")
    print(f"  {len(files)} file(s)\n")

    if dry_run:
        print("Dry run complete. No files uploaded.")
        return

    # --- Actual upload -----------------------------------------------------
    from huggingface_hub import HfApi

    api = HfApi()

    # Create repo if needed (model type is default)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"Repo ensured: https://huggingface.co/{repo_id}")

    for local_path, repo_path in files:
        print(f"  Uploading {repo_path} ({_human_size(local_path.stat().st_size)}) ...", end=" ")
        sys.stdout.flush()
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="model",
        )
        print("done")

    url = f"https://huggingface.co/{repo_id}"
    print(f"\nUpload complete: {url}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish EISV-Lumen models to HuggingFace Hub",
    )
    sub = parser.add_subparsers(dest="model", required=True)

    # teacher subcommand
    t = sub.add_parser("teacher", help="Upload teacher LoRA adapter")
    t.add_argument("--dry-run", action="store_true", help="Show files without uploading")

    # student subcommand
    s = sub.add_parser("student", help="Upload student decision-forest model")
    s.add_argument("--dry-run", action="store_true", help="Show files without uploading")

    args = parser.parse_args()

    if args.model == "teacher":
        upload_model(TEACHER_DIR, TEACHER_REPO, TEACHER_FILES, dry_run=args.dry_run)
    elif args.model == "student":
        upload_model(STUDENT_DIR, STUDENT_REPO, STUDENT_FILES, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
