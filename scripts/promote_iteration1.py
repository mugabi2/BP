"""Promote validated BTTS Iteration 1 candidate to the live model.

Usage (from repository root):
    python scripts/promote_iteration1.py

Run the training notebook first and review its held-out evaluation. No collector
or database changes are made. Original V1 artifacts are backed up on promotion.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = PROJECT_ROOT / "models"

FILES = (
    ("btts_model_iteration1.joblib", "btts_model.joblib", "btts_model_v1_backup.joblib"),
    ("btts_feature_columns_iteration1.json", "btts_feature_columns.json",
     "btts_feature_columns_v1_backup.json"),
)


def main() -> None:
    sources = [(MODELS / src, MODELS / dest, MODELS / backup)
               for src, dest, backup in FILES]
    missing = [str(src) for src, _, _ in sources if not src.is_file()]
    if missing:
        raise SystemExit(
            "Iteration 1 candidate artifacts missing. Execute the training notebook first:\n"
            + "\n".join(missing)
        )
    metadata = json.loads(sources[1][0].read_text(encoding="utf-8"))
    if metadata.get("iteration") != 1 or not metadata.get("feature_columns"):
        raise SystemExit("Invalid Iteration 1 feature metadata; refusing to promote.")
    if not all(dest.is_file() for _, dest, _ in sources):
        raise SystemExit("V1 live model/metadata not found; refusing to promote.")
    for _, dest, backup in sources:
        if not backup.exists():
            shutil.copy2(dest, backup)
    for src, dest, _ in sources:
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        shutil.copy2(src, tmp)
        tmp.replace(dest)
    print(f"Promoted Iteration 1 model with {len(metadata['feature_columns'])} features.")
    print("Restart/reload Streamlit so its model cache is refreshed.")
    print("Original V1 backups preserved in models/.")


if __name__ == "__main__":
    main()
