"""Run BTTS training notebook code without starting a Jupyter kernel."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "01_btts_model_training.ipynb"
LAST_TRAINING_CELL = 30


def _display(value: object) -> None:
    print(value, flush=True)


def main() -> None:
    os.chdir(ROOT)
    os.environ.setdefault("MPLBACKEND", "Agg")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {
        "__name__": "__main__",
        "__file__": str(NOTEBOOK),
        "display": _display,
    }
    total_started = time.perf_counter()
    for cell_index, cell in enumerate(notebook["cells"]):
        if cell_index > LAST_TRAINING_CELL:
            break
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        lines = source.splitlines()
        first_line = lines[0][:90] if lines else "<empty>"
        print(f"CELL {cell_index} START {first_line}", flush=True)
        started = time.perf_counter()
        exec(compile(source, f"{NOTEBOOK}#cell-{cell_index}", "exec"), namespace)
        print(f"CELL {cell_index} DONE {time.perf_counter() - started:.1f}s", flush=True)
    print(f"TRAINING COMPLETE {time.perf_counter() - total_started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
