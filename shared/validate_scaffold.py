from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import yaml


CONDITIONS = ("source_only_zero_shot", "ssl_domain_adapted")
DOCS = ("TASKS.md", "MODEL.md", "TRAINING.md", "PIPELINE.md", "PREPROCESSING.md")


def validate(project_root: Path) -> None:
    required = [
        project_root / "README.md",
        project_root / "environment.yml",
        project_root / "notebooks/WiFi_CSI_HPE_A2.ipynb",
        project_root / "shared/configs/base.yaml",
    ]
    for condition in CONDITIONS:
        root = project_root / "conditions" / condition
        required.extend(root / "docs" / doc for doc in DOCS)
        required.extend((root / "configs/condition.yaml", root / "logs/.gitkeep", root / "results/.gitkeep"))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise AssertionError(f"Missing required files: {missing}")

    for path in project_root.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for path in project_root.rglob("*.yaml"):
        with path.open(encoding="utf-8") as handle:
            if yaml.safe_load(handle) is None:
                raise AssertionError(f"Empty YAML: {path}")
    for path in project_root.rglob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if notebook.get("nbformat") != 4:
            raise AssertionError(f"Unsupported notebook format: {path}")
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code" and (cell.get("execution_count") is not None or cell.get("outputs")):
                raise AssertionError(f"Notebook is not clean: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    validate(args.project_root)
    print(f"Scaffold validation passed: {args.project_root}")


if __name__ == "__main__":
    main()

