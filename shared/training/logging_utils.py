from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class EpochLogger:
    def __init__(self, log_dir: str | Path, stem: str = "epochs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / f"{stem}.csv"
        self.jsonl_path = self.log_dir / f"{stem}.jsonl"

    def log(self, record: dict[str, Any]) -> None:
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record))
            if write_header:
                writer.writeheader()
            writer.writerow(record)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
