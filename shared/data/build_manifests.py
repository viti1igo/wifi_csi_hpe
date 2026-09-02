"""Build lightweight file-reference manifests after archive extraction."""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/manifests"
WIPOSE = ROOT / "data/extracted/wipose/Wi-Pose"
WIMANS = ROOT / "data/extracted/wimans"

def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ("Train", "Test"):
        for p in sorted((WIPOSE / split).glob("*.mat")):
            m = re.match(r"(.+?)_(\d+)-frame(\d+)\.mat$", p.name)
            activity, recording, frame = m.groups() if m else ("unknown", "unknown", "unknown")
            rows.append({"sample_id": p.stem, "csi_path": str(p), "pose_path": str(p),
                         "participant": recording, "activity": activity, "archive_split": split})
    write_jsonl(OUT / "wipose_samples.jsonl", rows)
    rows = []
    for p in sorted((WIMANS / "wifi_csi").rglob("*")):
        if p.is_file():
            rows.append({"sample_id": p.stem, "csi_path": str(p), "source": "wimans"})
    write_jsonl(OUT / "wimans_csi_samples.jsonl", rows)
    print(f"Wi-Pose records: {len(list((OUT / 'wipose_samples.jsonl').open()))}")
    print(f"WiMANS CSI records: {len(list((OUT / 'wimans_csi_samples.jsonl').open()))}")

if __name__ == "__main__":
    main()
