from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil


WIPOSE_URL = "https://drive.google.com/file/d/1WL6bJ-rSVdsclRt9RFc0l5hhtXYmfNg9/view?usp=sharing"
WIMANS_HANDLE = "shuokanghuang/wimans"


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: Path, dataset: str, source: str, files: list[Path], status: str) -> None:
    payload = {
        "dataset": dataset,
        "source": source,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "files": [
            {"path": str(file), "bytes": file.stat().st_size, "sha256": sha256(file)}
            for file in sorted(files)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def download_wipose(raw_dir: Path, manifest_dir: Path) -> None:
    import gdown

    destination = raw_dir / "wipose"
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "Wi-Pose.rar"
    gdown.download(url=WIPOSE_URL, output=str(archive), quiet=False, resume=True)
    if not archive.is_file() or archive.stat().st_size == 0:
        raise RuntimeError("Wi-Pose download did not produce a non-empty archive")
    write_manifest(manifest_dir / "wipose.json", "Wi-Pose", WIPOSE_URL, [archive], "verified_archive")


def download_wimans(raw_dir: Path, manifest_dir: Path) -> None:
    """Download/preserve the Kaggle archive without semantic inspection.

    KaggleHub keeps the downloaded archive at a stable cache path. Reusing that
    path avoids a second automatic extraction when this command is rerun.
    """
    import kagglehub

    destination = raw_dir / "wimans"
    destination.mkdir(parents=True, exist_ok=True)
    cache_archive = Path.home() / ".cache/kagglehub/datasets/shuokanghuang/wimans/1.archive"
    if cache_archive.is_file() and cache_archive.stat().st_size > 0:
        downloaded = cache_archive
    else:
        downloaded = Path(kagglehub.dataset_download(WIMANS_HANDLE))
    if not downloaded.exists():
        raise RuntimeError("KaggleHub returned a path that does not exist")
    preserved: list[Path] = []
    if downloaded.is_file():
        target = destination / downloaded.name
        if target.name == "1.archive":
            target = destination / "WiMANS.zip"
        if downloaded.resolve() != target.resolve():
            shutil.copy2(downloaded, target)
        preserved.append(target)
    else:
        # KaggleHub may expose public datasets as cached files rather than one archive.
        # Copying is byte-preserving; semantic inspection is deferred.
        target = destination / "kagglehub_download"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite {target}")
        shutil.copytree(downloaded, target)
        preserved.extend(path for path in target.rglob("*") if path.is_file())
    if not preserved:
        raise RuntimeError("WiMANS download produced no files")
    write_manifest(manifest_dir / "wimans.json", "WiMANS", WIMANS_HANDLE, preserved, "verified_files")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw Wi-Pose and WiMANS data without preprocessing")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", choices=("all", "wipose", "wimans"), default="all")
    args = parser.parse_args()
    raw_dir = args.project_root / "data" / "raw"
    manifest_dir = args.project_root / "data" / "manifests"
    if args.dataset in ("all", "wipose"):
        download_wipose(raw_dir, manifest_dir)
    if args.dataset in ("all", "wimans"):
        download_wimans(raw_dir, manifest_dir)


if __name__ == "__main__":
    main()
