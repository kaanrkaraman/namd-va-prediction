from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import requests

from namd_replication.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("download_dryad")

DRYAD_LANDING: Final = "https://datadryad.org/dataset/doi:10.5061/dryad.573n5tb5d"
DATASET_DOI: Final = "doi:10.5061/dryad.573n5tb5d"
DATASET_VERSION: Final = 103316

BROWSER_UA: Final = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


@dataclass(frozen=True)
class PinnedFile:
    name: str
    file_id: int
    size_bytes: int
    sha256: str

    @property
    def stream_url(self) -> str:
        return f"https://datadryad.org/downloads/file_stream/{self.file_id}"


PINNED: Final[tuple[PinnedFile, ...]] = (
    PinnedFile(
        "dataframe.csv",
        571013,
        1_489_723,
        "4b03ca0ac9b1cb0b8ee2431290fec4058ef33d42451a49bfdc781c1db44206f0",
    ),
    PinnedFile(
        "R_code_for_analysis.R",
        571011,
        6_302,
        "db5dd3330cdbe92d81183b913b3678c140ab202ca41713b07f1569ecc9efaa7e",
    ),
    PinnedFile(
        "README.rtf",
        571012,
        2_873,
        "448db3a8b98c45d2f34ed8ebda86aadfa11977d59793384114aa88364d3f2a3f",
    ),
)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def try_download(url: str, dest: Path) -> bool:
    try:
        with requests.get(
            url,
            stream=True,
            timeout=30,
            headers={"User-Agent": BROWSER_UA, "Accept": "*/*"},
            allow_redirects=True,
        ) as r:
            if r.status_code in (202, 401, 403):
                log.warning(
                    "Download blocked (HTTP %d): Dryad bot deterrent.",
                    r.status_code,
                )
                return False
            r.raise_for_status()
            with dest.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
    except requests.RequestException as exc:
        log.warning("Programmatic download of %s failed: %s", url, exc)
        return False
    return True


def ensure_file(pin: PinnedFile, raw_dir: Path) -> bool:
    dest = raw_dir / pin.name
    if dest.exists():
        actual = sha256_of(dest)
        if actual == pin.sha256:
            log.info("OK %s (sha256 verified, %d bytes)", pin.name, dest.stat().st_size)
            return True
        log.warning(
            "%s exists but SHA-256 mismatch:\n  expected %s\n  got      %s",
            pin.name,
            pin.sha256,
            actual,
        )
        return False
    log.info("Attempting programmatic download of %s ...", pin.name)
    if try_download(pin.stream_url, dest):
        actual = sha256_of(dest)
        if actual == pin.sha256:
            log.info("OK %s (downloaded, sha256 verified)", pin.name)
            return True
        log.error(
            "%s downloaded but SHA mismatch: expected %s got %s",
            pin.name,
            pin.sha256,
            actual,
        )
        dest.unlink(missing_ok=True)
    return False


def print_manual_instructions(missing: list[PinnedFile], raw_dir: Path) -> None:
    sys.stderr.write(
        f"\nManual download required. Open {DRYAD_LANDING} in a browser "
        f"and save each file below into {raw_dir}:\n\n"
    )
    for pin in missing:
        sys.stderr.write(
            f"  - {pin.name}  ({pin.size_bytes:,} bytes,"
            f" sha256 {pin.sha256[:12]}...)\n"
        )
        sys.stderr.write(f"    direct URL: {pin.stream_url}\n")
    sys.stderr.write("\nThen re-run this script.\n\n")


def write_manifest(raw_dir: Path) -> Path:
    manifest = {
        "dataset_doi": DATASET_DOI,
        "dataset_version": DATASET_VERSION,
        "verified_at_utc": _dt.datetime.now(_dt.UTC).isoformat(),
        "files": {
            pin.name: {
                "size_bytes": pin.size_bytes,
                "sha256": pin.sha256,
                "source_url": pin.stream_url,
                "local_path": str(raw_dir / pin.name),
            }
            for pin in PINNED
        },
    }
    path = raw_dir / "dryad_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("Wrote manifest -> %s", path)
    return path


def main() -> int:
    raw_dir = settings.data_raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    missing = [pin for pin in PINNED if not ensure_file(pin, raw_dir)]
    if missing:
        print_manual_instructions(missing, raw_dir)
        return 1
    write_manifest(raw_dir)
    log.info("All %d Dryad files present and verified in %s.", len(PINNED), raw_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
