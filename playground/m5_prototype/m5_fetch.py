#!/usr/bin/env python3
"""Download the M5 Forecasting — Accuracy competition files and the official reference files.

Every M5 script reads the raw competition CSVs from ``playground/resource/m5-forecasting-accuracy/``.
This script puts them there: it runs ``kaggle competitions download``, extracts the archive, deletes
it (``*.zip`` is not gitignored), and checks that the five expected files are present. It then
downloads the four reference files below into the same directory.

Prerequisites
-------------
1. The Kaggle CLI — ``pip install kaggle``. It is a playground prerequisite, not a bunobee dependency.
2. Kaggle credentials — ``kaggle auth login``, a ``KAGGLE_API_TOKEN`` environment variable, or a
   token saved to ``~/.kaggle/access_token`` (generate one at https://www.kaggle.com/settings/api).
3. The competition rules, accepted once while signed in at
   https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules.

Reference files
---------------
The Kaggle files carry sales up to ``d_1941`` only. After the competition the organisers released the
test-period actuals and the official scoring weights (see the Google Drive "Dataset" folder linked from
https://github.com/Mcompetitions/M5-methods). A public zip mirror of that dataset (:data:`REFERENCE_URL`)
needs no credentials; this script extracts only :data:`REFERENCE_FILES` from it:

- ``sales_test_validation.csv`` / ``sales_test_evaluation.csv`` — actuals for d_1914–d_1941 and
  d_1942–d_1969. ``m5_preprocess.py`` fills the ``test`` split from the evaluation file, so the
  ``_evaluation`` rows of a submission can be scored locally.
- ``weights_validation.csv`` / ``weights_evaluation.csv`` — the official per-series WRMSSE weights,
  which ``m5_check_weights.py`` compares with the local scorer's.

Usage
-----
    python playground/m5_prototype/m5_fetch.py           # no-op when all nine files are present
    python playground/m5_prototype/m5_fetch.py --force   # re-download and overwrite
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from m5_common import DATA_DIR

COMPETITION = "m5-forecasting-accuracy"
EXPECTED_FILES = (
    "calendar.csv",
    "sell_prices.csv",
    "sales_train_validation.csv",
    "sales_train_evaluation.csv",
    "sample_submission.csv",
)

REFERENCE_URL = "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip"
REFERENCE_FILES = (
    "sales_test_validation.csv",
    "sales_test_evaluation.csv",
    "weights_validation.csv",
    "weights_evaluation.csv",
)

_RULES_URL = f"https://www.kaggle.com/competitions/{COMPETITION}/rules"

# Kaggle CLI failure signatures. Current releases print "Authentication required ..." when no
# credentials are found (and on a 401); older ones raise "Could not find kaggle.json". An
# unaccepted competition is a 403 ("403 Client Error: Forbidden" / "403 - Forbidden").
_CREDENTIALS_ERROR = re.compile(r"authentication required|could not find kaggle\.json|\b401\b|unauthori[sz]ed", re.I)
_RULES_ERROR = re.compile(r"\b403\b|forbidden", re.I)


class FetchError(RuntimeError):
    """A failed fetch, carrying a message that says how to fix it."""


def missing_files(data_dir: Path | str, names: tuple[str, ...] = EXPECTED_FILES) -> list[str]:
    """Return the files in ``names`` that are absent from ``data_dir``.

    Parameters
    ----------
    data_dir : Path | str
        Directory that should hold the raw M5 CSV files.
    names : tuple of str
        File names to look for, by default :data:`EXPECTED_FILES`.

    Returns
    -------
    list[str]
        Names from ``names`` with no matching file, in that order.
    """
    data_dir = Path(data_dir)
    return [name for name in names if not (data_dir / name).is_file()]


def _diagnose(output: str) -> str:
    """Map Kaggle CLI output to an actionable message."""
    if _CREDENTIALS_ERROR.search(output):
        return (
            "Kaggle credentials are missing or invalid. Run `kaggle auth login`, or create a token at "
            "https://www.kaggle.com/settings/api and export it as KAGGLE_API_TOKEN (or save it to "
            "~/.kaggle/access_token)."
        )
    if _RULES_ERROR.search(output):
        return (
            f"Kaggle refused the download (403). Accept the competition rules at {_RULES_URL} while "
            "signed in to the account your credentials belong to, then re-run."
        )
    return "The Kaggle CLI failed; its output is below."


def download(data_dir: Path | str, kaggle: str = "kaggle") -> None:
    """Download and extract the competition archive into ``data_dir``.

    Existing files are overwritten. The archive is deleted afterwards, even if extraction fails.

    Parameters
    ----------
    data_dir : Path | str
        Destination directory; created if missing.
    kaggle : str
        Name or path of the Kaggle CLI executable.

    Raises
    ------
    FetchError
        If the CLI is not installed, credentials are missing or invalid, the competition rules have not
        been accepted, or the download fails for another reason.
    """
    executable = shutil.which(kaggle)
    if executable is None:
        raise FetchError(
            f"The Kaggle CLI ({kaggle!r}) was not found on PATH. Install it with `pip install kaggle` "
            "(any environment on PATH will do), then re-run."
        )

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    command = [executable, "competitions", "download", "-c", COMPETITION, "-p", str(data_dir), "--force"]
    result = subprocess.run(command, capture_output=True, text=True)
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if result.returncode != 0:
        raise FetchError(f"{_diagnose(output)}\n\n$ {' '.join(command)}\n{output}")

    archive = data_dir / f"{COMPETITION}.zip"
    if not archive.is_file():
        raise FetchError(f"The Kaggle CLI exited cleanly but did not write {archive}.\n\n{output}")
    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(data_dir)
    except zipfile.BadZipFile as e:
        raise FetchError(f"{archive} is not a valid zip archive ({e}); re-run to download it again.") from e
    finally:
        archive.unlink(missing_ok=True)


def download_reference(data_dir: Path | str, url: str = REFERENCE_URL) -> None:
    """Download the reference mirror and extract :data:`REFERENCE_FILES` into ``data_dir``.

    Existing files are overwritten. The archive is deleted afterwards, even if extraction fails.

    Parameters
    ----------
    data_dir : Path | str
        Destination directory; created if missing.
    url : str
        Zip archive holding the reference files at its root, by default :data:`REFERENCE_URL`.

    Raises
    ------
    FetchError
        If the download fails, the archive is not a valid zip, or it lacks a reference file.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "m5_reference.zip"
    try:
        try:
            urllib.request.urlretrieve(url, archive)
        except OSError as e:
            raise FetchError(f"Could not download the reference files from {url} ({e}).") from e
        with zipfile.ZipFile(archive) as zf:
            absent = sorted(set(REFERENCE_FILES) - set(zf.namelist()))
            if absent:
                raise FetchError(f"{url} does not contain {', '.join(absent)}.")
            for name in REFERENCE_FILES:
                zf.extract(name, data_dir)
    except zipfile.BadZipFile as e:
        raise FetchError(f"{url} did not return a valid zip archive ({e}); re-run to download it again.") from e
    finally:
        archive.unlink(missing_ok=True)


def _fetch(label: str, names: tuple[str, ...], fetch, data_dir: Path, force: bool) -> bool:
    """Run ``fetch(data_dir)`` unless all of ``names`` are present; return False on failure."""
    missing = missing_files(data_dir, names)
    if not missing and not force:
        print(f"All {len(names)} {label} files are in {data_dir}; nothing to do (--force re-downloads).")
        return True
    if missing and len(missing) < len(names):
        print(f"Missing from {data_dir}: {', '.join(missing)}")

    print(f"Downloading the {label} files into {data_dir} …", flush=True)
    try:
        fetch(data_dir)
    except FetchError as e:
        print(f"error: {e}", file=sys.stderr)
        return False

    missing = missing_files(data_dir, names)
    if missing:
        print(f"error: the download did not contain {', '.join(missing)}", file=sys.stderr)
        return False
    for name in names:
        print(f"  {name:<28} {(data_dir / name).stat().st_size / 1e6:8.1f} MB")
    return True


def main() -> int:
    """Fetch the competition and reference files unless they are present; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="raw M5 CSV directory")
    parser.add_argument("--force", action="store_true", help="re-download even if all files are present")
    args = parser.parse_args()

    ok = _fetch(COMPETITION, EXPECTED_FILES, download, args.data_dir, args.force)
    ok = _fetch("reference", REFERENCE_FILES, download_reference, args.data_dir, args.force) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
