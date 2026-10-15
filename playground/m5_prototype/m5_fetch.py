#!/usr/bin/env python3
"""Download the M5 Forecasting — Accuracy competition files with the Kaggle CLI.

Every M5 script reads the raw competition CSVs from ``playground/resource/m5-forecasting-accuracy/``.
This script puts them there: it runs ``kaggle competitions download``, extracts the archive, deletes
it (``*.zip`` is not gitignored), and checks that the five expected files are present.

Prerequisites
-------------
1. The Kaggle CLI — ``pip install kaggle``. It is a playground prerequisite, not a bunobee dependency.
2. Kaggle credentials — ``kaggle auth login``, a ``KAGGLE_API_TOKEN`` environment variable, or a
   token saved to ``~/.kaggle/access_token`` (generate one at https://www.kaggle.com/settings/api).
3. The competition rules, accepted once while signed in at
   https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules.

Test-window actuals
-------------------
The Kaggle files carry sales up to ``d_1941`` only. The ``d_1942``–``d_1969`` actuals were released by
the organisers after the competition as ``sales_test_evaluation.csv`` — see the Google Drive "Dataset"
folder linked from https://github.com/Mcompetitions/M5-methods. This script does not fetch them.

Usage
-----
    python playground/m5_prototype/m5_fetch.py           # no-op when all five files are present
    python playground/m5_prototype/m5_fetch.py --force   # re-download and overwrite
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

COMPETITION = "m5-forecasting-accuracy"
EXPECTED_FILES = (
    "calendar.csv",
    "sell_prices.csv",
    "sales_train_validation.csv",
    "sales_train_evaluation.csv",
    "sample_submission.csv",
)

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "resource" / COMPETITION
_RULES_URL = f"https://www.kaggle.com/competitions/{COMPETITION}/rules"

# Kaggle CLI failure signatures. Current releases print "Authentication required ..." when no
# credentials are found (and on a 401); older ones raise "Could not find kaggle.json". An
# unaccepted competition is a 403 ("403 Client Error: Forbidden" / "403 - Forbidden").
_CREDENTIALS_ERROR = re.compile(r"authentication required|could not find kaggle\.json|\b401\b|unauthori[sz]ed", re.I)
_RULES_ERROR = re.compile(r"\b403\b|forbidden", re.I)


class FetchError(RuntimeError):
    """A failed fetch, carrying a message that says how to fix it."""


def missing_files(data_dir: Path | str) -> list[str]:
    """Return the expected competition files that are absent from ``data_dir``.

    Parameters
    ----------
    data_dir : Path | str
        Directory that should hold the raw M5 CSV files.

    Returns
    -------
    list[str]
        Names from :data:`EXPECTED_FILES` with no matching file, in that order.
    """
    data_dir = Path(data_dir)
    return [name for name in EXPECTED_FILES if not (data_dir / name).is_file()]


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


def main() -> int:
    """Fetch the M5 files unless they are already present; return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR, help="raw M5 CSV directory")
    parser.add_argument("--force", action="store_true", help="re-download even if all files are present")
    args = parser.parse_args()

    missing = missing_files(args.data_dir)
    if not missing and not args.force:
        print(f"All {len(EXPECTED_FILES)} M5 files are in {args.data_dir}; nothing to do (--force re-downloads).")
        return 0
    if missing and len(missing) < len(EXPECTED_FILES):
        print(f"Missing from {args.data_dir}: {', '.join(missing)}")

    print(f"Downloading {COMPETITION} into {args.data_dir} …", flush=True)
    try:
        download(args.data_dir)
    except FetchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    missing = missing_files(args.data_dir)
    if missing:
        print(f"error: the download did not contain {', '.join(missing)}", file=sys.stderr)
        return 1
    for name in EXPECTED_FILES:
        print(f"  {name:<28} {(args.data_dir / name).stat().st_size / 1e6:8.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
