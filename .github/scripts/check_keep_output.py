"""Fail CI when a notebook opts out of output stripping without being on the allowlist.

``nbstripout --verify`` skips any notebook (or cell) marked ``"keep_output": true`` in its metadata, so a
one-line metadata edit can commit rendered outputs unnoticed. This script makes every opt-out an explicit,
reviewed edit to the allowlist file.

Usage::

    python .github/scripts/check_keep_output.py .github/keep_output_notebooks.txt $(git ls-files '*.ipynb')
"""

import json
import sys
from pathlib import Path


def read_allowlist(path):
    """Read the allowlist of notebooks that may keep their outputs.

    Parameters
    ----------
    path : str or pathlib.Path
        Text file with one repo-relative notebook path per line. Blank lines and ``#`` comments are ignored.

    Returns
    -------
    set of str
        The allowed notebook paths.
    """
    lines = (line.strip() for line in Path(path).read_text().splitlines())
    return {line for line in lines if line and not line.startswith("#")}


def keeps_output(path):
    """Report whether a notebook opts out of output stripping.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to an ``.ipynb`` file.

    Returns
    -------
    bool
        True if the notebook, or any of its cells, has ``keep_output`` set to true in its metadata.
    """
    nb = json.loads(Path(path).read_text())
    if nb.get("metadata", {}).get("keep_output") is True:
        return True
    return any(cell.get("metadata", {}).get("keep_output") is True for cell in nb.get("cells", []))


def main(argv):
    """Compare the notebooks that keep outputs against the allowlist.

    Parameters
    ----------
    argv : list of str
        The allowlist path followed by the tracked notebook paths.

    Returns
    -------
    int
        0 if every opt-out is allowlisted and every allowlist entry exists, otherwise 1.
    """
    allowlist = read_allowlist(argv[0])
    notebooks = argv[1:]
    opted_out = {nb for nb in notebooks if keeps_output(nb)}

    status = 0
    for nb in sorted(opted_out - allowlist):
        print(f"::error file={nb}::keep_output is set but the notebook is not in {argv[0]}")
        status = 1
    for nb in sorted(allowlist - set(notebooks)):
        print(f"::error file={argv[0]}::{nb} is allowlisted but is not a tracked notebook")
        status = 1
    if status == 0:
        print(f"{len(opted_out)} notebook(s) keep outputs, all allowlisted.")
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
