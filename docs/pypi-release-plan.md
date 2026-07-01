# PyPI Release Readiness Plan

Status: DRAFT — checklist for moving `bunobee` from Test PyPI to real PyPI.

## Preconditions (verified 2026-07-01)

- Package name `bunobee` is **available** on real PyPI (`https://pypi.org/pypi/bunobee/json` → 404).
- Build junk (`dist/`, `venv/`, `.DS_Store`, old artifacts) is gitignored, **not** tracked. No cleanup needed.
- Git remote: `git@github.com:edwinnglabs/bunobee.git`.

## 1. Real-PyPI publish workflow (only Test PyPI exists today)

- Existing: `.github/workflows/publish-test-pypi.yaml` triggers on **every push to main** → `test.pypi.org`.
- Add: `.github/workflows/publish-pypi.yaml` triggered on `release: published` (or tag `v*`).
- Use **trusted publishing** (OIDC, `permissions: id-token: write`), `environment: pypi`, drop
  `repository-url` (defaults to real PyPI).
- Register the publisher on pypi.org under the `pypi` environment before first run.

## 2. Version strategy (currently inconsistent)

- `pyproject.toml` hardcodes `version = "0.0.3dev4"` but build-requires lists `setuptools-scm`
  (unused, no `[tool.setuptools_scm]` section).
- Pick ONE:
  - **Tag-driven (recommended):** add `dynamic = ["version"]` + `[tool.setuptools_scm]`, drop hardcoded
    `version`, release by tagging `v0.1.0`. `src/bunobee/__init__.py` already reads installed metadata.
  - **Manual:** remove `setuptools-scm` from build requires.
- `0.0.3dev4` is a dev pre-release (pip needs `--pre`). Cut a clean `0.1.0` for first formal release.

## 3. Packaging metadata (thin in pyproject.toml)

- Add `[project.urls]` — Homepage / Repository / Issues / Documentation.
- Add `classifiers` — Development Status, `Programming Language :: Python :: 3.11/3.12/3.13`,
  `License :: OSI Approved :: MIT License`, Intended Audience, `Topic :: Scientific/Engineering`.
- Add `keywords` — e.g. `time-series, forecasting, bayesian, jax, numpyro, kalman-filter`.
- Fix license form: replace `license = { text = "MIT License" }` (deprecated) with
  `license = "MIT"` (SPDX, PEP 639) + `license-files = ["LICENSE"]`.

## 4. README as PyPI landing page

- `README.md` is the `readme` → becomes the PyPI description. Currently no install / usage.
- Add: `pip install bunobee`, a ~10-line quickstart (DLT fit or Kalman filter), badges
  (PyPI version, CI, license, Python versions), JAX install caveat (CPU/GPU differ).

## 5. Dependencies

- Verify `jax>=0.8.0,<0.10.0` and `numpyro>=0.20.0` match what CI tests against.
- Move `ipywidgets` OUT of core runtime deps into an extra:
  `[project.optional-dependencies] notebook = ["ipywidgets"]` (it's a Jupyter UI package).

## 6. Nice-to-haves for "formal"

- Add empty `src/bunobee/py.typed` marker (PEP 561) — ships type hints downstream.
- Add `CHANGELOG.md`.
- Curate top-level `__init__.py` public API + `__all__` (currently only sets `__version__`).
- CI: add `python -m build` + `twine check dist/*` + install wheel in a clean venv and import it.
- Audit `src/bunobee/models/m5/*` — mark experimental modules or exclude from public API.

## Minimal path to first real release

1. Cut `ipywidgets` to an extra; add urls/classifiers/keywords; fix license form.
2. Decide version model; set `0.1.0`.
3. Flesh out README (install + quickstart).
4. Add `publish-pypi.yaml` on release/tag with trusted publishing to `pypi` environment.
5. Add `twine check` + clean-venv install to CI.
6. Tag `v0.1.0` → GitHub Release → workflow publishes.

## Key file paths

- Metadata: `pyproject.toml`
- Test PyPI workflow: `.github/workflows/publish-test-pypi.yaml`
- CI workflow: `.github/workflows/ci.yaml`
- New real-PyPI workflow (to create): `.github/workflows/publish-pypi.yaml`
- README (PyPI description): `README.md`
- Package init / version: `src/bunobee/__init__.py`
