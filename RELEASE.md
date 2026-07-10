# Release process

This describes how version bumps in `pyproject.toml` flow through to Test PyPI and PyPI.

> **Implemented in** [`.github/workflows/publish.yaml`](.github/workflows/publish.yaml).
> A single workflow builds the distribution once and fans it out to Test PyPI and
> (for final versions) PyPI, so both indexes receive identical bytes.

## How it should work

Every push to `main` is inspected for the `version` in `pyproject.toml`. That version is
parsed as [PEP 440](https://peps.python.org/pep-0440/) and classified as either a
**prerelease** (`.devN`, `aN`, `bN`, `rcN`) or a **final** release. The classification
alone decides which index/indices get published to — there's no separate manual trigger.

| Version string | `is_prerelease` | TestPyPI job | PyPI job | Approval needed? |
|---|---|---|---|---|
| `0.0.5.dev0`, `.dev1`, ... | `true` | ✅ runs | ⛔ skipped | No — `testpypi` environment has no protection rule, publishes unattended |
| `0.0.5a1`, `0.0.5b1`, `0.0.5rc1` | `true` | ✅ runs | ⛔ skipped | No |
| `0.0.5` (final) | `false` | ✅ runs | ✅ runs | **Yes** — see below |
| Same version pushed again (no bump) | — | ✅ runs, no-ops (`skip-existing: true`) | same as above, no-op if already published | Same as above |

## The approval gate

The `pypi` GitHub environment already has a required-reviewer rule (you as approver). The
`pypi` job only *triggers* when `is_prerelease == false`, but hitting that environment
pauses the job in a **"Waiting"** state on the Actions run page — no PyPI upload happens
until you click **Review deployments → Approve**.

`testpypi` has no such rule, so it always publishes immediately.

This gives a natural test-before-promote flow for a final version, e.g. `0.0.5`:

1. Push `0.0.5`. `testpypi` publishes right away.
2. Once `testpypi` succeeds, the `pypi` job (which `needs: testpypi`) enters its
   pending-approval wait — nothing has hit real PyPI yet. If the Test PyPI build/publish
   fails, `pypi` never starts, so you're never asked to approve a broken release.
3. Install from Test PyPI and sanity-check the release:
   `pip install -i https://test.pypi.org/simple/ bunobee==0.0.5`.
4. Happy with it → approve the pending run → the `pypi` job executes and publishes for
   real. Not happy → don't approve (or cancel the run) — nothing reaches PyPI.

## Idempotency

Both publish steps use `skip-existing: true`, so re-pushing the same version (e.g. a
follow-up commit that doesn't touch `version`) never fails with `400 File already exists`
— it just no-ops for whichever index already has that version. This replaces the old
`HEAD~1` git-diff check, which broke on merge commits (see the 2026-07-06 21:29 run that
hit exactly this 400 error).

## Bumping a version

- Dev iteration you want on Test PyPI only: bump to `X.Y.Z.dev0` (increment the `devN`
  suffix for subsequent test builds of the same target version).
- Ready to ship: bump to the final `X.Y.Z` — either from a `.devN` of the same version or
  directly from the prior release. Both paths publish to Test PyPI immediately and to PyPI
  once approved.
