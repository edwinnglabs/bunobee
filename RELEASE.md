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
5. Once `pypi` succeeds, the `github-release` job tags the commit `vX.Y.Z` and creates the
   GitHub release, using the matching `## [vX.Y.Z]` section of `CHANGELOG.md` as the notes
   (falling back to GitHub's generated notes if that section is missing). It no-ops if the
   release already exists. Keep the changelog section written before you bump the version.
6. **Required:** once the GitHub release exists, bump `main` to the next dev version (see
   [After a release](#after-a-release)). Do not leave `main` at the final version.

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

## After a release

While `main` sits at a final version, every later push to `main` (docs, notebooks, CI changes) reaches
the `pypi` job and leaves a "Waiting" run that needs approval or cancelling. To avoid that, bump
`pyproject.toml` to the next dev version as soon as the GitHub release for `vX.Y.Z` exists:
`X.Y.(Z+1).dev0` (for example `0.0.5` -> `0.0.6.dev0`). Pushes then publish to Test PyPI only,
unattended, until the next final version is bumped in.

### Why this step is not optional

PyPI never accepts the same version twice, so which commit gets published is decided by whichever
`pypi` run you approve first. After `0.0.6` was released `main` stayed at `0.0.6`, so every merged PR
(#134, #137, #138, #139, #140) queued another "Waiting" `pypi` run. The run for the release commit
`b2f9216` was never approved; the run for a later commit `ad0b994` was, so PyPI `0.0.6` and the `v0.0.6`
tag point at `ad0b994`, not the release commit. Later approvals were no-ops (`skip-existing`).

Checklist when releasing `X.Y.Z`:

1. Merge the release PR (version `X.Y.Z`, changelog section `## [vX.Y.Z]`) and approve only that run.
2. Wait for `github-release` to succeed.
3. Immediately merge a `chore: bump main to X.Y.(Z+1).dev0` PR, and cancel any other "Waiting" `pypi` runs.
4. Use the canonical PEP 440 spelling `X.Y.Z.devN` (with the dot).
