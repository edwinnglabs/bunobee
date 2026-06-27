# Issue labels

The bunobee backlog lives in GitHub Issues. Every issue carries **one kind** and **one priority**, and
optionally **one area**. `/create-issue` applies these and creates any that are missing; this file is
the single source of truth for their names, meanings, and colors.

## Kind (what the issue is)

| Label | Color | Use when |
|-------|-------|----------|
| `bug` | `#d73a4a` | Something is broken or behaves incorrectly. |
| `feature` | `#0e8a16` | A new capability the package does not have yet. |
| `housekeeping` | `#bfd4f2` | Chore: deps, CI, packaging, config, cleanup — no behavior change for users. |
| `idea` | `#fbca04` | An open proposal or exploration, not yet committed to. |
| `refactor` | `#5319e7` | Internal restructuring; behavior stays the same. |
| `review` | `#cc317c` | Needs a review, audit, or discussion before action. |
| `test` | `#1d76db` | Add, fix, or improve tests / coverage. |

## Priority (how urgent)

| Label | Color | Meaning |
|-------|-------|---------|
| `P0` | `#b60205` | Critical — blocks work or produces incorrect output. Fix now. |
| `P1` | `#d93f0b` | High — should be the next thing picked up. |
| `P2` | `#fef2c0` | Normal / low — do it when time permits. |

## Area (optional — which part of the code)

One `area:<name>` label, derived from the module the issue touches under `src/bunobee/`
(e.g. `area:ssp`, `area:dlt`, `area:m5`, `area:saturation`, `area:regression`). All share color
`ededed`. Omit when no source file is involved (e.g. CI or docs).

## Seeding all labels

Run once to create the full set on the repo (idempotent — safe to re-run):

```sh
gh label create bug          --color d73a4a --description "Broken or incorrect behavior"        --force
gh label create feature      --color 0e8a16 --description "New capability"                       --force
gh label create housekeeping --color bfd4f2 --description "Documentation, workflow, misc."  --force
gh label create idea         --color fbca04 --description "Open proposal / exploration"          --force
gh label create refactor     --color 5319e7 --description "Internal restructuring, no behavior change" --force
gh label create review       --color cc317c --description "Needs review or discussion"           --force
gh label create test         --color 1d76db --description "Add, fix, or improve tests"           --force
gh label create P0           --color b60205 --description "Critical — fix now"                    --force
gh label create P1           --color d93f0b --description "High — do next"                        --force
gh label create P2           --color fef2c0 --description "Normal / low"                          --force
```
