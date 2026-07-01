# Glossary & naming conventions

Shared vocabulary for bunobee. Keep these terms consistent across code, docstrings, and issues so the
same concept reads the same way everywhere.

## Prior / posterior / inference data

Follow the [ArviZ](https://python.arviz.org/) / [NumPyro](https://num.pyro.ai/) community convention.
The two libraries name **different objects at different layers**, so we name identifiers by their
**data type** — that is the distinction that actually matters when reading the code:

- **`idata` — an `xr.Dataset`.** The structured inference/results container: an ArviZ-style dataset
  with `(chain, draw, ...)` dims (plus any derived summaries like `*_p50`). ArviZ groups are singular
  and carry no `*_samples` suffix — the plurality of draws lives in the `chain` / `draw` dims. Use
  `idata` for anything you build, pass around, or return as an `xr.Dataset`.
- **`posterior_dict` — a `dict`.** The *unstructured* draws straight out of `mcmc.get_samples()`, a
  plain `dict[str, array]` before it is wrapped into an `xr.Dataset`. This is NumPyro's
  `Predictive(model, posterior_samples=...)` layer; we suffix it `_dict` to make the type explicit
  and to distinguish it from `idata`.
- **`prior` / `posterior` (singular)** — a single distribution or its specification (e.g. a Kalman
  `P_{t|t}`, or a natural-scale prior spec in the SSP models). Keep the noun singular.
- **Never** use the bare plurals `posteriors` / `priors` as an identifier.

| Use | Type | For |
|-----|------|-----|
| `idata` | `xr.Dataset` | a structured inference/results dataset (draws + summaries) |
| `posterior_dict` | `dict` | raw draws from `mcmc.get_samples()` |
| `posterior` / `prior` | scalar/spec | a single distribution or its specification |

Plain prose may still say "the coefficient priors" when describing several prior *distributions*
(one per coefficient) — the rule above governs *identifiers* (variable names, function names, dict
keys, argument names), not every sentence.
