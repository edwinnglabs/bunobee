# Glossary & naming conventions

Shared vocabulary for bunobee. Keep these terms consistent across code, docstrings, and issues so the
same concept reads the same way everywhere.

## Prior / posterior

Follow the [ArviZ](https://python.arviz.org/) / [NumPyro](https://num.pyro.ai/) community convention.
The two libraries name **different objects at different layers**, so we split by *data structure*, not
by pluralization:

- **`posterior` / `prior` (singular)** — the structured inference object: an `xr.Dataset` with
  `(chain, draw, ...)` dims, i.e. an ArviZ `InferenceData` group. ArviZ groups are all singular
  (`posterior`, `prior`, `posterior_predictive`) and there is **no `*_samples` group** — the plurality
  of draws is carried by the `chain` / `draw` dimensions, not by the name. Use this for anything you
  build with, pass around as, or return as an `xr.Dataset`.
- **`posterior_samples` (raw dict)** — the *unstructured* draws straight out of
  `mcmc.get_samples()`: a plain `dict[str, array]` before it is wrapped in an `xr.Dataset`. This
  matches NumPyro's `Predictive(model, posterior_samples=...)`. Reserve the `_samples` suffix for this
  raw layer only.
- **Never** use the bare plurals `posteriors` / `priors` as an identifier.

| Use | For | Don't use |
|-----|-----|-----------|
| `posterior` / `prior` | an `xr.Dataset` inference object, or a single distribution/spec | `posteriors` / `priors` |
| `posterior_samples` | a raw `dict` of draws from `mcmc.get_samples()` | `posteriors` |

Plain prose may still say "the coefficient priors" when describing several prior *distributions*
(one per coefficient) — the rule above governs *identifiers* (variable names, function names, dict
keys, argument names), not every sentence.
