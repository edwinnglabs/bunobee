# Glossary & naming conventions

Shared vocabulary for bunobee. Keep these terms consistent across code, docstrings, and issues so the
same concept reads the same way everywhere.

## Prior / posterior

Follow the [ArviZ](https://python.arviz.org/) / [NumPyro](https://num.pyro.ai/) community convention:

- **The noun is always singular** — `prior`, `posterior`. It names a *distribution* (or a
  specification of one). ArviZ `InferenceData` groups are all singular (`posterior`, `prior`,
  `posterior_predictive`), and the plurality of draws is carried by the `chain` / `draw` dimensions,
  not by pluralizing the noun.
- **A collection of draws gets the `_samples` suffix** — `posterior_samples`, `prior_samples`. Never
  use the bare plurals `posteriors` / `priors` as an identifier for a draw collection. NumPyro follows
  the same pattern: `Predictive(model, posterior_samples=...)`.

| Use | Don't use | Means |
|-----|-----------|-------|
| `posterior` | `posteriors` | a single posterior distribution (e.g. a Kalman `P_{t|t}`) |
| `posterior_samples` | `posteriors` | an `xr.Dataset` / dict of posterior draws from MCMC |
| `prior` | `priors` | a single prior distribution or its specification |
| `prior_samples` | `priors` | a collection of prior draws |

Plain prose may still say "the coefficient priors" when describing several prior *distributions*
(one per coefficient) — the rule above governs *identifiers* (variable names, function names, dict
keys, argument names), not every sentence.
