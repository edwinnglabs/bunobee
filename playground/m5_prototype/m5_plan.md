# Scalability Plan for State Space Model Optimization

## Context

The M5 forecasting problem requires fitting 30,490 independent daily series (each 1,913 steps) with a local-level + weekly-seasonality state space model. Currently, `fit_one_series_opt` processes **one series at a time** via a Python `for` loop around a JIT'd Adam step — estimated ~12-24 hours for all series. The Kalman filter (`kalman_filter_1d`) already uses `jax.lax.scan` and is fully JAX-traceable, making it ripe for `vmap`/`pmap` batching.

---

## Tier 1: Batch Kalman Filter with `jax.vmap` (biggest win)

**File:** `src/wunkui/models/ssp/univariate.py`

Add `kalman_filter_1d_batch` — a thin `vmap` wrapper around the existing `kalman_filter_1d`:

- `y`: `(B, n_steps)` — batch axis 0
- `sigma_h`: `(B,)`, `sigma_q`: `(B, n_states)` — per-series
- `Z`: `(n_steps, n_states)` — **shared** via `in_axes=None` (the DOW design matrix is identical for all series)
- `a0`, `P0`: shared via `in_axes=None` (both are `zeros`/`ones` of shape `(n_states,)`)
- Skip `a_obs_loc`/`a_obs_var`/`positivity_idx` in the batch API initially (M5 doesn't use them)

`vmap` composes cleanly with `lax.scan` — JAX batches the scan body automatically. No changes to the existing `kalman_filter_1d` internals.

**Memory:** ~3-4 GB for 30K series at float32 (y + at + Pt outputs). Fits on a 16 GB GPU. Add a chunking wrapper for safety.

---

## Tier 2: Batch Optimizer Loop in JAX

**File:** `src/wunkui/models/m5/m5_ssp_optim.py`

Add `fit_batch_series_opt(sales_matrix, n_iter, lr, Z)` → fits **all B series in one JIT call**:

1. **Batched loss**: `neg_log_posterior_batch(params_batch: (B,3), y_batch: (B, n_steps), ...)` calls the Tier 1 batch Kalman filter. Gradient of `sum(losses)` w.r.t. `(B,3)` params gives correct per-series gradients (block-diagonal Jacobian).

2. **Replace Python `for` with `lax.scan`**: Use fixed iteration count + best-param tracking per series (avoids `while_loop` complexity):
   ```
   each scan step: compute loss+grad → adam update → track per-series best params/loss
   ```
   This is simpler and more GPU-efficient than per-series early stopping (no branch divergence).

3. **optax.adam** automatically handles `(B, 3)` params — momentum buffers become `(B, 3)`.

4. Add `predict_batch_series_opt(fit_results, Z_future)` for batched forecasting.

**Estimated speedup:** Serial → Tier 1+2 is **~200-500x** (12-24 hrs → 2-5 minutes on one GPU).

---

## Tier 3: Multi-Device with `pmap`

**File:** `src/wunkui/models/m5/m5_ssp_optim.py`

Thin wrapper: `pmap` over devices, `vmap` within each device. Pad series count to be divisible by `jax.local_device_count()`, reshape to `(n_devices, B_per_device, ...)`, apply `pmap(fit_batch_series_opt)`.

Linear speedup with device count. Only worthwhile if multiple GPUs/TPUs are available.

---

## Tier 4: Hierarchical Parameter Sharing

**Files:** `src/wunkui/models/m5/m5_ssp_optim.py`, potentially new `src/wunkui/models/m5/m5_hierarchy.py`

Share `sigma_q_seas` at the department × store level (70 groups instead of 30K params). Each series keeps its own `sigma_h` and `sigma_q_level`. This regularizes estimates for sparse/intermittent series.

Two approaches:
- **Hard sharing**: `sigma_q_seas = group_params[group_idx[i]]` — gradient flows via scatter-add
- **Hierarchical prior (shrinkage)**: group-level hyperparameters define prior mean/variance for series-level params

Most complex tier but highest modeling payoff on bottom-volume series.

---

## Implementation Order

```
Tier 1 (batch KF) → Tier 2 (batch optimizer) → Tier 3 (pmap)  
                            └→ Tier 4 (hierarchical)
```

| Config | Est. wall-clock (30K series) |
|---|---|
| Current (serial) | ~12-24 hours |
| Tier 1 only (vmap KF, Python opt loop) | ~2-4 hours |
| **Tier 1 + 2 (fully batched, 1 GPU)** | **~2-5 minutes** |
| Tier 1 + 2 + 3 (multi-GPU) | ~30s-1 min |

**Recommendation:** Implement Tiers 1+2 first — this is the critical path delivering ~200-500x speedup.

---

## Files to Modify

| File | Changes |
|---|---|
| `src/wunkui/models/ssp/univariate.py` | Add `kalman_filter_1d_batch` (vmap wrapper) |
| `src/wunkui/models/m5/m5_ssp_optim.py` | Add `fit_batch_series_opt`, `predict_batch_series_opt` |
| `playground/m5_prototype/m5_accuracy.ipynb` | Replace per-series loop with single batch call |

Existing single-series APIs (`fit_one_series_opt`, `predict_one_series_opt`) are preserved for debugging.

---

## Additional Considerations

- **Float64 for log-likelihood**: Accumulating over 1913 steps in float32 may lose precision. Consider `jax.config.update("jax_enable_x64", True)` or promote just the log-likelihood accumulator.
- **JIT compilation cache**: First call will take ~30-120s to compile the nested scan structure. Subsequent calls with same shapes are cached.
- **Chunking**: If device memory is tight, wrap in chunks of 4096-8192 series and concatenate.

## Verification

1. **Correctness**: Run `fit_batch_series_opt` on 10-20 series and compare `at`, `sigma_h`, `sigma_q` against `fit_one_series_opt` loop — results should match within float32 tolerance.
2. **Speed**: Time the batch fit on full 30K series vs. serial loop on a small subset — extrapolate and verify the expected speedup.
3. **Notebook**: Update `m5_accuracy.ipynb` to use the batch API on the full dataset and verify forecasts are reasonable.
