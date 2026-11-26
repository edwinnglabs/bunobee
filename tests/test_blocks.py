"""Tests of :mod:`bunobee.models.blocks`, the NumPyro pooling blocks.

Each block is traced under a fixed seed and its output checked against the sites it sampled: the walk against its
steps, the effects and shocks against their zero-sum draws and their groups. ``origin_level`` is checked against
values worked out by hand on a toy panel, which are also what wunkui's ``recent_level`` returns on it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import optax
import pandas as pd
import pytest
from numpyro import handlers
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import optax_to_numpyro

from bunobee import Hierarchy
from bunobee.models.blocks import Horizon, block_index, block_walk, group_shocks, origin_level, pooled_effect

GROUPS = np.array([0, 1, 0, 2, 1])  # five series in three groups
N_SERIES = GROUPS.size


def _run(fn, *args, seed: int = 0, **kwargs):
    """Trace ``fn`` under a seed; return its value and the trace."""
    traced = handlers.trace(handlers.seed(fn, jax.random.PRNGKey(seed)))
    value = traced(*args, **kwargs)
    return value, traced.trace


# ---------------------------------------------------------------------------------------------------------------------
# Horizon and block_index
# ---------------------------------------------------------------------------------------------------------------------


def test_horizon_duration_and_validation():
    assert Horizon(10, 3).duration == 13
    assert Horizon(10).future == 0
    for bad in ({"n_obs": -1}, {"n_obs": 2, "future": -1}, {"n_obs": 2.5}, {"n_obs": True}):
        with pytest.raises(ValueError):
            Horizon(**bad)


def test_block_index_counts_back_from_origin():
    block = np.asarray(block_index(Horizon(20, 5), block_len=4, n_blocks=3))
    # The last 4 training days are block 2, the 4 before block 1, everything earlier block 0, the forecast block 3.
    np.testing.assert_array_equal(block[16:20], 2)
    np.testing.assert_array_equal(block[12:16], 1)
    np.testing.assert_array_equal(block[:12], 0)
    np.testing.assert_array_equal(block[20:], 3)


# ---------------------------------------------------------------------------------------------------------------------
# block_walk
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("future", [0, 10])
def test_block_walk_ends_at_zero_and_is_constant_within_blocks(future):
    horizon, block_len, n_blocks = Horizon(60, future), 7, 4
    scale = jnp.linspace(0.5, 1.5, N_SERIES)

    def model():
        with numpyro.plate("series", N_SERIES):
            return block_walk("walk", scale, horizon, block_len, n_blocks)

    walk, trace = _run(model)
    walk = np.asarray(walk)
    assert walk.shape == (horizon.duration, N_SERIES)

    # Zero on the last training block, and not zero on the others.
    np.testing.assert_array_equal(walk[53:60], 0.0)
    assert np.all(walk[:53] != 0.0)

    # Constant within each block, and each block is minus the sum of the steps after it.
    block = np.asarray(block_index(horizon, block_len, n_blocks))
    step = np.asarray(trace["walk/step"]["value"])
    assert step.shape == (n_blocks - 1, N_SERIES)
    for b in range(n_blocks):
        days = walk[block == b]
        np.testing.assert_allclose(days, np.broadcast_to(days[0], days.shape))
        np.testing.assert_allclose(days[0], -step[b:].sum(0), rtol=1e-6)

    if future:
        # Exactly one more block, over every forecast day, moved by one more step.
        future_step = np.asarray(trace["walk/future"]["value"])
        assert future_step.shape == (1, N_SERIES)
        np.testing.assert_allclose(walk[60:], np.broadcast_to(future_step, (future, N_SERIES)))
        assert block.max() == n_blocks
    else:
        assert "walk/future" not in trace
        assert block.max() == n_blocks - 1


def test_block_walk_with_one_block_is_zero():
    walk, trace = _run(block_walk, "walk", jnp.ones(N_SERIES), Horizon(30), 7, 1)
    np.testing.assert_array_equal(np.asarray(walk), 0.0)
    assert "walk/step" not in trace


# ---------------------------------------------------------------------------------------------------------------------
# pooled_effect
# ---------------------------------------------------------------------------------------------------------------------

WEEKDAY = np.arange(4, 4 + 35) % 7  # five weeks starting on a Friday


def test_pooled_effect_is_shared_by_a_group_and_sums_to_zero():
    effect, trace = _run(pooled_effect, "dow", WEEKDAY, 7, group=GROUPS)
    effect = np.asarray(effect)
    assert effect.shape == (WEEKDAY.size, N_SERIES)
    group_effect = np.asarray(trace["dow/group"]["value"])
    assert group_effect.shape == (3, 7)

    # The series of a group share its effect, read off by weekday.
    np.testing.assert_allclose(effect, group_effect[GROUPS][:, WEEKDAY].T)
    np.testing.assert_array_equal(effect[:, 0], effect[:, 2])
    # Zero over any seven consecutive days.
    for start in range(WEEKDAY.size - 6):
        np.testing.assert_allclose(effect[start : start + 7].sum(0), 0.0, atol=1e-5)


def test_pooled_effect_with_deviations():
    deviation_scale = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])

    def model():
        series = numpyro.plate("series", N_SERIES)
        return pooled_effect("dow", WEEKDAY, 7, group=GROUPS, deviation_scale=deviation_scale, series=series)

    effect, trace = _run(model)
    effect = np.asarray(effect)
    group_effect = np.asarray(trace["dow/group"]["value"])
    deviation = np.asarray(trace["dow/deviation"]["value"])
    assert deviation.shape == (N_SERIES, 7)

    # Each series is its group's effect plus its own scaled deviation; both parts sum to zero over a week.
    expected = group_effect[GROUPS] + np.asarray(deviation_scale)[:, None] * deviation
    np.testing.assert_allclose(effect, expected[:, WEEKDAY].T, rtol=1e-6)
    np.testing.assert_allclose(deviation.sum(1), 0.0, atol=1e-5)
    np.testing.assert_allclose(effect[:7].sum(0), 0.0, atol=1e-5)
    assert not np.allclose(effect[:, 0], effect[:, 2])


def test_pooled_effect_per_series_and_from_a_level():
    def model():
        return pooled_effect("dom", np.arange(31), 31, series=numpyro.plate("series", N_SERIES))

    effect, trace = _run(model)
    assert np.asarray(trace["dom/series"]["value"]).shape == (N_SERIES, 31)
    np.testing.assert_allclose(np.asarray(effect).sum(0), 0.0, atol=1e-5)

    with pytest.raises(ValueError, match="series plate"):
        _run(pooled_effect, "dom", np.arange(31), 31)

    meta = pd.DataFrame({"store": ["a", "b", "a", "c", "b"]})
    level = Hierarchy.from_frame(meta, {"store": ["store"]})["store"]
    from_level, _ = _run(pooled_effect, "dow", WEEKDAY, 7, group=level)
    from_index, _ = _run(pooled_effect, "dow", WEEKDAY, 7, group=level.group_idx)
    np.testing.assert_array_equal(np.asarray(from_level), np.asarray(from_index))


# ---------------------------------------------------------------------------------------------------------------------
# group_shocks
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("future", [0, 10])
def test_group_shocks_sum_to_zero_per_period_and_are_shared(future):
    horizon, period = Horizon(40, future), 7
    scale = jnp.array([0.5, 1.0, 2.0])
    shocks, trace = _run(group_shocks, "store", scale, horizon, GROUPS, period)
    shocks = np.asarray(shocks)
    assert shocks.shape == (horizon.duration, N_SERIES)

    # The series of a group share its shocks; different groups differ.
    np.testing.assert_array_equal(shocks[:, 0], shocks[:, 2])
    np.testing.assert_array_equal(shocks[:, 1], shocks[:, 4])
    assert not np.allclose(shocks[:, 0], shocks[:, 1])

    # Zero over every whole period of training days, counted back from the origin.
    for end in range(horizon.n_obs, period - 1, -period):
        np.testing.assert_allclose(shocks[end - period : end].sum(0), 0.0, atol=1e-5)
    assert np.asarray(trace["store/shock"]["value"]).shape == (-(-40 // period), 3, period)

    if future:
        # Fresh periods over the forecast, starting at the origin.
        assert np.asarray(trace["store/future"]["value"]).shape == (2, 3, period)
        np.testing.assert_allclose(shocks[40:47].sum(0), 0.0, atol=1e-5)
        future_shock = np.asarray(trace["store/future"]["value"])
        np.testing.assert_allclose(shocks[40:47, 3], 2.0 * future_shock[0, 2], rtol=1e-6)
    else:
        assert "store/future" not in trace


# ---------------------------------------------------------------------------------------------------------------------
# origin_level
# ---------------------------------------------------------------------------------------------------------------------


def test_origin_level_with_fallbacks():
    # Six days, four series; the recent window is the last 2 days and the fallback the last 4.
    y = np.array(
        [
            [9, 9, 9, 0],
            [9, 9, 4, 0],
            [9, 3, 2, 0],
            [9, 5, 0, 0],
            [2, 0, 0, 0],
            [4, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    gate = np.array(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    level = np.asarray(origin_level(y, gate, days=2, fallback_days=4, floor=0.05))
    # Series 0: listed on the last 2 days, mean 3. Series 1: not listed recently, mean 4 over its 2 listed days of the
    # last 4. Series 2: listed once in the last 4 days, mean 2. Series 3: never listed, the floor.
    np.testing.assert_allclose(level, np.log(np.array([3.0, 4.0, 2.0, 0.0]) + 0.05), rtol=1e-6)

    # The default fallback is all of y: series 2 is then not listed recently but its 3 listed days average 5.
    level_all = np.asarray(origin_level(y, gate, days=3))
    np.testing.assert_allclose(level_all, np.log(np.array([5.0, 5.0, 5.0, 0.0]) + 0.05), rtol=1e-6)

    # Without a gate every day counts.
    np.testing.assert_allclose(np.asarray(origin_level(y, days=2)), np.log(y[-2:].mean(0) + 0.05), rtol=1e-6)

    with pytest.raises(ValueError, match="same shape"):
        origin_level(y, gate[:, :2])


# ---------------------------------------------------------------------------------------------------------------------
# Scoping and subsampling
# ---------------------------------------------------------------------------------------------------------------------


def test_two_uses_of_a_block_get_distinct_sites():
    horizon = Horizon(21, 7)

    def model():
        a = pooled_effect("dow_a", WEEKDAY[: horizon.duration], 7, group=GROUPS)
        b = pooled_effect("dow_b", WEEKDAY[: horizon.duration], 7, group=GROUPS)
        s = group_shocks("shock_a", 1.0, horizon, GROUPS) + group_shocks("shock_b", 1.0, horizon, GROUPS)
        with numpyro.plate("series", N_SERIES):
            w = block_walk("walk_a", 1.0, horizon, 7, 2) + block_walk("walk_b", 1.0, horizon, 7, 2)
        return a + b + s + w

    _, trace = _run(model)
    samples = {name for name, site in trace.items() if site["type"] == "sample"}
    assert {"dow_a/group", "dow_b/group", "shock_a/shock", "shock_b/shock", "walk_a/step", "walk_b/step"} <= samples
    plates = {name for name, site in trace.items() if site["type"] == "plate"}
    assert {"dow_a/groups", "dow_b/groups", "shock_a/groups", "shock_b/groups"} <= plates


def test_blocks_fit_with_a_subsampled_series_plate():
    """An AutoNormal guide subsamples the caller's series plate through the blocks' sites."""
    n_series, horizon = 40, Horizon(28)
    groups = np.arange(n_series) % 4
    rng = np.random.default_rng(0)
    y = jnp.asarray(rng.poisson(3.0, size=(horizon.n_obs, n_series)), dtype=jnp.float32)
    weekday = np.arange(horizon.n_obs) % 7

    def model(y, subsample_size=None):
        series = numpyro.plate("series", n_series, subsample_size=subsample_size)
        dow = pooled_effect("dow", weekday, 7, group=groups, deviation_scale=0.1, series=series)
        shock = group_shocks("shock", 0.1, horizon, groups, series=series)
        with series:
            level = numpyro.subsample(origin_level(y), event_dim=0)
            walk = block_walk("walk", 0.1, horizon, 7, 4)
            with numpyro.plate("days", horizon.n_obs, dim=-2):
                numpyro.sample(
                    "y",
                    numpyro.distributions.Poisson(jnp.exp(level + walk + dow + shock)),
                    obs=numpyro.subsample(y, event_dim=0),
                )

    guide = AutoNormal(model)
    svi = SVI(model, guide, optax_to_numpyro(optax.adam(0.01)), Trace_ELBO())
    result = svi.run(jax.random.PRNGKey(0), 5, y, subsample_size=10, progress_bar=False)
    assert np.all(np.isfinite(np.asarray(result.losses)))
    assert result.params["walk/step_auto_loc"].shape == (3, n_series)
    assert result.params["dow/deviation_auto_loc"].shape == (n_series, 6)
