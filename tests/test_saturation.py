from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from bunobee.saturation import hill


def test_namespace_follows_input_type():
    out_np = hill(np.array([1.0, 2.0]), K=1.0, S=2.0)
    assert isinstance(out_np, np.ndarray)

    out_jnp = hill(jnp.array([1.0, 2.0]), K=1.0, S=2.0)
    assert isinstance(out_jnp, jax.Array)


def test_jax_param_promotes_numpy_input():
    # a JAX scalar param (e.g. a sampled NumPyro site) should promote the result
    out = hill(np.array([1.0, 2.0]), K=jnp.array(1.0), S=2.0)
    assert isinstance(out, jax.Array)


def test_anchor_points():
    # h(0) = 0 exactly, h(K) = max/2 exactly
    out = hill(np.array([0.0, 3.0]), K=3.0, S=2.5, max_effect=4.0)
    np.testing.assert_allclose(out, [0.0, 2.0])


def test_approaches_ceiling():
    # h(x) -> max_effect as x grows large (finite inputs only)
    out = hill(1e8, K=3.0, S=2.5, max_effect=4.0)
    np.testing.assert_allclose(out, 4.0, rtol=1e-9)


def test_monotonic_and_bounded():
    x = np.linspace(0.0, 50.0, 200)
    out = hill(x, K=5.0, S=1.5)
    assert np.all(np.diff(out) >= 0)
    assert np.all((out >= 0.0) & (out < 1.0))


def test_numpy_and_jax_agree():
    x = np.linspace(0.0, 20.0, 100)
    out_np = hill(x, K=4.0, S=2.0, max_effect=3.0)
    out_jnp = hill(jnp.asarray(x), K=4.0, S=2.0, max_effect=3.0)
    np.testing.assert_allclose(out_np, np.asarray(out_jnp), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("s", [0.5, 1.0, 2.0])
def test_gradient_finite_at_zero(s):
    # double-where guard must keep the gradient finite at x == 0 for S < 1
    g = jax.grad(lambda x: hill(x, K=1.0, S=s))(0.0)
    assert jnp.isfinite(g)


def test_works_under_jit():
    f = jax.jit(lambda x: hill(x, K=2.0, S=1.5, max_effect=2.0))
    out = f(jnp.array([0.0, 2.0, 100.0]))
    np.testing.assert_allclose(np.asarray(out), [0.0, 1.0, hill(100.0, 2.0, 1.5, 2.0)], rtol=1e-9)
