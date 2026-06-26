"""Saturation transforms for NumPyro kernels.

Marketing-mix / response models map an input ``x`` (e.g. spend) through a
concave, bounded response curve before it contributes to ``y``.  The Hill
function is the canonical choice: a smooth, monotone S-curve that is ``0`` at
``x = 0`` and approaches a ceiling as ``x -> inf``.  It is model-agnostic — the
SSP observation model is one consumer, but any NumPyro kernel can reuse it.

These helpers are written to run unchanged on plain NumPy inputs (eager
analysis, plotting, tests) and on JAX inputs (inside ``jax.jit`` / a NumPyro
model).  The array namespace is chosen from the inputs at call time, so callers
never have to branch on which kind of array they hold.
"""

from __future__ import annotations

from typing import Any

import jax
from jax import numpy as jnp
import numpy as np

ArrayLike = Any


def _array_namespace(*arrays: Any):
    """Return ``jnp`` if any argument is a JAX array/tracer, else ``np``.

    Under ``jax.jit`` the inputs are tracers, which are instances of
    ``jax.Array``; concrete device arrays are too.  Plain NumPy arrays and
    Python scalars are not, so they fall through to NumPy.  A single argument
    being a JAX array is enough to promote the whole computation to JAX.
    """
    for arr in arrays:
        if isinstance(arr, jax.Array):
            return jnp
    return np


def hill(
    x: ArrayLike,
    K: ArrayLike,
    S: ArrayLike,
    max_effect: ArrayLike = 1.0,
) -> ArrayLike:
    r"""Hill saturation transform.

    .. math::

        h(x) = \text{max\_effect} \cdot
               \frac{x^{S}}{K^{S} + x^{S}}

    This is the shape effect ``Hill(x; K, S)`` of Jin et al. (2017), "Bayesian
    Methods for Media Mix Modeling with Carryover and Shape Effects"
    (https://research.google/pubs/archive/46001.pdf) -- the parameter names
    ``K`` and ``S`` follow that paper -- with an added ``max_effect`` ceiling.

    The curve is ``0`` at ``x = 0``, equals ``max_effect / 2`` at ``x = K``,
    increases monotonically, and tends to ``max_effect`` as ``x -> inf``.  ``S``
    controls how sharply the response bends: ``S > 1`` is S-shaped (increasing
    then diminishing returns), ``S = 1`` is the purely concave Michaelis-Menten
    / hyperbolic curve, and ``S < 1`` is steeper concave.  Larger ``S``
    approaches a step.

    Inputs may be plain NumPy arrays/scalars or JAX arrays (including tracers
    inside ``jax.jit`` / a NumPyro model); the output matches the input
    namespace.  Any argument being a JAX array promotes the result to JAX.

    Parameters
    ----------
    x : array_like
        Input signal (e.g. spend), assumed non-negative and finite.  Negative
        entries are clamped to a ``0`` response.  Typically normalized (e.g.
        divided by its median) so ``K`` and ``S`` sit on a comparable scale
        across channels.
    K : array_like
        Half-saturation point (``K`` in the paper): the value of ``x`` at which
        the response reaches half of ``max_effect``.  Must be positive and
        should lie within the observed range of ``x`` (it is poorly identified
        outside it).  Broadcasts against ``x``.  Typical prior: ``Beta(2, 2)``
        rescaled to the observed range, or ``Uniform`` over it.
    S : array_like
        Hill coefficient / shape parameter (``S`` in the paper).  Must be
        positive.  Broadcasts against ``x``.  Typical prior: ``Gamma`` with a
        positive mode (e.g. shape 2, rate 1), or fixed at ``1``.
    max_effect : array_like, optional
        Ceiling the curve approaches as ``x -> inf`` (an extension beyond the
        paper, whose Hill is in ``[0, 1]``).  Defaults to ``1.0``.  Broadcasts
        against ``x``.  Inside a regression ``beta * hill(x, ...)`` it is
        confounded with the channel coefficient ``beta``, so it is usually left
        at ``1.0``.

    Returns
    -------
    array_like
        Saturated response, same shape as the broadcast of the inputs and the
        same array namespace (NumPy or JAX) as the inputs.

    Notes
    -----
    The ``x^S / (K^S + x^S)`` form is used rather than the algebraically equal
    ``1 - K^S / (x^S + K^S)``: it stays accurate at low ``x`` (where the
    response is near ``0``), whereas the subtractive form computes
    ``1 - (almost 1)`` there and loses precision (most visibly in float32).
    The subtractive form's only edge -- robustness as ``x -> inf`` when ``x^S``
    overflows -- does not apply to normalized spend, where ``x^S`` stays finite.

    The power is guarded with a ``where`` so that ``x == 0`` yields exactly
    ``0`` with finite gradients even when ``S < 1`` (a bare ``0 ** S`` would
    otherwise produce a ``nan`` gradient under JAX autodiff, which would break
    gradient-based samplers like NUTS).

    Examples
    --------
    >>> import numpy as np
    >>> hill(np.array([0.0, 1.0, 3.0]), K=1.0, S=2.0)
    array([0. , 0.5, 0.9])
    """
    xp = _array_namespace(x, K, S, max_effect)
    x = xp.asarray(x)
    K = xp.asarray(K)
    S = xp.asarray(S)
    max_effect = xp.asarray(max_effect)

    # x is assumed non-negative; clamp the base so the power is well-defined and
    # differentiable at x == 0 (double-where keeps the gradient finite there).
    positive = x > 0
    x_safe = xp.where(positive, x, 1.0)
    x_pow = xp.where(positive, x_safe**S, 0.0)
    k_pow = K**S
    return max_effect * x_pow / (k_pow + x_pow)
