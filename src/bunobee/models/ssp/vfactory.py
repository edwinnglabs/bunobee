"""factory design pattern for batched Kalman filters"""

import jax.numpy as jnp
from jax import vmap
from typing import Callable


def make_kalman_batch(
    fn: Callable[..., tuple],
    chunk_size: int | None = 4096,
) -> Callable[..., tuple]:
    """Return a batched Kalman filter that vmaps ``fn`` over independent series.

    The returned callable processes a batch of B series in parallel using
    ``jax.vmap`` over ``sigma_h``, ``sigma_q``, and ``y``. Batches larger
    than ``chunk_size`` are split to limit peak memory. Works with any Kalman
    filter whose per-series arguments are ``sigma_h``, ``sigma_q``, and ``y``.

    Parameters
    ----------
    fn : Callable[..., tuple]
        Single-series Kalman filter. Must accept ``sigma_h``, ``sigma_q``,
        and ``y`` as keyword arguments. All remaining arguments are treated
        as shared across the batch and forwarded from the caller unchanged.
    chunk_size : int | None, optional
        Default maximum series per vmap call. ``None`` disables chunking.
        Default 4096. Can be overridden per-call via ``chunk_size`` kwarg.

    Returns
    -------
    Callable[..., tuple]
        Batched filter with call signature::

            batched_fn(
                sigma_h,          # (B,)
                sigma_q,          # (B, n_states)
                y,                # (B, n_steps)
                chunk_size=<factory default>,
                **shared_kwargs,  # forwarded verbatim to fn
            ) -> tuple            # each element has a leading batch dim B

    Examples
    --------
    >>> kf_batch = make_kalman_batch(kalman_filter_1d)
    >>> log_p, at, Pt, vt, Ft, Kt = kf_batch(
    ...     sigma_h, sigma_q, y, a0=a0, P0=P0, Z=Z, logp=True,
    ... )
    >>> ekf_batch = make_kalman_batch(kalman_filter_1d_ekf)
    >>> log_p, at, Pt, vt, Ft, Kt = ekf_batch(
    ...     sigma_h, sigma_q, y, a0=a0, P0=P0, Z=Z, logp=True, exponent=0.5,
    ... )
    """

    def _batched(
        sigma_h: jnp.ndarray,
        sigma_q: jnp.ndarray,
        y: jnp.ndarray,
        **shared_kwargs,
    ) -> tuple:
        _chunk = shared_kwargs.pop("chunk_size", chunk_size)
        _kf = vmap(
            lambda sh, sq, yi: fn(sigma_h=sh, sigma_q=sq, y=yi, **shared_kwargs),
            in_axes=(0, 0, 0),
        )
        B = y.shape[0]
        if _chunk is None or B <= _chunk:
            return _kf(sigma_h, sigma_q, y)
        outs: list[tuple] = []
        for start in range(0, B, _chunk):
            end = min(start + _chunk, B)
            outs.append(_kf(sigma_h[start:end], sigma_q[start:end], y[start:end]))
        n_out = len(outs[0])
        return tuple(jnp.concatenate([o[i] for o in outs], axis=0) for i in range(n_out))
