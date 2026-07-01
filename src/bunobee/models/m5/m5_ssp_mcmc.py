def fit_one_series(
    sales: np.ndarray,
    num_warmup: int = 100,
    num_samples: int = 100,
    num_chains: int = 4,
    seed: int = 0,
    Z: jnp.ndarray | None = None,
) -> dict:
    """Fit a local-level + weekly-seasonality state-space model to one series.

    Parameters
    ----------
    sales : np.ndarray
        1-D array of daily unit sales (n_steps,).
    num_warmup : int
        NUTS warmup iterations per chain.
    num_samples : int
        NUTS posterior samples per chain.
    num_chains : int
        Number of MCMC chains.
    seed : int
        PRNG seed for reproducibility.
    Z : jnp.ndarray | None
        (n_steps, n_states) pre-built design matrix. When provided the internal
        dummy build is skipped. Pass ``Z_shared`` to avoid redundant work when
        fitting many series of the same length.

    Returns
    -------
    dict
        Keys: ``posterior_dict`` (MCMC samples), ``response_norm`` (float),
        ``Z`` (design matrix), ``a0``, ``P0``.
    """
    sales_clipped = np.clip(sales, 1e-1, None).astype(np.float32)
    response_norm = float(sales_clipped.mean())
    y = jnp.array(np.log(sales_clipped / response_norm))

    n_steps = len(y)

    if Z is None:
        weekly_dummies = make_peridoic_dummies(n_steps, period=7, drop_first=True)
        Z = jnp.concatenate([jnp.ones((n_steps, 1)), weekly_dummies], axis=1)
    n_states = Z.shape[1]

    a0 = jnp.zeros(n_states)
    P0 = jnp.ones(n_states)

    sigma_q_loc_prior = jnp.array([0.05, 0.01])
    sigma_q_scale_prior = jnp.array([0.05, 0.01])

    def _nuts_fn(a0, P0):
        sigma_h = numpyro.sample(
            "sigma_h",
            dist.TruncatedNormal(0.1, 1.0, high=1.0, low=1e-5),
        )
        sigma_q_raw = numpyro.sample(
            "sigma_q",
            dist.TruncatedNormal(
                sigma_q_loc_prior,
                sigma_q_scale_prior,
                high=0.1,
                low=1e-5,
            ),
        )
        n_seas = n_states - 1
        sigma_q = jnp.concatenate([sigma_q_raw[:1], jnp.repeat(sigma_q_raw[1:], n_seas)])

        lp, at, _, _, _, _ = kalman_filter_1d(
            a0=a0,
            P0=P0,
            sigma_h=sigma_h,
            sigma_q=sigma_q,
            y=y,
            Z=Z,
            logp=True,
        )
        numpyro.factor("lp", lp)
        numpyro.deterministic("at", at)
        numpyro.deterministic("mu", jnp.sum(Z * at, -1))

    rng_key = random.PRNGKey(seed)
    mcmc = MCMC(NUTS(_nuts_fn), num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(random.split(rng_key, 1)[0], a0, P0)

    return {
        "posterior_dict": mcmc.get_samples(),
        "response_norm": response_norm,
        "Z": Z,
        "a0": a0,
        "P0": P0,
    }


def predict_one_series(
    fit_result: dict,
    horizon: int = HORIZON,
    Z_future: np.ndarray | None = None,
) -> np.ndarray:
    """Generate point forecast (median) from a fitted single-series model.

    Parameters
    ----------
    fit_result : dict
        Output of ``fit_one_series()``.
    horizon : int
        Number of future steps to forecast.
    Z_future : np.ndarray | None
        (horizon, n_states) pre-built future design matrix. When provided the
        internal dummy continuation is skipped. Pass ``Z_future_shared`` to
        avoid redundant work when forecasting many series.

    Returns
    -------
    np.ndarray
        Point forecasts of shape (horizon,).
    """
    posterior_dict = fit_result["posterior_dict"]
    response_norm = fit_result["response_norm"]

    at_samples = np.array(posterior_dict["at"])
    sigma_h_samples = np.array(posterior_dict["sigma_h"])

    n_steps = at_samples.shape[1]

    if Z_future is None:
        weekly_dummies_full = make_peridoic_dummies(n_steps + horizon, period=7, drop_first=True)
        Z_future = np.concatenate([np.ones((horizon, 1)), weekly_dummies_full[n_steps:]], axis=1)

    a_last = at_samples[:, -1, :]  # (n_samples, n_states)
    mu_future = a_last @ Z_future.T  # (n_samples, horizon)

    eps = np.random.default_rng(42).normal(0, sigma_h_samples[:, None], size=mu_future.shape)
    yhat_samples = np.exp(mu_future + eps) * response_norm

    return np.median(yhat_samples, axis=0)
