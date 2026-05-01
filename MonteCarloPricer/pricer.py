from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class MCPricerConfig:
    S0: float
    K: float
    r: float
    sigma: float
    T: float
    N: int = 100_000
    seed: Optional[int] = None
    use_antithetic: bool = True
    block_size: int = 10_000

    def validate(self) -> None:
        if self.S0 <= 0:
            raise ValueError("S0 must be positive")
        if self.K <= 0:
            raise ValueError("K must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.T <= 0:
            raise ValueError("T must be positive")
        if self.N <= 0:
            raise ValueError("N must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")


def black_scholes_call(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def _rng_from_seed(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def _generate_Z(rng: np.random.Generator, n: int, antithetic: bool) -> np.ndarray:
    if not antithetic:
        return rng.standard_normal(n)

    half = n // 2
    Z = rng.standard_normal(half)
    Z = np.concatenate([Z, -Z])
    if n % 2:
        Z = np.append(Z, rng.standard_normal(1))
    return Z


def simulate_terminal_prices(config: MCPricerConfig, Z: np.ndarray) -> np.ndarray:
    drift = (config.r - 0.5 * config.sigma**2) * config.T
    diffusion = config.sigma * np.sqrt(config.T)
    return config.S0 * np.exp(drift + diffusion * Z)


def mc_call_from_Z(config: MCPricerConfig, Z: np.ndarray) -> Tuple[float, dict]:
    config.validate()
    ST = simulate_terminal_prices(config, Z)
    payoffs = np.maximum(ST - config.K, 0.0)

    discount = np.exp(-config.r * config.T)
    mean_payoff = float(payoffs.mean())
    std_payoff = float(payoffs.std(ddof=1)) if len(payoffs) > 1 else 0.0

    price = discount * mean_payoff
    std_error = discount * std_payoff / np.sqrt(len(payoffs)) if len(payoffs) > 0 else np.nan
    ci95 = (price - 1.96 * std_error, price + 1.96 * std_error)

    stats = {
        "price": float(price),
        "std_error": float(std_error),
        "std_dev": float(discount * std_payoff),
        "confidence_interval_95": (float(ci95[0]), float(ci95[1])),
        "num_paths": int(len(payoffs)),
        "relative_error": float(std_error / price) if price != 0 else float("inf"),
        "ST": ST,
        "payoffs": payoffs,
    }
    return float(price), stats


def mc_price_call(config: MCPricerConfig) -> Tuple[float, dict]:
    config.validate()
    rng = _rng_from_seed(config.seed)
    Z = _generate_Z(rng, config.N, config.use_antithetic)
    price, stats = mc_call_from_Z(config, Z)

    if config.use_antithetic:
        half = len(stats["payoffs"]) // 2
        p = stats["payoffs"]
        pair_mean = 0.5 * (p[:half] + p[half : 2 * half]) if half > 0 else p
        var_pair = float(pair_mean.var(ddof=1)) if len(pair_mean) > 1 else 0.0
        var_individual = float(p.var(ddof=1)) if len(p) > 1 else 0.0
        stats["variance_reduction_ratio"] = (var_individual / var_pair) if var_pair > 0 else 1.0
    else:
        stats["variance_reduction_ratio"] = 1.0

    return price, stats


def mc_running_convergence(config: MCPricerConfig) -> dict:
    config.validate()
    rng = _rng_from_seed(config.seed)

    discount = np.exp(-config.r * config.T)
    drift = (config.r - 0.5 * config.sigma**2) * config.T
    diffusion = config.sigma * np.sqrt(config.T)

    running_n = []
    running_price = []
    running_se = []

    sum_payoff = 0.0
    sum_payoff2 = 0.0
    n_seen = 0

    for i in range(0, config.N, config.block_size):
        n_block = min(config.block_size, config.N - i)
        Z = _generate_Z(rng, n_block, config.use_antithetic)
        ST = config.S0 * np.exp(drift + diffusion * Z)
        payoffs = np.maximum(ST - config.K, 0.0)

        n_seen += len(payoffs)
        sum_payoff += float(payoffs.sum())
        sum_payoff2 += float((payoffs**2).sum())

        mean = sum_payoff / n_seen
        if n_seen > 1:
            var = max(sum_payoff2 / n_seen - mean**2, 0.0) * (n_seen / (n_seen - 1))
            se = discount * np.sqrt(var) / np.sqrt(n_seen)
        else:
            se = 0.0

        running_n.append(n_seen)
        running_price.append(discount * mean)
        running_se.append(se)

    return {
        "n": np.array(running_n, dtype=int),
        "price": np.array(running_price, dtype=float),
        "std_error": np.array(running_se, dtype=float),
    }


def bump_greeks_mc(config: MCPricerConfig, dS: float, dSigma: float) -> dict:
    config.validate()
    rng = _rng_from_seed(config.seed)
    Z = _generate_Z(rng, config.N, config.use_antithetic)

    base_price, _ = mc_call_from_Z(config, Z)

    cfg_up_S = MCPricerConfig(**{**config.__dict__, "S0": config.S0 + dS})
    cfg_dn_S = MCPricerConfig(**{**config.__dict__, "S0": max(1e-12, config.S0 - dS)})
    up_S, _ = mc_call_from_Z(cfg_up_S, Z)
    dn_S, _ = mc_call_from_Z(cfg_dn_S, Z)
    delta = (up_S - dn_S) / (2.0 * dS)

    cfg_up_v = MCPricerConfig(**{**config.__dict__, "sigma": config.sigma + dSigma})
    cfg_dn_v = MCPricerConfig(**{**config.__dict__, "sigma": max(1e-12, config.sigma - dSigma)})
    up_v, _ = mc_call_from_Z(cfg_up_v, Z)
    dn_v, _ = mc_call_from_Z(cfg_dn_v, Z)
    vega = (up_v - dn_v) / (2.0 * dSigma)

    return {"price": float(base_price), "delta": float(delta), "vega": float(vega)}


def sample_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = _rng_from_seed(seed)
    dt = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)
    W = np.cumsum(np.sqrt(dt) * rng.standard_normal((n_paths, n_steps)), axis=1)
    W = np.concatenate([np.zeros((n_paths, 1)), W], axis=1)
    S = S0 * np.exp((r - 0.5 * sigma**2) * t + sigma * W)
    return t, S

