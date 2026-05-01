import numpy as np

from MonteCarloPricer.pricer import MCPricerConfig, black_scholes_call, mc_price_call


# Example usage and convergence plotting
if __name__ == "__main__":
    config = MCPricerConfig(S0=100, K=100, r=0.05, sigma=0.2, T=1.0, N=500000)
    price, stats = mc_price_call(config)
    bs = black_scholes_call(config.S0, config.K, config.r, config.sigma, config.T)

    print(f"MC Price: {price:.4f} ± {stats['std_error']:.4f}")
    print(
        f"95% CI: [{stats['confidence_interval_95'][0]:.4f}, "
        f"{stats['confidence_interval_95'][1]:.4f}]"
    )
    print(f"BS Price: {bs:.4f}")
    print(f"Relative Error: {abs(price - bs) / bs:.4%}")