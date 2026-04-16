import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from pathlib import Path

class HullWhiteModel:
    def __init__(self, a, sigma, discount_curve):
        self.a = a # mean reversion speed
        self.sigma = sigma # volatility
        self.discount_curve = discount_curve # initial market curve
        # Avoid including t=0 exactly to prevent numerical issues (e.g., 0/0 in curve formulas).
        self.t_grid = np.linspace(1e-6, 30, 3000)
        self.theta = self._calibrate_drift()
    
    def _calibrate_drift(self):
        """Calibrate theta(t) to match initial term structure"""
        t = self.t_grid
        P = self.discount_curve(t)
        f = -np.log(P)
        
        # Compute forward rates
        f_forward = -np.gradient(np.log(P), t)
        
        # Compute theta(t)
        theta = np.gradient(f_forward, t) + self.a * f_forward + (self.sigma**2 /(2*self.a)) * (1 - np.exp(-2*self.a * t))
        return CubicSpline(t, theta)
    
    def B(self, t, T):
        return (1 - np.exp(-self.a * (T - t))) / self.a
    
    def A(self, t, T):
        """Function A(t, T) for bond pricing"""
        POT = self.discount_curve(T)
        POt = self.discount_curve(t)
        BtT = self.B(t, T)
        term1 = np.log(POT / POt)
        term2 = BtT * self.theta(t)
        term3 = (self.sigma**2 / (4*self.a**3)) * (1 - np.exp(-self.a*(T-t)))**2 * (1 - np.exp(-2*self.a * t))
        return np.exp(term1 + term2 + term3)
    
    def zero_coupon_bond(self, t, T, r_t):
        """Price of zero-coupon bond at time t in order to make swaptions easier"""
        return self.A(t, T) * np.exp(-self.B(t, T) * r_t)
    
    def swaption_vol(self, expiry, tenor):
        """Calculate model swaption normal vol"""
        # implementation of jamshidian decomposition
        # simplified for brevity - full version requires swap rate calculation
        B_T = self.B(expiry, expiry + tenor)
        vol = self.sigma * B_T / np.sqrt(expiry)
        return vol
    
def calibration_objective(params, market_vols, expiries, tenors, discount_curve, weights=None):
    a, sigma = params
    if a <= 0 or sigma <= 0:
        return 1e6
    
    model = HullWhiteModel(a, sigma, discount_curve)
    model_vols = np.array([model.swaption_vol(Texp, Tten) for Texp, Tten in zip(expiries, tenors)])
    errors = (model_vols - market_vols) / market_vols
    rmse = np.sqrt(np.mean(errors**2))
    
    # Regularization
    penalty = 0.01 * (a**2 + sigma**2)
    return rmse + penalty

# Usage Example
if __name__ == "__main__":
    # Load market data
    here = Path(__file__).resolve().parent
    market_data = np.loadtxt(here / "swaption_vols.csv", delimiter=",")
    expiries = market_data[:,0]
    tenors = market_data[:,1]
    market_vols = market_data[:,2]
    
    # Example discount curve (Nelson-Siegel)
    def _phi(x):
        """Stable (1 - exp(-x)) / x with correct limit at x=0."""
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)
        small = np.isclose(x, 0.0)
        out[small] = 1.0
        out[~small] = (1.0 - np.exp(-x[~small])) / x[~small]
        return out

    def nelson_siegel(T, beta0=0.05, beta1=-0.02, beta2=-0.01, lambda0=0.5):
        T = np.asarray(T, dtype=float)
        x = lambda0 * T
        phi = _phi(x)
        return beta0 + beta1 * phi + beta2 * (phi - np.exp(-x))
    
    discount_curve = lambda T: np.exp(-nelson_siegel(T) * T)
    
    # initial guess
    x0 = [0.1, 0.01]
    
    # bounds
    bounds = [(0.01, 1.0), (0.001, 0.05)]
    
    # optimize
    result = minimize(calibration_objective, x0, args=(market_vols, expiries, tenors, discount_curve),
                      method='L-BFGS-B',
                      bounds=bounds,
                      options={'ftol': 1e-6, 'maxiter': 1000})
    
    print(f"Calibrated parameters: a = {result.x[0]:.4f}, sigma = {result.x[1]:.4f}")
    print(f"Final RMSE: {result.fun:.6f}")