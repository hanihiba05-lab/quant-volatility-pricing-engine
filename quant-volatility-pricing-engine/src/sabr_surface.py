import numpy as np
from scipy.optimize import minimize

def sabr_implied_volatility(F, K, T, alpha, beta, rho, nu):
    """
    Hagan's closed-form SABR implied volatility approximation.
    """
    if F == K:
        first_term = (alpha / (F ** (1 - beta)))
        second_term = 1 + (((1 - beta) ** 2 / 24) * (alpha ** 2 / (F ** (2 - 2 * beta))) +
                           (1 / 4) * (rho * beta * nu * alpha / (F ** (1 - beta))) +
                           ((2 - 3 * rho ** 2) / 24) * (nu ** 2)) * T
        return first_term * second_term

    log_FK = np.log(F / K)
    FK_beta = (F * K) ** ((1 - beta) / 2)
    z = (nu / alpha) * FK_beta * log_FK
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

    num = alpha * (1 + (((1 - beta) ** 2 / 24) * (alpha ** 2 / (FK_beta ** 2)) +
                        (1 / 4) * (rho * beta * nu * alpha / FK_beta) +
                        ((2 - 3 * rho ** 2) / 24) * (nu ** 2)) * T)
    den = FK_beta * (1 + ((1 - beta) ** 2 / 24) * (log_FK ** 2) + ((1 - beta) ** 4 / 1920) * (log_FK ** 4))
    
    return (num / den) * (z / x_z)

def calibrate_sabr(forward, strikes, market_vols, T, beta=0.5):
    """
    Calibrates SABR parameters (alpha, rho, nu) to market implied volatilities.
    """
    def objective(params):
        alpha, rho, nu = params
        if alpha <= 0 or abs(rho) >= 1 or nu <= 0:
            return 1e6
        model_vols = [sabr_implied_volatility(forward, K, T, alpha, beta, rho, nu) for K in strikes]
        return np.sum((np.array(model_vols) - np.array(market_vols)) ** 2)

    initial_guess = [0.2, 0.0, 0.3]
    bounds = [(1e-4, 2.0), (-0.999, 0.999), (1e-4, 5.0)]
    res = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    return res.x # Returns [alpha, rho, nu]