import numpy as np

def compute_numerical_greeks(pricing_func, S0, K, T, r, sigma, h=1e-2):
    """
    Computes numerical Greeks (Delta, Gamma, Vega) via central finite differences.
    """
    V = pricing_func(S0, K, T, r, sigma)
    V_up_S = pricing_func(S0 + h, K, T, r, sigma)
    V_down_S = pricing_func(S0 - h, K, T, r, sigma)
    
    delta = (V_up_S - V_down_S) / (2 * h)
    gamma = (V_up_S - 2 * V + V_down_S) / (h ** 2)
    
    V_up_vol = pricing_func(S0, K, T, r, sigma + h)
    V_down_vol = pricing_func(S0, K, T, r, sigma - h)
    vega = (V_up_vol - V_down_vol) / (2 * h)
    
    return {'Delta': delta, 'Gamma': gamma, 'Vega': vega}