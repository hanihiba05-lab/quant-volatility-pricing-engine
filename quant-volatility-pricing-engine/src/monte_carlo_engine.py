import numpy as np

def monte_carlo_barrier_option(S0, K, T, r, sigma, barrier, option_type='call', barrier_type='out', num_sims=100000, num_steps=252):
    """
    Vectorized Monte Carlo engine with Antithetic Variates for Path-Dependent Barrier Options.
    """
    dt = T / num_steps
    half_sims = num_sims // 2
    
    # Generate antithetic standard normal random variables
    Z = np.random.normal(0, 1, size=(half_sims, num_steps))
    Z = np.vstack((Z, -Z))
    
    # Simulate geometric Brownian motion asset paths
    drift = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    price_paths = S0 * np.exp(np.cumsum(drift + diffusion, axis=1))
    
    # Prepend initial spot price
    price_paths = np.hstack((np.full((num_sims, 1), S0), price_paths))
    
    # Barrier condition checking
    if barrier_type == 'out':
        active_paths = np.all(price_paths < barrier, axis=1) if option_type == 'call' else np.all(price_paths > barrier, axis=1)
    else:
        active_paths = np.any(price_paths >= barrier, axis=1)
        
    ST = price_paths[:, -1]
    payoffs = np.maximum(ST - K, 0) if option_type == 'call' else np.maximum(K - ST, 0)
    payoffs = payoffs * active_paths
    
    price = np.exp(-r * T) * np.mean(payoffs)
    stderr = np.exp(-r * T) * np.std(payoffs) / np.sqrt(num_sims)
    return price, stderr