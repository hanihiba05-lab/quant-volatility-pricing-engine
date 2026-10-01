# Multi-Asset Volatility Surface & Exotic Option Pricing Engine

A Python library for calibrating implied volatility surfaces and pricing exotic options using Monte Carlo simulations and Finite Difference implicit methods.

---

## What It Does

This repository provides tools to fit market volatility surfaces and price path-dependent or early-exercise options:

* **SABR Volatility Surface:** Fits Hagan's SABR model ($\alpha, \beta, \rho, \nu$) to market option chains (S&P 500 / Euro Stoxx 50) using L-BFGS-B optimization.
* **Vectorized Monte Carlo Engine:** Prices path-dependent options (such as Knock-Out Barriers) using antithetic variates to reduce variance.
* **Crank-Nicolson PDE Solver:** Solves the Black-Scholes PDE using an implicit finite difference scheme for options with early-exercise features.
* **Greeks & Risk Analysis:** Calculates Delta, Gamma, Vega, Theta, and Rho numerically to simulate dynamic hedging under transaction costs.
* **Dashboard:** Streamlit interface to visualize 3D volatility surfaces and test pricing models.

---

## Model Setup & Math

### 1. SABR Model Calibration
The asset price $F_t$ and volatility $\alpha_t$ follow:

$$dF_t = \alpha_t F_t^\beta dW_t^1$$

$$d\alpha_t = \nu \alpha_t dW_t^2$$

$$dW_t^1 dW_t^2 = \rho dt$$

Parameters ($\alpha, \rho, \nu$) are fitted by minimizing the sum of squared errors between market implied volatilities and SABR closed-form approximations:

$$\min_{\alpha, \rho, \nu} \sum_{i=1}^{M} \left( \sigma_{market}(K_i, T_i) - \sigma_{SABR}(K_i, T_i; \alpha, \beta, \rho, \nu) \right)^2$$

### 2. Crank-Nicolson PDE Scheme
The Black-Scholes PDE is discretized over space ($S$) and time ($t$):

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - rV = 0$$

Setting $\theta = 0.5$ gives an unconditionally stable scheme with $O(\Delta t^2, \Delta S^2)$ convergence, solved via a tridiagonal matrix system:

$$A \cdot V^{n} = B \cdot V^{n+1} + C$$

---

## Directory Structure

```plaintext
quant-volatility-pricing-engine/
├── app.py                      # Streamlit 3D surface & pricing UI
├── src/
│   ├── data_ingestion.py       # Option chain data retrieval
│   ├── sabr_surface.py         # SABR calibration routines
│   ├── monte_carlo_engine.py   # Vectorized Monte Carlo simulation
│   ├── pde_solver.py           # Crank-Nicolson PDE implementation
│   └── greeks_hedging.py       # Numerical Greeks and hedging logic
├── tests/
│   └── test_pricing.py         # Unit tests against analytical benchmarks
├── requirements.txt
└── README.md
