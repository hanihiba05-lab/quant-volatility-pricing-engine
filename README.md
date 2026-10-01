# Multi-Asset Volatility Surface & Exotic Option Pricing Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen.svg)

## Overview

A production-grade, high-performance quantitative pricing and risk engine built in Python. This repository provides end-to-end capabilities for **Volatility Surface Calibration (SABR / SVI models)**, **Exotic Derivatives Pricing (Monte Carlo & Crank-Nicolson PDE)**, and **Real-Time Sensitivities Calculation (Greeks & Dynamic Delta-Gamma Hedging)**.

Engineered to benchmark market-implied volatility dynamics and price path-dependent options (Barrier Knock-Out, Bermudians) with institutional accuracy and variance reduction techniques.

---

## Key Features & Quantitative Highlights

* **Arbitrage-Free Volatility Surface Fitting:** Implements the **SABR Stochastic Volatility Model** ($\alpha, \beta, \rho, \nu$) calibrated to live market option chains (S&P 500 / Euro Stoxx 50) using non-linear L-BFGS-B optimization.
* **Dual-Engine Pricing Framework:**
  * **Vectorized Monte Carlo:** Path-dependent exotic option pricing featuring **Antithetic Variates** and **Control Variates** for rapid variance reduction ($O(1/\sqrt{N})$ speedup).
  * **Implicit PDE Solver:** Crank-Nicolson Finite Difference scheme for American and Bermudian early-exercise options.
* **Risk & Greeks Engine:** Exact analytical and numerical computation of Delta ($\Delta$), Gamma ($\Gamma$), Vega ($V$), Theta ($\Theta$), and Rho ($\rho$) with automated dynamic hedging simulation under transaction costs.
* **Interactive Quantitative Dashboard:** Streamlit UI displaying interactive 3D Volatility Surfaces, Monte Carlo convergence diagnostics, and Greeks profiles.

---

## Mathematical Formulation & Core Models

### 1. SABR Volatility Model Calibration
The SABR model assumes the forward asset price $F_t$ and its volatility $\alpha_t$ follow the stochastic differential equations:

$$dF_t = \alpha_t F_t^\beta dW_t^1$$

$$d\alpha_t = \nu \alpha_t dW_t^2$$

$$dW_t^1 dW_t^2 = \rho dt$$

The closed-form approximation for implied volatility $\sigma_{SABR}(K, f)$ is calibrated by minimizing the Sum of Squared Errors (SSE) against market-observed implied volatilities:

$$\min_{\alpha, \rho, \nu} \sum_{i=1}^{M} \left( \sigma_{market}(K_i, T_i) - \sigma_{SABR}(K_i, T_i; \alpha, \beta, \rho, \nu) \right)^2$$

### 2. Crank-Nicolson Finite Difference Scheme (PDE Engine)
Black-Scholes PDE discretized across spatial grid $S$ and time $t$:

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - rV = 0$$

Using Crank-Nicolson ($\theta = 0.5$) for unconditional stability ($O(\Delta t^2, \Delta S^2)$ accuracy), leading to the tridiagonal matrix system:

$$A \cdot V^{n} = B \cdot V^{n+1} + C$$

---

## Repository Structure

```plaintext
quant-volatility-pricing-engine/
├── app.py                      # Interactive 3D Volatility Surface & Pricing Dashboard
├── src/
│   ├── data_ingestion.py       # YFinance / Option Chains Ingestion & Preprocessing
│   ├── sabr_surface.py         # SABR Model Calibration & Interpolation
│   ├── monte_carlo_engine.py   # Vectorized Monte Carlo (Barriers, Bermudians, Variance Reduction)
│   ├── pde_solver.py           # Crank-Nicolson Finite Difference PDE Engine
│   └── greeks_hedging.py       # Numerical Greeks & Dynamic Delta-Gamma Hedging Simulator
├── tests/
│   └── test_pricing.py         # Unit tests validating MC vs PDE vs Black-Scholes benchmark
├── requirements.txt
└── README.md
