import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.data_ingestion import fetch_option_chain
from src.sabr_surface import sabr_implied_volatility, calibrate_sabr
from src.monte_carlo_engine import monte_carlo_barrier_option
from src.pde_solver import crank_nicolson_pricing

st.set_page_config(page_title="Quantitative Volatility Engine", layout="wide")

st.title("Multi-Asset Volatility Surface & Exotic Option Pricing Engine")

st.sidebar.header("Pricing & Surface Parameters")
spot_price = st.sidebar.number_input("Spot Price (S0)", value=100.0)
strike_price = st.sidebar.number_input("Strike Price (K)", value=100.0)
ttm = st.sidebar.slider("Time to Maturity (Years)", 0.1, 2.0, 1.0)
r = st.sidebar.slider("Risk-Free Rate", 0.0, 0.10, 0.05)

tab1, tab2 = st.tabs(["3D Volatility Surface (SABR)", "Exotic Option Pricing Engine"])

with tab1:
    st.subheader("SABR Volatility Surface Model Calibration")
    strikes = np.linspace(spot_price * 0.7, spot_price * 1.3, 15)
    maturities = np.linspace(0.1, 2.0, 10)
    K_grid, T_grid = np.meshgrid(strikes, maturities)
    
    # Synthetic surface generation using SABR params
    alpha, beta, rho, nu = 0.2, 0.5, -0.3, 0.4
    Z_vols = np.array([[sabr_implied_volatility(spot_price, k, t, alpha, beta, rho, nu) 
                       for k in strikes] for t in maturities])
    
    fig = go.Figure(data=[go.Surface(z=Z_vols, x=K_grid, y=T_grid, colorscale='Viridis')])
    fig.update_layout(title="Arbitrage-Free Implied Volatility Surface", 
                      scene=dict(xaxis_title="Strike (K)", yaxis_title="TTM (Years)", zaxis_title="Implied Vol"))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Path-Dependent Exotic Option Pricing (Monte Carlo vs PDE)")
    barrier = st.sidebar.number_input("Barrier Level", value=120.0)
    
    mc_price, stderr = monte_carlo_barrier_option(spot_price, strike_price, ttm, r, 0.20, barrier)
    pde_price = crank_nicolson_pricing(spot_price, strike_price, ttm, r, 0.20)
    
    col1, col2 = st.columns(2)
    col1.metric("Monte Carlo Knock-Out Call Price", f"${mc_price:.4f}", f"StdErr: ±{stderr:.4f}")
    col2.metric("Crank-Nicolson PDE Vanilla Price", f"${pde_price:.4f}")