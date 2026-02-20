# machine-llearning-heston-pricer
Python implementation of the Heston stochastic volatility model with FFT-based option pricing and machine-learning surrogate model for Reverse Convertible structured products.

Run:
- install requirements
- python Interface.py

Implementation developed as part of a Master’s thesis in Quantitative Finance (Université Paris 1 Panthéon-Sorbonne).

This repository provides a pricing framework for **Reverse Convertible structured products** based on the **Heston stochastic volatility model**, with a **machine-learning surrogate** enabling fast valuation once calibrated.

---

## Overview

Reverse Convertibles are equity-linked structured products whose valuation depends critically on volatility dynamics.  
Standard constant-volatility models fail to capture market smiles and skew.

This project:

- calibrates the **Heston stochastic volatility model** to market option data  
- prices the embedded option component via **FFT (Carr–Madan)**  
- trains a **machine-learning surrogate model** to approximate product prices  
- enables fast pricing across market scenarios  

---

## Methodology

The pricing framework follows four stages.

### 1. Market Data Processing
- Equity spot and rates retrieval  
- Implied volatility inputs  
- Data preprocessing  

### 2. Heston Calibration
Model parameters are calibrated to market option prices:

κ, θ, ξ, ρ, v₀  

Calibration minimizes the error between model and market prices.

### 3. Reverse Convertible Pricing
The payoff is decomposed into:

Bond component + Short European put  

The embedded option is priced under Heston dynamics using **FFT pricing**.

### 4. Machine-Learning Surrogate
A regression model learns the mapping:

(market state, Heston parameters) → product price  

Once trained, pricing becomes near-instantaneous.

---

## Heston Model

Stochastic volatility dynamics:

dSₜ = μSₜ dt + √vₜ Sₜ dWₜ¹  
dvₜ = κ(θ − vₜ) dt + ξ√vₜ dWₜ²  
corr(dW¹, dW²) = ρ  

European option pricing uses the **Carr–Madan Fourier transform**.

---

## Project Structure
-  BSModel.py : Black-Scholes benchmark pricing
-  HestonModel.py : Heston characteristic function and FFT pricing
-  Calibration.py : Heston parameter calibration
-  MarketData.py : Market data ingestion and preprocessing
-  MachineLearning.py ML surrogate pricing model
-  Interface.py : Workflow interface
-  main.py : Entry point


---

## Requirements

Python 3.9+

Libraries:

- numpy
- scipy
- pandas
- matplotlib
- yfinance
- torch

Install:
pip install -r requirements.txt

---

## Usage

Run the interface:

python Interface.py

---

## Thesis

This repository accompanies the Master’s thesis:

**“Heston Model Calibration with Machine Learning for Reverse Convertible Pricing”**  
Ruben Obadia — Université Paris 1 Panthéon-Sorbonne  

📄 [Read the full thesis (PDF)](docs/Obadia_Ruben_Heston_ML_RC_Thesis.pdf)

---


## Author

Ruben Obadia  
Master in Quantitative Finance  
Université Paris 1 Panthéon-Sorbonne
