#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 14:45:28 2025

@author: rubenobadia
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

class BlackScholesModel:
    
    def __init__(self, S0, r, sigma=0.2):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        
    
    def price(self, K, T, opt_type="call"):
        if T <= 0:
            return max(0.0, (self.S0 - K) if opt_type == "call" else (K - self.S0))

        d1 = (np.log(self.S0 / K) + (self.r + 0.5 * self.sigma**2) * T) / (self.sigma * np.sqrt(T))
        d2 = d1 - self.sigma * np.sqrt(T)

        if opt_type == "call":
            return self.S0 * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
        else:  # put
            return K * np.exp(-self.r * T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)
    
    def implied_vol(self, market_price, K, T, opt_type="call"):
        
        def objective(sigma):
            self.sigma = sigma
            return self.price(K, T, opt_type) - market_price
        
        try:
            iv = brentq(objective, 1e-4, 2, xtol = 1e-6)
        except (ValueError, RuntimeError):
            iv = np.nan
        
        return iv


    def simulate_paths(self, T, N, n_paths):
        # Simule des trajectoires selon BS
        pass

    def price_PDI(self, K, H, T, opt_type='put'):
        if opt_type != 'put':
            raise ValueError("Seul le Put Down-and-In est implémenté avec cette formule.")



        # d_H- et d_H+ selon la formule de ton image
        dH_minus = (np.log(self.S0 / H) + (self.r - 0.5 * self.sigma**2) * T) / (self.sigma * np.sqrt(T))
        dH_plus = dH_minus + self.sigma * np.sqrt(T)

        # Formule du prix du Put Down-and-In
        PDI = K * np.exp(-self.r * T) * norm.cdf(-dH_minus) - self.S0 * norm.cdf(-dH_plus)
        return PDI
    
    def delta_PDI_BS(self, K, H, T):
        """
        Calcule le delta du Put Down-and-In européen sous Black-Scholes (formule analytique)
        """
    
        dH_minus = (np.log(self.S0 / H) + (self.r - 0.5 * self.sigma**2) * T) / (self.sigma * np.sqrt(T))
        dH_plus = dH_minus + self.sigma * np.sqrt(T)
        
        # N(-dH_plus) et N'(dH_plus)
        N_neg_dH_plus = norm.cdf(-dH_plus)
        Nprime_dH_plus = norm.pdf(dH_plus)
    
        # Formule du delta
        delta = -N_neg_dH_plus + (1 - K / H) * (Nprime_dH_plus / (self.sigma * np.sqrt(T)))
        return delta
    
    def delta_put_BS(self, K, T):
        """
        Calcule le delta d'un put européen classique sous Black-Scholes
        """
        d1 = (np.log(self.S0 / K) + (self.r + 0.5 * self.sigma**2) * T) / (self.sigma * np.sqrt(T))
        delta = norm.cdf(d1) - 1
        return delta

