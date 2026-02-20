#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 14:46:45 2025

@author: rubenobadia
"""

from scipy.fftpack import ifft
from scipy.interpolate import interp1d
from functools import partial
import numpy as np

        	
def fft(K, S0, r, T, cf): # interp support cubic 
    """ 
    K = vector of strike
    S0 = spot price scalar
    cf = characteristic function
    """
    N=2**15                         # FFT more efficient for N power of 2
    B = 500                         # integration limit 
    
    dx = B/N
    x = np.arange(N) * dx
    
    weight = 3 + (-1)**(np.arange(N)+1) # Simpson weights
    weight[0] = 1; weight[N-1]=1
        
    dk = 2*np.pi/B
    b = N * dk /2
    ks = -b + dk * np.arange(N)
    
    integrand = np.exp(- 1j * b * np.arange(N)*dx) * cf(x - 0.5j) * 1/(x**2 + 0.25) * weight * dx/3
    integral_value = np.real( ifft(integrand)*N )
    spline_cub = interp1d(ks, integral_value, kind="cubic") # cubic will fit better than linear
    prices = S0 - np.sqrt(S0 * K) * np.exp(-r*T)/np.pi * spline_cub( np.log(S0/K) )
    
    return prices    
        

class HestonModel:
    
    def __init__(
        self,
        S0,                 # spot
        r,                  # taux sans risque
        kappa, theta, sigma, rho,  # paramètres Heston
        v0,                 # variance initiale
        T,                  # maturité (années)
        payoff="put"       # "call" ou "put"
    ):
        # --- paramètres de marché / process
        self.S0   = float(S0)
        self.r    = float(r)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.rho   = float(rho)
        self.v0    = float(v0)

        # --- paramètres option
        self.T      = float(T)   # maturité unique pour cette instance
        self.payoff = payoff.lower()  # type d’option
          

    # payoff function
    def payoff_f(self, S):
        if self.payoff == "call":
            Payoff = np.maximum( S - self.K, 0 )
        elif self.payoff == "put":    
            Payoff = np.maximum( self.K - S, 0 )  
        return Payoff
    
    # FFT method. It returns a vector of prices.
    def FFT(self, K): # K: strikes
        K = np.array(K)

        # Heston characteristic function (proposed by Schoutens 2004)
        def cf_Heston_good(u, t, v0, mu, kappa, theta, sigma, rho):
            xi = kappa - sigma*rho*u*1j
            d = np.sqrt( xi**2 + sigma**2 * (u**2 + 1j*u) )
            g1 = (xi+d)/(xi-d)
            g2 = 1/g1
            cf = np.exp( 1j*u*mu*t + (kappa*theta)/(sigma**2) * ( (xi-d)*t - 2*np.log( (1-g2*np.exp(-d*t))/(1-g2) ))\
                      + (v0/sigma**2)*(xi-d) * (1-np.exp(-d*t))/(1-g2*np.exp(-d*t)) )
            return cf
        
        cf_H_b_good = partial(cf_Heston_good, t=self.T, v0=self.v0, mu=self.r, theta=self.theta, 
                                  sigma=self.sigma, kappa=self.kappa, rho=self.rho)
        if self.payoff == "call":
            return fft(K, self.S0, self.r, self.T, cf_H_b_good)
        elif self.payoff == "put":        # put-call parity
            return fft(K, self.S0, self.r, self.T, cf_H_b_good) - self.S0 + K*np.exp(-self.r*self.T)

    def simulate_paths(self, N, M):
        dt = self.T / N
        S = np.zeros((M, N + 1))
        v = np.zeros((M, N + 1))

        S[:, 0] = self.S0
        v[:, 0] = self.v0

        for t in range(1, N + 1):
            Z1 = np.random.normal(size=M)
            Z2 = np.random.normal(size=M)
            W1 = Z1
            W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2

            vt = np.maximum(v[:, t-1], 0)  # Full truncation
            v[:, t] = v[:, t-1] + self.kappa * (self.theta - vt) * dt + self.sigma * np.sqrt(vt * dt) * W2
            v[:, t] = np.maximum(v[:, t], 0)  # Keep positivity

            S[:, t] = S[:, t-1] * np.exp((self.r - 0.5 * vt) * dt + np.sqrt(vt * dt) * W1)

        return S, v

    def price_PDI(self, K, H, N=100, M=100000):
        S, _ = self.simulate_paths( N, M)
        ST = S[:, -1]

        # Payoff du Put Down-and-In (activation de la barrière uniquement si S_T < H)
        payoff = np.where(ST < H, np.maximum(K - ST, 0), 0)
        discounted_payoff = np.exp(-self.r * self.T) * payoff
        return np.mean(discounted_payoff), np.std(discounted_payoff) / np.sqrt(M)


    def estimate_delta_PDI(self, K, H, eps=1, N=100, M=100000):
        original_S0 = self.S0
    
        # S0 + epsilon
        self.S0 = original_S0 + eps
        price_plus, _ = self.price_PDI(K, H, N, M)
    
        # S0 - epsilon
        self.S0 = original_S0 - eps
        price_minus, _ = self.price_PDI(K, H, N, M)
    
        # Restaure le S0 original
        self.S0 = original_S0
    
        # Approximation du delta par différences centrées
        delta = (price_plus - price_minus) / (2 * eps)
        return delta

