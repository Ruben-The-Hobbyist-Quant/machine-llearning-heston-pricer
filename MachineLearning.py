#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 14:47:15 2025

@author: rubenobadia
"""

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from MarketData import MarketData
import numpy as np

class MLApproach:
    def __init__(
        self,
        market_data: MarketData,
        hidden_sizes=(256, 256, 128),
        device=None
    ):
        self.md = market_data
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_sizes = hidden_sizes
        self.model = None

    # ---------- DATASET ----------
    def prepare_dataset(self, n_samples, test_ratio=0.2, **sim_kwargs):
        X, y = self.md.simulate_heston_surfaces(n_samples, **sim_kwargs)
    
        # Conversion en tensors
        X = torch.from_numpy(X).float().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)
    
        # Split train/test
        n_test = int(test_ratio * n_samples)
        self.X_train, self.X_test = X[:-n_test], X[-n_test:]
        self.y_train, self.y_test = y[:-n_test], y[-n_test:]
    
        # DataLoader d'entraînement
        train_dataset = TensorDataset(self.X_train, self.y_train)
        self.loader = DataLoader(train_dataset, batch_size=512, shuffle=True)



    # ---------- NETWORK ----------
    def build_model(self):
        layers = []
        in_dim = self.X_train.shape[1]
        for h in self.hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, 5)]            # (v0, θ, κ, σ, ρ)
        self.model = nn.Sequential(*layers).to(self.device)

    def train(self, epochs=50, lr=1e-3, return_losses=False):
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        train_losses = []
        test_losses = []
    
        for epoch in range(1, epochs + 1):
            self.model.train()
            running_train = 0.0
            for Xb, yb in self.loader:
                opt.zero_grad()
                pred = self.model(Xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                running_train += loss.item()
    
            avg_train_loss = running_train / len(self.loader)
            train_losses.append(avg_train_loss)
    
            # --- calcul de la loss sur le test set ---
            self.model.eval()
            with torch.no_grad():
                X_test, y_test = self.X_test, self.y_test
                y_pred = self.model(X_test)
                test_loss = loss_fn(y_pred, y_test).item()
                test_losses.append(test_loss)
    
            if epoch % 10 == 0:
                print(f"[{epoch}/{epochs}] train_loss = {avg_train_loss:.5f} | test_loss = {test_loss:.5f}")
    
        if return_losses:
            return train_losses, test_losses



    # ---------- PREDICT ----------
    @torch.no_grad()
    def predict_params(self, surface_IV):
        x = torch.tensor(surface_IV, dtype=torch.float32, device=self.device).unsqueeze(0)
        params_hat = self.model(x).cpu().detach().numpy().flatten()
    
        # Application de bornes réalistes sur les paramètres :
        bounds = {
            "v0":   (1e-4, 1.0),      # Variance initiale entre 0.0001 et 1
            "theta": (1e-4, 1.0),     # Variance long terme
            "kappa": (0.1, 10.0),     # Vitesse de retour à l'équilibre
            "sigma": (1e-4, 2.0),     # Vol de vol
            "rho":  (-1, 0.99),     # Corrélation
        }
    
        clipped_params = {
            k: float(np.clip(v, *bounds[k]))
            for k, v in zip(["v0", "theta", "kappa", "sigma", "rho"], params_hat)
        }
    
        return clipped_params