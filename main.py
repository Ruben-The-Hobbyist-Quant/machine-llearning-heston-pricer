#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 14:47:42 2025

@author: rubenobadia
"""

from BSModel import BlackScholesModel
from HestonModel import HestonModel
from MarketData import MarketData
from Calibration import Calibration
from MachineLearning import MLApproach
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import time

class Main:

    def __init__(self, ticker="AMZN", r=0.05, option_type="put"):
        self.md_BS = MarketData(ticker=ticker, r=r, option_type=option_type)
        self.md_Heston = MarketData(ticker=ticker, r=r, option_type=option_type)
        self.df_market_BS = self.md_BS.load()
        self.df_market_Heston = self.md_Heston.load()
        self.calibrator_BS = Calibration(self.df_market_BS, BlackScholesModel)
        self.calibrator_Heston = Calibration(self.df_market_Heston, BlackScholesModel)

        
        
        
    def calibrate_heston(self, init_vals):
        params = self.calibrator_Heston.run_heston_calibration(init_vals)
        print("\n=== Paramètres calibrés ===")
        print(params)
        return params

    def table_price_vs_heston(self):
        print("\n=== Extrait des prix marché vs Heston ===")
        return self.calibrator_Heston.market_data[["S0", "days_to_expiry", "K", "r", "price", "price_heston"]].head(50)

    def calibrate_bs_surface(self):
        surface_bs = self.calibrator_BS.calibrate_bs()
        print("\n=== Extrait de la surface BS calibrée ===")
        print(surface_bs.dropna().head(50))
        return surface_bs

    def plot_vol_surface_BS(self):
        surface_BS = self.calibrator_BS.calibrate_bs()
        self.calibrator_BS.plot_vol_surfaces_bs(surface_BS)

    def plot_vol_surface_Heston(self):
        self.df_market_Heston["price"] = self.calibrator_Heston.market_data["price_heston"]
        surface_heston = self.calibrator_Heston.calibrate_bs()
        self.calibrator_Heston.plot_vol_surfaces_heston(surface_heston)

    def plot_vol_skew_heston(self, maturity_days):
        # Calibrage BS : création de la surface BS si nécessaire
        surface_BS = self.calibrator_BS.calibrate_bs()
        
        # Injection du prix Heston dans les données marché pour calibrage BS sur ces prix
        self.df_market_Heston["price"] = self.calibrator_Heston.market_data["price_heston"]
        
        # Calibrage BS sur les prix Heston pour obtenir surface Heston
        surface_heston = self.calibrator_Heston.calibrate_bs()
        print(surface_heston.head(50))
        # Affichage du skew (volatilité implicite) comparée entre les deux surfaces
        self.calibrator_Heston.plot_vol_line_comparison(
            maturity_days=maturity_days,
            surface1=surface_heston,
            surface2=surface_BS,
            label1="IV Heston calibrée",
            label2="IV BS calibrée"
            )

    def calibrate_heston_ml(self,ml):
        # --- construction dataset simulé
        #ml = MLApproach(self.md_Heston)
        #ml.prepare_dataset(n_samples=5)   # ⇦ on peut monter sans problème
        #ml.build_model()
        #ml.train(epochs=150)

        # --- surface réelle (Yahoo) à la même grilles
        #   1) on groupe par maturité la plus proche et on interpole aux mêmes moneyness
        surface_iv_market = self._surface_iv_market_same_grid()
        params_ml = ml.predict_params(surface_iv_market)
        print(surface_iv_market)
        print("\n=== Paramètres prédits par NN ===")
        for k, v in params_ml.items():
            print(f"{k:6s} : {v:.5f}")

        return params_ml

    # ---- helper pour mettre la surface du marché sur la même grille (exemple rapide)
    def _surface_iv_market_same_grid(self):
        df = self.df_market_Heston.copy()
        # calcul moneyness
        df["mny"] = df["K"] / df["S0"]
        # interpolation bi-linéaire K/S0 × T vers la grille simulée
        grid_KS = np.linspace(0.7, 1.3, 15)
        T_min = max(0.1, df["days_to_expiry"].min() / 365.0)
        T_max = min(1.0, df["days_to_expiry"].max() / 365.0)
        grid_T = np.linspace(T_min, T_max, 10)
        iv_matrix = griddata(
            (df["mny"], df["days_to_expiry"]/365.0),
            df["implied_vol_yahoo"],
            (grid_KS[:, None], grid_T[None, :]),
            method="linear"
        )
        return iv_matrix.ravel()
    
    def predict_on_market_surface(self, trained_model, target_maturity):
        """
        Utilise un modèle ML entraîné pour prédire des paramètres Heston sur une nouvelle surface IV
        et compare les prix obtenus avec ceux du marché pour une maturité donnée.
        
        Paramètres
        ----------
        trained_model : modèle entraîné (type MLApproach)
        target_maturity : float
            Maturité (en années) à isoler pour le tracé du graphe.
        """
        # Interpolation de la surface de marché réelle sur la grille simulée
        surface_iv_market = self._surface_iv_market_same_grid()
        print(surface_iv_market)
        #clean_surface = pd.DataFrame(surface_iv_market).dropna()
        #surface_clean = clean_surface.to_numpy()
        # Prédiction des paramètres Heston à partir de cette surface
        params_pred = trained_model.predict_params(surface_iv_market)
    
        print("\n=== Paramètres prédits par modèle pré-entraîné ===")
        for k, v in params_pred.items():
            print(f"{k:6s} : {v:.5f}")
    
        # Repricing avec modèle Heston FFT
        strikes = self.df_market_Heston["K"].values
        maturities = self.df_market_Heston["days_to_expiry"].values / 365
        prices_nn = []
    
        for K_i, T_i in zip(strikes, maturities):
            hm = HestonModel(
                S0=self.df_market_Heston["S0"].iloc[0],
                r=0.05,
                kappa=params_pred["kappa"],
                theta=params_pred["theta"],
                sigma=params_pred["sigma"],
                rho=params_pred["rho"],
                v0=params_pred["v0"],
                T=T_i,
                payoff="put"
            )
            price = hm.FFT(K_i)
            prices_nn.append(price)
    
        self.df_market_Heston["price_nn"] = prices_nn
    
        # Filtrer uniquement les lignes avec la maturité cible (tolérance faible)
        tol = 1e-4
        mask = np.abs(maturities - target_maturity) < tol
        df_filtered = self.df_market_Heston.loc[mask]
    
        if df_filtered.empty:
            print(f"Aucun point trouvé pour la maturité {target_maturity} ans.")

    
        # Tracé graphique
        plt.figure(figsize=(8, 6))
        plt.plot(df_filtered["K"], df_filtered["price"], label="Prix marché", marker='o')
        plt.plot(df_filtered["K"], df_filtered["price_nn"], label="Prix prédit (Heston NN)", marker='x')
        plt.xlabel("Strike")
        plt.ylabel("Prix")
        plt.title(f"Comparaison prix marché vs Heston (maturité {target_maturity:.2f} an)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
        return params_pred


    def compare_losses(self, sample_list, epochs=150):
        """
        Compare la convergence des pertes d'entraînement et de test pour différents n_samples.
        
        Args:
            sample_list (list): Liste de 3 tailles de jeu d'entraînement, ex: [100, 1000, 5000]
            epochs (int): Nombre d'epochs pour chaque entraînement
        """

    
        assert len(sample_list) == 3, "Tu dois passer exactement 3 tailles de n_samples."
    
        train_losses_all = []
        test_losses_all = []
    
        for n in sample_list:
            print(f"\n🔧 Entraînement avec n_samples = {n}...")
            ml = MLApproach(self.md_Heston)
            ml.prepare_dataset(n_samples=n)
            ml.build_model()
            train_losses, test_losses = ml.train(epochs=epochs, return_losses=True)
            train_losses_all.append(train_losses)
            test_losses_all.append(test_losses)
    
        # 📈 Tracé des courbes
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
        for i, losses in enumerate(train_losses_all):
            axs[0].plot(losses, label=f"{sample_list[i]} samples")
        axs[0].set_title("Train loss vs epochs")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("MSE Loss")
        axs[0].legend()
        axs[0].grid(True)
    
        for i, losses in enumerate(test_losses_all):
            axs[1].plot(losses, label=f"{sample_list[i]} samples")
        axs[1].set_title("Test loss vs epochs")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("MSE Loss")
        axs[1].legend()
        axs[1].grid(True)
    
        plt.suptitle("Comparaison train/test loss pour différents n_samples")
        plt.tight_layout()
        plt.show()



   