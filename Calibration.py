#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Surface IV – Yahoo vs Calibrée (Black-Scholes)
@author: rubenobadia
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import griddata
from BSModel import BlackScholesModel
from HestonModel import HestonModel
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize



def Feller(x):
    return 2 * x[3] * x[2] - x[1]**2 - 1e-6

cons = ({"fun": Feller, "type": "ineq"})



class Calibration:
    def __init__(self, market_data, bs_model_class, tol=1e-10):
        self.market_data = market_data
        self.bs_model_class = bs_model_class
        self.tol = tol
        self.bs_surface = None
        self.heston_params = None

    # ------------------------------------------------------------------
    # 1. CALIBRATION BS + TABLE
    # ------------------------------------------------------------------
    def calibrate_bs(self):
        rows = []
        for _, row in self.market_data.iterrows():
            S, K, days, r, price, opt_type, iv_yahoo = (
                row["S0"], row["K"], row["days_to_expiry"],
                row["r"], row["price"], row["type"],
                row["implied_vol_yahoo"]
            )
            T = days / 365  # BS attend T en années

            def objective(sigma):
                mdl = self.bs_model_class(S0=S, r=r, sigma=sigma)
                return mdl.price(K=K, T=T, opt_type=opt_type) - price

            try:
                iv = brentq(objective, 0.001, 2, xtol=self.tol)
                price_model = self.bs_model_class(S0=S, r=r, sigma=iv).price(
                    K=K, T=T, opt_type=opt_type
                )
            except (ValueError, RuntimeError):
                iv = np.nan
                price_model = np.nan


            rows.append({
                "K": K,
                "days_to_expiry": days,
                "T_years": round(T, 5),
                "sigma_calib": iv,
                "sigma_yahoo": iv_yahoo,
                #"err_vol_%": err_rel,
                "price_mkt": price,
                "price_model": price_model
            })

        self.bs_surface = pd.DataFrame(rows)
        return self.bs_surface

    # ------------------------------------------------------------------
    # 2. PLOT DES DEUX SURFACES
    # ------------------------------------------------------------------
    def plot_vol_surfaces_bs(self,surface):
        if surface is None:
            raise RuntimeError("Calibre d’abord la surface !")

        df = surface.dropna(subset=["sigma_calib", "sigma_yahoo"])
        if df.empty:
            print("[INFO] Aucune donnée valide pour tracer la surface.")
            return

        X, Y = df["K"], df["days_to_expiry"]
        Z2 = df["sigma_calib"]

        # Grille régulière pour interpolation
        xi = np.linspace(X.min(), X.max(), 150)
        yi = np.linspace(Y.min(), Y.max(), 150)
        Xi, Yi = np.meshgrid(xi, yi)
    
        # Interpolation sur la grille
        Zi_calib = griddata((X, Y), Z2, (Xi, Yi), method='linear')
        Zi_calib_smooth = gaussian_filter(Zi_calib, sigma=0.8)


        # Tracé
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection="3d")


        ax.plot_surface(Xi, Yi, Zi_calib_smooth, color='orange', alpha=0.6)

        ax.set_xlabel("Strike K")
        ax.set_ylabel("Jours avant maturité")
        ax.set_zlabel("Volatilité implicite")
        ax.set_title("Surfaces IV – Yahoo (bleu) vs Calibrée BS (orange)")

        plt.tight_layout()
        plt.show()
    
    


    def _heston_objective(self, x):
        """
        x = [rho, log_sigma, log_theta, log_kappa, log_v0]  (désaturé)
        On travaille en log pour garantir la positivité et la stabilité.
        """
        rho       = np.tanh(x[0])          # contraint naturellement [-1,1]
        sigma     = np.exp(x[1])
        theta     = np.exp(x[2])
        kappa     = np.exp(x[3])
        v0        = np.exp(x[4])

        # Feller 2κθ < σ²
        pen_feller = max(0, 1e3 * (sigma**2 - 2*kappa*theta))

        err  = 0.0
        pvec = np.empty(len(self.market_data))

        # --------- groupé par maturité ---------
        for d, grp in self.market_data.groupby("days_to_expiry"):
            T  = d / 252.0                              # convention jours de marché
            Ks = grp["K"].to_numpy()
            S0 = grp["S0"].iloc[0]
            r  = grp["r"].iloc[0]
            payoff = grp["type"].iloc[0]
            
            model = HestonModel(S0=S0, r=r, kappa=kappa, theta=theta,
                            sigma=sigma, rho=rho, v0=v0,
                            T=T, payoff=payoff)

            try:
                prices = model.FFT(Ks)
            except Exception:
                return 1e10  # crash num.

            pvec[grp.index] = prices
            diff = (prices - grp["price"].to_numpy()) / np.maximum(grp["spread"].to_numpy(), 1e-4)
            err += np.sum(diff**2)
        self._last_price_vector = pvec            # pour export

        return err + pen_feller

    # --------------------------------------------------------
    # b)  calibrateur
    # --------------------------------------------------------
    def run_heston_calibration(self, init_vals):
        # on passe les init en log (sauf rho)
        x0 = [np.arctanh(init_vals[0]),
              np.log(init_vals[1]),
              np.log(init_vals[2]),
              np.log(init_vals[3]),
              np.log(init_vals[4])]

        res = minimize(self._heston_objective, x0,
                       method="L-BFGS-B",
                       options={"maxiter": 800, "ftol": 1e-6})

        if not res.success:
            print("[ERREUR CALIBRATION]", res.message)
            raise RuntimeError("Non convergé :", res.message)


        # décode les params calibrés
        rho   = np.tanh(res.x[0])
        sigma = np.exp(res.x[1])
        theta = np.exp(res.x[2])
        kappa = np.exp(res.x[3])
        v0    = np.exp(res.x[4])

        self.heston_params = {"rho":rho,"sigma":sigma,"theta":theta,"kappa":kappa,"v0":v0}
        self.market_data["price_heston"] = self._last_price_vector
        return self.heston_params
    
    def plot_vol_surfaces_heston(self,surface):
        if self.bs_surface is None:
            raise RuntimeError("Calibre d’abord la surface !")

        df = surface.dropna(subset=["sigma_calib", "sigma_yahoo"])
        if df.empty:
            print("[INFO] Aucune donnée valide pour tracer la surface.")
            return

        X, Y = df["K"], df["days_to_expiry"]
        Z2 = df["sigma_calib"]

        # Grille régulière pour interpolation
        xi = np.linspace(X.min(), X.max(), 50)
        yi = np.linspace(Y.min(), Y.max(), 50)
        Xi, Yi = np.meshgrid(xi, yi)
    
        # Interpolation sur la grille
        Zi_calib = griddata((X, Y), Z2, (Xi, Yi), method='linear')
        Zi_calib_smooth = gaussian_filter(Zi_calib, sigma=0.8)
        


        # Tracé
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection="3d")

        

        ax.plot_surface(Xi, Yi, Zi_calib_smooth, color='orange', alpha=0.6)

        ax.set_xlabel("Strike K")
        ax.set_ylabel("Jours avant maturité")
        ax.set_zlabel("Volatilité implicite Heston")
        ax.set_title("Surfaces IV – Yahoo (bleu) vs Calibrée Heston (orange)")

        plt.tight_layout()
        plt.show()

    def plot_vol_line_comparison(self, maturity_days, surface1, surface2,
                             label1="IV surface 1", label2="IV surface 2"):
        """
        Compare l'IV entre deux surfaces pour une même maturité, en fonction de la moneyness.
        """
        def get_surface_at_maturity(surface, maturity_days):
            df = surface[surface["days_to_expiry"] == maturity_days]
            if df.empty:
                closest = surface["days_to_expiry"].sub(maturity_days).abs().idxmin()
                near_mat = int(surface.loc[closest, "days_to_expiry"])
                print(f"[INFO] Maturité {maturity_days}j introuvable – utilisation de {near_mat}j.")
                df = surface[surface["days_to_expiry"] == near_mat]
                return df.sort_values("K"), near_mat
            return df.sort_values("K"), maturity_days

        df1, used_mat1 = get_surface_at_maturity(surface1, maturity_days)
        df2, used_mat2 = get_surface_at_maturity(surface2, maturity_days)
    
        S0 = self.market_data["S0"].iloc[0]

        # Calcul de la moneyness
        df1["moneyness"] = df1["K"] / S0
        df2["moneyness"] = df2["K"] / S0

        # Filtrage moneyness entre 0.5 et 1.5
        df1 = df1[(df1["moneyness"] >= 0.6) & (df1["moneyness"] <= 1.4)]
        df2 = df2[(df2["moneyness"] >= 0.6) & (df2["moneyness"] <= 1.4)]

        plt.figure(figsize=(9, 5))

        # Heston (surface1)
        plt.plot(df1["moneyness"], df1["sigma_calib"], 's--', label=label1)

        # BS (surface2) avec lissage
        sorted_df2 = df2.sort_values("moneyness")
        moneyness_sorted = sorted_df2["moneyness"]
        iv_sorted = sorted_df2["sigma_calib"].values
        plt.plot(moneyness_sorted, iv_sorted, 'o-', label=label2 )

        plt.xlabel("Moneyness $K/S_0$")
        plt.ylabel("Volatilité implicite")
        plt.title(f"Comparaison des IV – maturité {used_mat1} jours")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
