#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 16:51:26 2025

@author: rubenobadia
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import time
from BSModel import BlackScholesModel
from HestonModel import HestonModel
from MachineLearning import MLApproach
from main import Main
from scipy.interpolate import griddata


class ReverseConvertibleApp(tk.Tk):
    def __init__(self, tickers_dict):
        super().__init__()
        self.title("Reverse Convertible Pricer")
        self.geometry("950x850")
        self.configure(bg="white")

        self.tickers_dict = tickers_dict
        self.model_var = tk.StringVar(value="Black-Scholes")
        self.calibration_var = tk.StringVar(value="ML")
        self.freq_var = tk.StringVar(value="Trimestrielle")

        self.selected_tickers = {}
        self.selected_barriers = {}
        self.selected_maturities = {}

        self._build_interface()

    def _build_interface(self):
        def make_section(title):
            lf = tk.LabelFrame(self, text=title, font=('Arial', 12, 'bold'), bg="white", fg="black")
            lf.pack(fill="x", padx=15, pady=10)
            return lf

        def arrange_checkboxes(frame, items_dict, storage_dict, items_per_col=4, prefix=''):
            cols = (len(items_dict) + items_per_col - 1) // items_per_col
            for col in range(cols):
                sub_frame = tk.Frame(frame, bg="white")
                sub_frame.pack(side="left", padx=10)
                for i, (key, label) in enumerate(list(items_dict.items())[col * items_per_col: (col + 1) * items_per_col]):
                    var = tk.BooleanVar()
                    txt = f"{prefix}{key} ({label})" if label else f"{prefix}{key}"
                    tk.Checkbutton(sub_frame, text=txt, variable=var, bg="white").pack(anchor="w")
                    storage_dict[key] = var

        # ==== Actions ====
        lf_tickers = make_section("Sélection des sous-jacents")
        arrange_checkboxes(lf_tickers, self.tickers_dict, self.selected_tickers, items_per_col=4)

        # ==== Maturités ====
        lf_matu = make_section("Choix des maturités")
        for mat in ['3 mois', '6 mois', '1 an', '18 mois']:
            var = tk.BooleanVar()
            tk.Checkbutton(lf_matu, text=mat, variable=var, bg="white").pack(anchor="w")
            self.selected_maturities[mat] = var

        # ==== Barrières ====
        lf_barriers = make_section("🛡 Barrières de protection")
        barrier_dict = {f"{b}%": '' for b in range(40, 95, 5)}
        arrange_checkboxes(lf_barriers, barrier_dict, self.selected_barriers, items_per_col=4)
        
        # ==== Fréquence des coupons ====
        lf_freq = make_section("📅 Fréquence des coupons")
        for freq in ["Trimestrielle", "Semestrielle", "Annuelle"]:
            tk.Radiobutton(lf_freq, text=freq, variable=self.freq_var, value=freq, bg="white").pack(anchor="w")

        # ==== Modèle ====
        lf_model = make_section("🧮 Modèle de pricing")
        tk.Radiobutton(lf_model, text="Black-Scholes", variable=self.model_var, value="Black-Scholes", bg="white").pack(anchor="w")
        tk.Radiobutton(lf_model, text="Heston", variable=self.model_var, value="Heston", bg="white").pack(anchor="w")

        # ==== Calibration ====
        lf_calib = make_section(" Calibration Heston")
        tk.Radiobutton(lf_calib, text="Machine Learning", variable=self.calibration_var, value="ML", bg="white").pack(anchor="w")
        tk.Radiobutton(lf_calib, text="Classique", variable=self.calibration_var, value="Classic", bg="white").pack(anchor="w")

        # ==== Bouton ====
        tk.Button(self, text="Lancer le Pricing", bg='green', fg='black', font=('Arial', 12, 'bold'),
                  command=self.run_pricing).pack(pady=20)

    def show_results_table(self, results):
        window = tk.Toplevel(self)
        window.title("📊 Résultats du Pricing")
        window.geometry("1000x400")

        columns = ["Ticker", "Maturité", "Modèle", "Calibration", "Barrière (%)", "Prix PDI (%)", "Coupon (%)", "Delta", "Durée (s)"]
        tree = ttk.Treeview(window, columns=columns, show="headings")

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=110)

        for row in results:
            tree.insert("", tk.END, values=row)

        tree.pack(fill=tk.BOTH, expand=True)

    def run_pricing(self):
        start_time = time.time()
        results = []

        tickers = [t for t, v in self.selected_tickers.items() if v.get()]
        maturities = [m for m, v in self.selected_maturities.items() if v.get()]
        barriers = [int(b.strip('%')) for b, v in self.selected_barriers.items() if v.get()]
        model = self.model_var.get()
        use_ml = self.calibration_var.get()
        freq_str = self.freq_var.get()

        if not tickers or not maturities or not barriers:
            messagebox.showerror("Erreur", "Veuillez sélectionner au moins une action, une maturité et une barrière.")
            return

        maturity_map = {'3 mois': 0.25, '6 mois': 0.5, '1 an': 1.0, '18 mois': 1.5}
        freq_map = {"Trimestrielle": 4, "Semestrielle": 2, "Annuelle": 1}
        freq = freq_map[freq_str]

        if model == "Heston" and use_ml == "ML":
            main_ml = Main()
            ml_model = MLApproach(main_ml.md_Heston)
            ml_model.prepare_dataset(n_samples=5)
            ml_model.build_model()
            ml_model.train(epochs=150)

        for ticker in tickers:
            print(f"🔧 Ticker: {ticker}")
            main = Main(ticker)
            S0 = main.df_market_BS["S0"].iloc[0] if model == "Black-Scholes" else main.df_market_Heston["S0"].iloc[0]

            if model == "Black-Scholes":
                surface = main.calibrate_bs_surface()
                calib_type = "BS"
            elif model == "Heston" and use_ml == "Classic":
                heston_params = main.calibrate_heston([-0.4, 0.3, 0.04, 1.0, 0.04])
                calib_type = "Classique"
            elif model == "Heston" and use_ml == "ML":
                main_ml = Main(ticker)
                heston_params = main.calibrate_heston_ml(ml_model)
                calib_type = "ML"

            for mat in maturities:
                T = maturity_map[mat]
                ZC = 100 * np.exp(-0.0422 * T)

                for barrier in barriers:
                    K = S0
                    H = barrier / 100 * S0

                    if model == "Black-Scholes":
                        def sigma_interp(K_val, T_val):
                            points = surface[["K", "T_years"]].values
                            values = surface["sigma_calib"].values
                            iv = griddata(points, values, (K_val, T_val), method='linear')
                            if np.isnan(iv):
                                iv = griddata(points, values, (K_val, T_val), method='nearest')
                            return iv

                        sigma_fn = sigma_interp(K, T)
                        bs_model = BlackScholesModel(S0=S0, r=0.0422, sigma=sigma_fn)
                        price = (bs_model.price_PDI(K=K, H=H, T=T) / S0**2) * 100**2
                        delta = bs_model.delta_PDI_BS(K=K, H=H, T=T)
                    else:
                        heston_model = HestonModel(
                            S0=S0, r=0.0422,
                            kappa=heston_params["kappa"],
                            theta=heston_params["theta"],
                            sigma=heston_params["sigma"],
                            rho=heston_params["rho"],
                            v0=heston_params["v0"],
                            T=T, payoff="put"
                        )
                        price = (heston_model.price_PDI(K=K, H=H)[0] / S0**2) * 100**2
                        delta = heston_model.estimate_delta_PDI(K=K, H=H)

                    N_coupons = int(T * freq)
                    if N_coupons == 0:
                        coupon = 0.0
                    else:
                        q = np.exp(-0.0422 * T / N_coupons)
                        coupon = (100 - ZC + price) / ((q - q ** (N_coupons + 1)) / (1 - q))

                    elapsed = time.time() - start_time
                    results.append([
                        ticker,
                        f"{T:.2f} an",
                        model,
                        calib_type,
                        f"{barrier}%",
                        f"{price:.4f}",
                        f"{coupon:.4f}",
                        f"{delta:.4f}",
                        f"{elapsed:.2f}"
                    ])
                    print(f"✅ {ticker} | {T:.2f} an | {barrier}% | Prix = {price:.2f}% | Coupon = {coupon:.2f}% | Delta = {delta:.4f}")
                    print(100 - ZC + price)
        self.show_results_table(results)
        total_time = time.time() - start_time
        messagebox.showinfo("Terminé", f"✅ Pricing terminé en {total_time:.2f} secondes.")


# Exemple d’appel
if __name__ == "__main__":
    tickers_dict = {
        "AMZN": "Amazon", "AAPL": "Apple", "MSFT": "Microsoft",
        "TSLA": "Tesla", "GOOG": "Alphabet", "META": "Meta", "NVDA": "NVIDIA",
        "ASML": "ASML NA"
    }
    app = ReverseConvertibleApp(tickers_dict)
    app.mainloop()