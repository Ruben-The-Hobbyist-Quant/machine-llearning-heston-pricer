import yfinance as yf
import pandas as pd
import numpy as np
from HestonModel import HestonModel
from BSModel import BlackScholesModel
from datetime import datetime 
from multiprocessing import Pool,cpu_count
from functools import partial
import time







    # ---- PARTIE SIMULATION ----

def _gen_surface(idx, grid_flat, param_ranges, r, option_type, seed_offset):
    rng = np.random.default_rng(seed_offset + idx)
    
    
    # 1) tirage aléatoire des 5 paramètres Heston
    params = {k: rng.uniform(*v) for k, v in param_ranges.items()}
    y_i = np.fromiter(params.values(), dtype=np.float32)   # label

    hm = HestonModel(
        S0=1.0, r=r,
        kappa=params["kappa"], theta=params["theta"],
        sigma=params["sigma"], rho=params["rho"], v0=params["v0"],
        T=1.0                           # sera écrasé dans la boucle T
    )

    vols = []
    bs = BlackScholesModel(S0=hm.S0, r=r)

    # 2) boucle vectorisée (150 points) pour cette surface
    for KS, T in grid_flat:
        K = KS * hm.S0
        hm.T = float(T)

        price = hm.FFT([K])[0]          # prix Heston

        
        iv    = bs.implied_vol(price, K, hm.T, option_type)
        vols.append(iv)

    return np.asarray(vols, dtype=np.float32), y_i

class MarketData:
    def __init__(
        self,
        ticker="^SPX",
        r=0.05,
        option_type="call",
        max_expiries=None,
        min_bid=0.01
    ):
        self.ticker = ticker.upper()
        self.option_type = option_type.lower()
        self.r = r
        self.max_expiries = max_expiries
        self.min_bid = min_bid
        self.df = None

    def load(self):
        stock = yf.Ticker(self.ticker)
        spot = stock.history(period="1d")["Close"].iloc[-1]

        expiries = stock.options
        if self.max_expiries:
            expiries = expiries[: self.max_expiries]

        today = datetime.utcnow().date()
        frames = []

        for exp in expiries:
            expiry_date = datetime.strptime(exp, "%Y-%m-%d").date()
            days_to_expiry = (expiry_date - today).days

            # Filtrage : entre 1 et 365 jours jusqu'à maturité
            if not (100 <= days_to_expiry <= 300):
                continue

            chain = stock.option_chain(exp)
            df_opt = chain.calls if self.option_type == "call" else chain.puts
            df_opt = df_opt[df_opt["bid"] >= self.min_bid]
            df_opt = df_opt[   0.1 < (df_opt["impliedVolatility"] < 1.5)
]


            if df_opt.empty:
                continue

            # Création du DataFrame avec la vol implicite de Yahoo incluse
            df_tmp = pd.DataFrame({
                "expiry": exp,
                "days_to_expiry": days_to_expiry,
                "S0": spot,
                "K": df_opt["strike"],
                "r": self.r,
                "price": (df_opt["bid"] + df_opt["ask"]) / 2,
                "spread" : df_opt["ask"] - df_opt["bid"],
                "type": self.option_type,
                "implied_vol_yahoo": df_opt["impliedVolatility"]
            })
            df_tmp = df_tmp.sort_values(by="K", ascending=True).reset_index(drop=True)
            frames.append(df_tmp)
            df_tmp = df_tmp[df_tmp["spread"] >= 0.01]
            df_tmp = df_tmp[ 0.6 <(df_tmp["K"]/df_tmp["S0"] < 1.4)]

        self.df = pd.concat(frames, ignore_index=True)
        return self.df

    
    # ──────────────────────────────────────────────────────────────
    #  APPEL “HIGH-LEVEL” : génère N surfaces en parallèle
    # ──────────────────────────────────────────────────────────────
    def simulate_heston_surfaces(
        self,
        n_samples,
        moneyness_grid=np.linspace(0.7, 1.3, 15),
        maturities=np.linspace(0.1, 1.0, 10),
        param_ranges=None,
        seed=42,
        n_processes=4
    ):
        """
        Retourne :
            X : (n_samples, n_grid)  Vols implicites flatten
            y : (n_samples, 5)       [v0, theta, kappa, sigma, rho]
        """
    
        if param_ranges is None:
            param_ranges = {
                "v0":    (0.02, 0.12),     # plutôt autour de 0.06–0.1
                "theta": (0.05, 0.2),
                "kappa": (1, 11.0),
                "sigma": (0.15, 1.0),
                "rho":   (-0.9, -0.1)
            }
    
        # grille (moneyness, maturité)
        grid_KS, grid_T = np.meshgrid(moneyness_grid, maturities, indexing="ij")
        grid_flat = np.column_stack([grid_KS.ravel(), grid_T.ravel()])  # (150, 2)
           
        worker = partial(
            _gen_surface,
            grid_flat=grid_flat,
            param_ranges=param_ranges,
            r=self.r,
            option_type=self.option_type,
            seed_offset=seed
        )


        total_start = time.time()
        with Pool(processes=n_processes or cpu_count()) as pool:
            results = pool.map(worker, range(n_samples))
        total_end = time.time()
        print(f"\n⏱️ Temps total pour {n_samples} surfaces : {total_end - total_start:.2f} secondes")

        
        
    
        X, y = zip(*results)
        return np.stack(X), np.stack(y)
