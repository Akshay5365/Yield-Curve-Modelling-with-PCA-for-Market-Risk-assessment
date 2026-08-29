# ==============================================================================
# FIXED INCOME RISK & TERM STRUCTURE FACTOR ENGINE
# Direct Ingestion: 'bonds-data.csv', 'USTREASURY.csv', 'svensson-pars.csv'
# ==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, minimize
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple, Optional

# Set global seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ------------------------------------------------------------------------------
# 1. INDIVIDUAL BOND PRICING & ANALYTICAL SENSITIVITY ENGINE
# ------------------------------------------------------------------------------

class CouponBond:
    """
    Fixed-rate coupon bond analytical pricing and risk sensitivity engine.
    Supports semi-annual and annual cash flow structures.
    """
    def __init__(self, name: str, par_value: float, maturity_date: pd.Timestamp, 
                 settle_date: pd.Timestamp, coupon_rate: float, price: float, 
                 freq: int = 2, posted_ytm: Optional[float] = None):
        if par_value <= 0 or price <= 0:
            raise ValueError("Par value and price must be positive.")
        if maturity_date <= settle_date:
            raise ValueError("Maturity date must be after settlement date.")
        
        self.name = name
        self.par_value = par_value
        self.settle_date = settle_date
        self.maturity_date = maturity_date
        self.coupon_rate = coupon_rate    # Decimal (e.g., 0.015 for 1.5%)
        self.price = price
        self.freq = freq                  # 2 = Semi-annual
        self.posted_ytm = posted_ytm      # Stored for delta comparison
        
        # Cash flow dates generation
        self.payment_dates = self._generate_payment_dates()
        self.n_periods = len(self.payment_dates)
        self.period_coupon = (self.par_value * self.coupon_rate) / self.freq
        self.maturity_years = (self.maturity_date - self.settle_date).days / 365.25
        
        # Caches
        self._ytm: Optional[float] = None
        self._duration: Optional[float] = None
        self._modified_duration: Optional[float] = None

    def _generate_payment_dates(self) -> List[pd.Timestamp]:
        """Generates backward payment dates from maturity to settlement."""
        dates = []
        curr = self.maturity_date
        step_months = 12 // self.freq
        while curr > self.settle_date:
            dates.append(curr)
            curr = curr - pd.DateOffset(months=step_months)
        return sorted(dates)

    def present_value_at_rate(self, annual_rate: float) -> float:
        """Computes clean present value given an annualized discount rate."""
        per_rate = annual_rate / self.freq
        t_arr = np.arange(1, self.n_periods + 1)
        pv_coupons = np.sum(self.period_coupon / (1.0 + per_rate) ** t_arr)
        pv_principal = self.par_value / (1.0 + per_rate) ** self.n_periods
        return pv_coupons + pv_principal

    def yield_to_maturity(self, tol: float = 1e-7) -> float:
        """Solves for annualized Yield-to-Maturity (YTM) via Brent's method."""
        if self._ytm is not None:
            return self._ytm
        
        def obj(r):
            return self.present_value_at_rate(r) - self.price

        try:
            sol = root_scalar(obj, bracket=[-0.5, 2.0], method='brentq', xtol=tol)
            self._ytm = sol.root
        except Exception:
            opt = minimize(lambda r: (self.present_value_at_rate(r[0]) - self.price)**2, x0=[0.05])
            self._ytm = float(opt.x[0])
            
        return self._ytm

    def annualized_compounded_ytm(self) -> float:
        """Computes annual effective YTM: ((1 + ytm/2)^2 - 1) * 100."""
        semi_ytm = self.yield_to_maturity()
        return (((1.0 + semi_ytm / self.freq) ** self.freq) - 1.0) * 100.0

    def macaulay_duration(self) -> float:
        """Macaulay duration (in years)."""
        if self._duration is not None:
            return self._duration
        
        ytm = self.yield_to_maturity()
        per_ytm = ytm / self.freq
        t_arr = np.arange(1, self.n_periods + 1)
        
        cf_pv = self.period_coupon / (1.0 + per_ytm) ** t_arr
        principal_pv = self.par_value / (1.0 + per_ytm) ** self.n_periods
        total_pv = np.sum(cf_pv) + principal_pv
        
        weighted_periods = np.sum(t_arr * cf_pv) + (self.n_periods * principal_pv)
        self._duration = (weighted_periods / total_pv) / self.freq
        return self._duration

    def modified_duration(self) -> float:
        """Modified duration (interest rate price sensitivity)."""
        if self._modified_duration is not None:
            return self._modified_duration
        ytm = self.yield_to_maturity()
        self._modified_duration = self.macaulay_duration() / (1.0 + ytm / self.freq)
        return self._modified_duration

    def get_cash_flow_schedule(self) -> pd.DataFrame:
        """Returns cash flow schedule dataframe."""
        amounts = [self.period_coupon] * self.n_periods
        amounts[-1] += self.par_value
        return pd.DataFrame({
            'NAME': self.name,
            'PAYMENT_DATE': self.payment_dates,
            'CF': amounts
        })


# ------------------------------------------------------------------------------
# 2. SVENSSON ZERO-COUPON TERM STRUCTURE MODEL
# ------------------------------------------------------------------------------

class SvenssonYieldCurve:
    """
    Nelson-Siegel-Svensson (1994) 6-parameter term structure model:
    y(t) = beta0 + beta1*((1-exp(-t/tau1))/(t/tau1)) 
                 + beta2*(((1-exp(-t/tau1))/(t/tau1)) - exp(-t/tau1))
                 + beta3*(((1-exp(-t/tau2))/(t/tau2)) - exp(-t/tau2))
    """
    @staticmethod
    def spot_rate(t: np.ndarray, beta0: float, beta1: float, beta2: float, 
                  beta3: float, tau1: float, tau2: float) -> np.ndarray:
        t = np.maximum(np.asarray(t, dtype=float), 1e-6)
        term1 = (1.0 - np.exp(-t / tau1)) / (t / tau1)
        term2 = term1 - np.exp(-t / tau1)
        term3 = (1.0 - np.exp(-t / tau2)) / (t / tau2) - np.exp(-t / tau2)
        return beta0 + beta1 * term1 + beta2 * term2 + beta3 * term3


# ------------------------------------------------------------------------------
# 3. YIELD CURVE PCA & STATISTICAL BOOTSTRAP ENGINE
# ------------------------------------------------------------------------------

class YieldCurvePCA:
    """
    Principal Component Analysis on Term Structure daily changes (in basis points).
    Performs empirical SVD factor decomposition, window stability, and bootstrap inference.
    """
    def __init__(self, rates_df: pd.DataFrame):
        self.dates = pd.to_datetime(rates_df['DATE'].values)
        self.tenor_cols = [c for c in rates_df.columns if c != 'DATE']
        self.raw_rates = rates_df[self.tenor_cols].values
        
        # Daily changes in basis points: Delta y_t = (y_t - y_{t-1}) * 100
        self.delta_bps = np.diff(self.raw_rates, axis=0) * 100.0
        self.valid_dates = self.dates[1:]
        
        # Center and scale (Z-score standardization)
        self.mean_bps = np.mean(self.delta_bps, axis=0)
        self.std_bps = np.std(self.delta_bps, axis=0, ddof=1)
        self.scaled_bps = (self.delta_bps - self.mean_bps) / self.std_bps
        
        self.pca = PCA(n_components=min(11, self.delta_bps.shape[1]))
        self.scores = self.pca.fit_transform(self.scaled_bps)
        self.loadings = self.pca.components_
        
        # Align sign convention (PC1: Level > 0, PC2: Slope long > short)
        if np.mean(self.loadings[0]) < 0:
            self.loadings[0] = -self.loadings[0]
            self.scores[:, 0] = -self.scores[:, 0]
        if self.loadings[1, -1] < self.loadings[1, 0]:
            self.loadings[1] = -self.loadings[1]
            self.scores[:, 1] = -self.scores[:, 1]

    def variance_explained(self) -> pd.DataFrame:
        var_exp = self.pca.explained_variance_ratio_ * 100.0
        return pd.DataFrame({
            'PC': [f"PC{i+1}" for i in range(len(var_exp))],
            'Variance_Explained_Pct': var_exp,
            'Cumulative_Variance_Pct': np.cumsum(var_exp)
        })

    def run_bootstrap_validation(self, n_reps: int = 2000, sample_prop: float = 1/6) -> Dict[str, np.ndarray]:
        """Runs non-parametric bootstrap resampling to evaluate eigenvalue/loading stability."""
        n_obs = len(self.scaled_bps)
        sub_n = int(n_obs * sample_prop)
        
        boot_eigenvals = []
        boot_loadings = {f"w{k+1}": [] for k in range(3)}
        boot_cum_var3 = []
        
        for _ in range(n_reps):
            sample_idx = np.random.choice(n_obs, size=sub_n, replace=True)
            sample_data = self.scaled_bps[sample_idx, :]
            
            boot_pca = PCA(n_components=3)
            boot_pca.fit(sample_data)
            
            boot_eigenvals.append(boot_pca.explained_variance_)
            boot_cum_var3.append(np.sum(boot_pca.explained_variance_ratio_[:3]))
            
            for k in range(3):
                loading = boot_pca.components_[k]
                if np.dot(loading, self.loadings[k]) < 0:
                    loading = -loading
                boot_loadings[f"w{k+1}"].append(loading)
                
        return {
            'eigenvalues': np.array(boot_eigenvals),
            'loadings': {k: np.array(v) for k, v in boot_loadings.items()},
            'cum_var_pc3': np.array(boot_cum_var3)
        }


# ------------------------------------------------------------------------------
# 4. PORTFOLIO KEY-RATE MAPPING & RISK ATTRIBUTION ENGINE
# ------------------------------------------------------------------------------

class FixedIncomePortfolio:
    """
    Portfolio Risk Engine: Aggregates cash flows across payment dates,
    evaluates Svensson zero rates, calculates analytical PV01 vectors,
    and projects risk factor sensitivities (RF1, RF2, RF3).
    """
    def __init__(self, bonds: List[CouponBond], settle_date: pd.Timestamp):
        self.bonds = bonds
        self.settle_date = settle_date
        self._aggregate_cash_flows()

    def _aggregate_cash_flows(self):
        cf_list = [b.get_cash_flow_schedule() for b in self.bonds]
        all_cfs = pd.concat(cf_list, ignore_index=True)
        
        # Aggregate multiple cash flows sharing the exact same settlement date
        self.portfolio_cf = all_cfs.groupby('PAYMENT_DATE')['CF'].sum().reset_index()
        self.portfolio_cf['matSven'] = (self.portfolio_cf['PAYMENT_DATE'] - self.settle_date).dt.days / 360.0
        self.portfolio_cf = self.portfolio_cf.sort_values('matSven').reset_index(drop=True)

    def calculate_risk_sensitivities(self, latest_sven_params: np.ndarray, 
                                     historical_sven_df: pd.DataFrame,
                                     dates_series: pd.Series) -> pd.DataFrame:
        maturities = self.portfolio_cf['matSven'].values
        cfs = self.portfolio_cf['CF'].values
        
        # Evaluate current spot rate curve on cash flow dates
        spot_rates_pct = SvenssonYieldCurve.spot_rate(maturities, *latest_sven_params)
        spot_rates_dec = spot_rates_pct / 100.0
        
        # Exact analytical discounting: PV = CF / (1 + r)^t
        pval = cfs / ((1.0 + spot_rates_dec) ** maturities)
        # Shift down by 1 basis point (0.01% = 0.0001)
        pval001 = cfs / ((1.0 + spot_rates_dec - 0.0001) ** maturities)
        pv01_vec = pval001 - pval
        
        self.portfolio_cf['Spot_Rate_Pct'] = spot_rates_pct
        self.portfolio_cf['Present_Value'] = pval
        self.portfolio_cf['PV01'] = pv01_vec
        self.total_pv01 = np.sum(pv01_vec)
        
        # ----------------------------------------------------------------------
        # Historical term structure PCA on all cash flow maturities
        # ----------------------------------------------------------------------
        sven_history = []
        for i in range(len(historical_sven_df)):
            p = historical_sven_df.iloc[i].values
            sven_history.append(SvenssonYieldCurve.spot_rate(maturities, *p))
            
        sven_history_mat = np.array(sven_history)
        sven_diff_bps = np.diff(sven_history_mat, axis=0) * 100.0
        sven_diff_scaled = (sven_diff_bps - np.mean(sven_diff_bps, axis=0)) / np.std(sven_diff_bps, axis=0, ddof=1)
        
        pca_sven = PCA(n_components=3)
        self.sven_scores = pca_sven.fit_transform(sven_diff_scaled)
        self.sven_loadings = pca_sven.components_
        
        # Standardize signs
        if np.mean(self.sven_loadings[0]) < 0:
            self.sven_loadings[0] = -self.sven_loadings[0]
            self.sven_scores[:, 0] = -self.sven_scores[:, 0]
        if self.sven_loadings[1, -1] < self.sven_loadings[1, 0]:
            self.sven_loadings[1] = -self.sven_loadings[1]
            self.sven_scores[:, 1] = -self.sven_scores[:, 1]
            
        self.portfolio_cf['w1_Level'] = self.sven_loadings[0]
        self.portfolio_cf['w2_Slope'] = self.sven_loadings[1]
        self.portfolio_cf['w3_Curvature'] = self.sven_loadings[2]
        
        # Risk factor sensitivities: RF_k = sum_i (PV01_i * w_{k,i})
        self.RF1 = np.sum(pv01_vec * self.sven_loadings[0])
        self.RF2 = np.sum(pv01_vec * self.sven_loadings[1])
        self.RF3 = np.sum(pv01_vec * self.sven_loadings[2])
        self.risk_factors = np.array([self.RF1, self.RF2, self.RF3])
        
        # Daily Historical PnL Simulation: PnL_t = - (PC_t @ RF)
        self.pnl_series = pd.Series(
            - np.dot(self.sven_scores, self.risk_factors),
            index=dates_series.iloc[1:].values,
            name="Portfolio_PnL"
        )
        return self.portfolio_cf


# ------------------------------------------------------------------------------
# 5. MASTER EXECUTION PIPELINE (DIRECT CSV INGESTION)
# ------------------------------------------------------------------------------

def main():
    print("================================================================================")
    print("      FIXED INCOME QUANTITATIVE RISK & TERM STRUCTURE FACTOR ENGINE             ")
    print("================================================================================")

    # 1. Ingest CSV Datasets
    bonds_path = "bonds-data.csv"
    rates_path = "USTREASURY.csv"
    sven_path = "svensson-pars.csv"

    for path in [bonds_path, rates_path, sven_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required dataset: '{path}' in working directory.")

    print("\n[1] Loading CSV datasets...")
    bonds_raw = pd.read_csv(bonds_path)
    rates_raw = pd.read_csv(rates_path)
    sven_raw = pd.read_csv(sven_path)

    settle_date = pd.to_datetime("2020-03-10")
    rates_raw['DATE'] = pd.to_datetime(rates_raw['DATE'])
    rates_df = rates_raw[rates_raw['DATE'] <= settle_date].sort_values('DATE').reset_index(drop=True)
    sven_df = sven_raw.iloc[:len(rates_df)].reset_index(drop=True)

    print(f"    - Bonds Dataset:       {len(bonds_raw)} US Treasury Bonds loaded.")
    print(f"    - Spot Curves (Rates): {rates_df.shape[0]} daily dates ({rates_df['DATE'].iloc[0].strftime('%Y-%m-%d')} to {rates_df['DATE'].iloc[-1].strftime('%Y-%m-%d')}).")
    print(f"    - Svensson Parameters: {sven_df.shape[0]} calibrated parameter vectors.")

    # 2. Parse Bond Portfolio & Compare YTMs
    print("\n[2] Initializing Bond Portfolio & Verifying Posted vs Calculated YTM...")
    bonds_list = []
    ytm_comparisons = []

    for _, row in bonds_raw.iterrows():
        b = CouponBond(
            name=row['NAME'],
            par_value=float(row['FACE_VAL']),
            maturity_date=pd.to_datetime(row['MATURITY']),
            settle_date=settle_date,
            coupon_rate=float(row['CP_RATE']) / 100.0,
            price=float(row['MRKT_PRICE']),
            freq=2,
            posted_ytm=float(row['YTM'])
        )
        bonds_list.append(b)
        calc_ann_ytm = b.annualized_compounded_ytm()
        ytm_comparisons.append({
            'NAME': b.name,
            'POSTED_YTM': b.posted_ytm,
            'CALC_YTM_ANN': calc_ann_ytm,
            'DELTA_YTM': calc_ann_ytm - b.posted_ytm,
            'DURATION': b.macaulay_duration(),
            'MOD_DURATION': b.modified_duration()
        })

    df_ytm_comp = pd.DataFrame(ytm_comparisons)
    print(df_ytm_comp[['NAME', 'POSTED_YTM', 'CALC_YTM_ANN', 'DELTA_YTM', 'DURATION', 'MOD_DURATION']].head(5).to_string(index=False))

    # 3. PCA on Historical Benchmark Spot Curves (USTREASURY.csv)
    print("\n[3] Computing Term Structure PCA on US Treasury Yield Changes...")
    pca_ust = YieldCurvePCA(rates_df)
    var_df = pca_ust.variance_explained()
    print(var_df.head(4).to_string(index=False))

    # 4. Bootstrap Resampling Validation (1,000 Iterations)
    print("\n[4] Executing 1,000-Iteration Bootstrap Validation for Eigenvalue Stability...")
    boot_res = pca_ust.run_bootstrap_validation(n_reps=1000, sample_prop=1/6)
    mean_boot_var = np.mean(boot_res['cum_var_pc3']) * 100.0
    print(f"    - Bootstrap Mean Cumulative Variance (3 PCs): {mean_boot_var:.2f}% (Std: {np.std(boot_res['cum_var_pc3'])*100:.2f}%)")

    # 5. Portfolio Cash Flow Mapping & Factor Sensitivities
    print("\n[5] Mapping Cash Flows to Svensson Spot Rates & Calculating Factor Sensitivities...")
    portfolio = FixedIncomePortfolio(bonds_list, settle_date)
    latest_params = sven_df.iloc[-1].values
    risk_cf_df = portfolio.calculate_risk_sensitivities(latest_params, sven_df, rates_df['DATE'])

    print(f"    - Unique Portfolio Cash Flow Vertices: {len(risk_cf_df)}")
    print(f"    - Portfolio Total Present Value:       ${risk_cf_df['Present_Value'].sum():,.2f}")
    print(f"    - Total Portfolio PV01:                 ${portfolio.total_pv01:.6f} per 1 bp parallel shift")
    print(f"    - Factor Sensitivity RF1 (Level):      ${portfolio.RF1:8.4f}")
    print(f"    - Factor Sensitivity RF2 (Slope):      ${portfolio.RF2:8.4f}")
    print(f"    - Factor Sensitivity RF3 (Curvature):  ${portfolio.RF3:8.4f}")

    pnl_99_var = np.percentile(portfolio.pnl_series, 1.0)
    print(f"    - Reconstructed 1-Day 99% Value at Risk (VaR): ${abs(pnl_99_var):.2f}")

    # --------------------------------------------------------------------------
    # 6. PUBLICATION DIAGNOSTIC VISUALIZATIONS
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Panel 1: Differences between Calculated and Posted YTM
    axes[0, 0].bar(df_ytm_comp['NAME'], df_ytm_comp['DELTA_YTM'], color='lightblue', edgecolor='grey')
    axes[0, 0].axhline(0, color='red', linestyle='--', lw=1)
    axes[0, 0].set_title("Differences: Calculated vs. Posted YTM (March 10, 2020)", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Obtained YTM - Posted YTM (%)")
    axes[0, 0].tick_params(axis='x', rotation=90, labelsize=7)
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: Portfolio Aggregated Cash Flow Profile & Key Tenors
    axes[0, 1].bar(risk_cf_df['matSven'], risk_cf_df['CF'], width=0.3, color='steelblue', edgecolor='navy', alpha=0.8)
    axes[0, 1].set_title(f"Aggregated Portfolio Cash Flows (Total Vertices: {len(risk_cf_df)})", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Maturity from Settlement (Years)")
    axes[0, 1].set_ylabel("Cash Flow Amount ($)")
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: Empirical PCA Loadings (Level, Slope, Curvature)
    mat_labels = pca_ust.tenor_cols
    mat_x = np.arange(len(mat_labels))
    axes[1, 0].plot(mat_x, pca_ust.loadings[0], 'b-o', lw=1.8, label=f"PC1 Level ({var_df.iloc[0, 1]:.1f}%)")
    axes[1, 0].plot(mat_x, pca_ust.loadings[1], 'g-s', lw=1.8, label=f"PC2 Slope ({var_df.iloc[1, 1]:.1f}%)")
    axes[1, 0].plot(mat_x, pca_ust.loadings[2], 'r-^', lw=1.8, label=f"PC3 Curvature ({var_df.iloc[2, 1]:.1f}%)")
    axes[1, 0].set_xticks(mat_x)
    axes[1, 0].set_xticklabels(mat_labels, rotation=45)
    axes[1, 0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].set_title("US Treasury Term Structure Eigenvector Loadings", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Loadings ($w_k$)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 4: PCA Factor Model Historical PnL Distribution
    axes[1, 1].hist(portfolio.pnl_series, bins=100, color='lightblue', edgecolor='grey', alpha=0.8)
    axes[1, 1].axvline(pnl_99_var, color='crimson', linestyle='--', lw=2, label=f"99% 1-Day VaR (${abs(pnl_99_var):.2f})")
    axes[1, 1].set_title("3-Factor PCA Historical Daily P&L Distribution", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Portfolio Daily Profit & Loss ($)")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()