import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, minimize
from sklearn.decomposition import PCA
from typing import List, Dict, Optional

np.random.seed(42)

DAYS_IN_YEAR = 365.25
CONTINUOUS_ZERO = True  # Svensson fit is assumed to give continuously compounded zero rates


def discount_factor(rate, t):
    if CONTINUOUS_ZERO:
        return np.exp(-rate * t)
    return (1.0 + rate) ** (-t)


def align_signs(loadings, scores):
    """PC1 positive on average (level), PC2 rising from short to long end (slope)."""
    if np.mean(loadings[0]) < 0:
        loadings[0] = -loadings[0]
        scores[:, 0] = -scores[:, 0]
    if loadings[1, -1] < loadings[1, 0]:
        loadings[1] = -loadings[1]
        scores[:, 1] = -scores[:, 1]


class CouponBond:
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
        self.coupon_rate = coupon_rate
        self.price = price
        self.freq = freq
        self.posted_ytm = posted_ytm

        self.payment_dates = self._generate_payment_dates()
        self.n_periods = len(self.payment_dates)
        self.period_coupon = (self.par_value * self.coupon_rate) / self.freq
        self.maturity_years = (self.maturity_date - self.settle_date).days / DAYS_IN_YEAR

        # the quoted price is clean, so add the interest accrued since the last coupon
        prev_coupon = self.payment_dates[0] - pd.DateOffset(months=12 // self.freq)
        period_days = (self.payment_dates[0] - prev_coupon).days
        self.accrued_frac = (self.settle_date - prev_coupon).days / period_days
        self.dirty_price = self.price + self.period_coupon * self.accrued_frac

        # time to each cash flow in coupon periods, the first one is a stub
        self.periods = np.arange(self.n_periods) + (1.0 - self.accrued_frac)

        self._ytm: Optional[float] = None
        self._duration: Optional[float] = None
        self._modified_duration: Optional[float] = None

    def _generate_payment_dates(self) -> List[pd.Timestamp]:
        dates = []
        curr = self.maturity_date
        step_months = 12 // self.freq
        while curr > self.settle_date:
            dates.append(curr)
            curr = curr - pd.DateOffset(months=step_months)
        return sorted(dates)

    def present_value_at_rate(self, annual_rate: float) -> float:
        """Full (dirty) present value at a nominal annual rate."""
        per_rate = annual_rate / self.freq
        pv_coupons = np.sum(self.period_coupon / (1.0 + per_rate) ** self.periods)
        pv_principal = self.par_value / (1.0 + per_rate) ** self.periods[-1]
        return pv_coupons + pv_principal

    def yield_to_maturity(self, tol: float = 1e-7) -> float:
        if self._ytm is not None:
            return self._ytm

        def obj(r):
            return self.present_value_at_rate(r) - self.dirty_price

        try:
            sol = root_scalar(obj, bracket=[-0.5, 2.0], method='brentq', xtol=tol)
            self._ytm = sol.root
        except Exception:
            opt = minimize(lambda r: obj(r[0]) ** 2, x0=[0.05])
            self._ytm = float(opt.x[0])
        return self._ytm

    def annualized_compounded_ytm(self) -> float:
        semi_ytm = self.yield_to_maturity()
        return (((1.0 + semi_ytm / self.freq) ** self.freq) - 1.0) * 100.0

    def macaulay_duration(self) -> float:
        if self._duration is not None:
            return self._duration

        per_ytm = self.yield_to_maturity() / self.freq
        cf_pv = self.period_coupon / (1.0 + per_ytm) ** self.periods
        principal_pv = self.par_value / (1.0 + per_ytm) ** self.periods[-1]
        total_pv = np.sum(cf_pv) + principal_pv

        weighted = np.sum(self.periods * cf_pv) + self.periods[-1] * principal_pv
        self._duration = (weighted / total_pv) / self.freq
        return self._duration

    def modified_duration(self) -> float:
        if self._modified_duration is not None:
            return self._modified_duration
        ytm = self.yield_to_maturity()
        self._modified_duration = self.macaulay_duration() / (1.0 + ytm / self.freq)
        return self._modified_duration

    def get_cash_flow_schedule(self) -> pd.DataFrame:
        amounts = [self.period_coupon] * self.n_periods
        amounts[-1] += self.par_value
        return pd.DataFrame({
            'NAME': self.name,
            'PAYMENT_DATE': self.payment_dates,
            'CF': amounts
        })


class SvenssonYieldCurve:
    """
    y(t) = b0 + b1*L1 + b2*(L1 - exp(-t/tau1)) + b3*(L2 - exp(-t/tau2)),
    where Li = (1 - exp(-t/taui)) / (t/taui)
    """
    @staticmethod
    def spot_rate(t, beta0, beta1, beta2, beta3, tau1, tau2):
        t = np.maximum(np.asarray(t, dtype=float), 1e-6)
        term1 = (1.0 - np.exp(-t / tau1)) / (t / tau1)
        term2 = term1 - np.exp(-t / tau1)
        term3 = (1.0 - np.exp(-t / tau2)) / (t / tau2) - np.exp(-t / tau2)
        return beta0 + beta1 * term1 + beta2 * term2 + beta3 * term3


class YieldCurvePCA:
    """PCA on standardized daily yield changes (bps)."""
    def __init__(self, rates_df: pd.DataFrame):
        self.dates = pd.to_datetime(rates_df['DATE'].values)
        self.tenor_cols = [c for c in rates_df.columns if c != 'DATE']
        self.raw_rates = rates_df[self.tenor_cols].values

        self.delta_bps = np.diff(self.raw_rates, axis=0) * 100.0
        self.valid_dates = self.dates[1:]

        self.mean_bps = np.mean(self.delta_bps, axis=0)
        self.std_bps = np.std(self.delta_bps, axis=0, ddof=1)
        self.scaled_bps = (self.delta_bps - self.mean_bps) / self.std_bps

        self.pca = PCA(n_components=min(11, self.delta_bps.shape[1]))
        self.scores = self.pca.fit_transform(self.scaled_bps)
        self.loadings = self.pca.components_
        align_signs(self.loadings, self.scores)

    def variance_explained(self) -> pd.DataFrame:
        var_exp = self.pca.explained_variance_ratio_ * 100.0
        return pd.DataFrame({
            'PC': [f"PC{i+1}" for i in range(len(var_exp))],
            'Variance_Explained_Pct': var_exp,
            'Cumulative_Variance_Pct': np.cumsum(var_exp)
        })

    def run_bootstrap_validation(self, n_reps: int = 1000, sample_prop: float = 1/6,
                                 block_len: int = 20) -> Dict[str, np.ndarray]:
        """Moving-block bootstrap, so the resamples keep some of the serial dependence in the changes."""
        n_obs = len(self.scaled_bps)
        sub_n = int(n_obs * sample_prop)
        n_blocks = int(np.ceil(sub_n / block_len))

        boot_eigenvals = []
        boot_loadings = {f"w{k+1}": [] for k in range(3)}
        boot_cum_var3 = []

        for _ in range(n_reps):
            starts = np.random.randint(0, n_obs - block_len + 1, size=n_blocks)
            idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:sub_n]

            boot_pca = PCA(n_components=3)
            boot_pca.fit(self.scaled_bps[idx, :])

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


class FixedIncomePortfolio:
    def __init__(self, bonds: List[CouponBond], settle_date: pd.Timestamp):
        self.bonds = bonds
        self.settle_date = settle_date
        self._aggregate_cash_flows()

    def _aggregate_cash_flows(self):
        all_cfs = pd.concat([b.get_cash_flow_schedule() for b in self.bonds], ignore_index=True)
        self.portfolio_cf = all_cfs.groupby('PAYMENT_DATE')['CF'].sum().reset_index()
        self.portfolio_cf['matSven'] = (self.portfolio_cf['PAYMENT_DATE'] - self.settle_date).dt.days / DAYS_IN_YEAR
        self.portfolio_cf = self.portfolio_cf.sort_values('matSven').reset_index(drop=True)

    def calculate_risk_sensitivities(self, latest_sven_params: np.ndarray,
                                     historical_sven_df: pd.DataFrame,
                                     dates_series: pd.Series) -> pd.DataFrame:
        maturities = self.portfolio_cf['matSven'].values
        cfs = self.portfolio_cf['CF'].values

        spot_rates_pct = SvenssonYieldCurve.spot_rate(maturities, *latest_sven_params)
        spot_rates_dec = spot_rates_pct / 100.0

        pval = cfs * discount_factor(spot_rates_dec, maturities)
        pval001 = cfs * discount_factor(spot_rates_dec - 0.0001, maturities)
        pv01_vec = pval001 - pval

        self.portfolio_cf['Spot_Rate_Pct'] = spot_rates_pct
        self.portfolio_cf['Present_Value'] = pval
        self.portfolio_cf['PV01'] = pv01_vec
        self.total_pv01 = np.sum(pv01_vec)

        sven_history = np.array([
            SvenssonYieldCurve.spot_rate(maturities, *historical_sven_df.iloc[i].values)
            for i in range(len(historical_sven_df))
        ])
        sven_diff_bps = np.diff(sven_history, axis=0) * 100.0
        sven_mean = np.mean(sven_diff_bps, axis=0)
        sven_std = np.std(sven_diff_bps, axis=0, ddof=1)
        sven_diff_scaled = (sven_diff_bps - sven_mean) / sven_std

        pca_sven = PCA(n_components=3)
        self.sven_scores = pca_sven.fit_transform(sven_diff_scaled)
        self.sven_loadings = pca_sven.components_
        align_signs(self.sven_loadings, self.sven_scores)

        self.portfolio_cf['w1_Level'] = self.sven_loadings[0]
        self.portfolio_cf['w2_Slope'] = self.sven_loadings[1]
        self.portfolio_cf['w3_Curvature'] = self.sven_loadings[2]

        # the scores are in standardized units, so the vertex volatility has to go back into the exposure
        self.risk_factors = self.sven_loadings @ (pv01_vec * sven_std)
        self.RF1, self.RF2, self.RF3 = self.risk_factors

        pnl_index = dates_series.iloc[1:].values
        drift = -np.dot(sven_mean, pv01_vec)
        self.pnl_series = pd.Series(-np.dot(self.sven_scores, self.risk_factors) + drift,
                                    index=pnl_index, name="Portfolio_PnL")
        self.full_pnl_series = pd.Series(-np.dot(sven_diff_bps, pv01_vec),
                                         index=pnl_index, name="Portfolio_PnL_full")
        return self.portfolio_cf


def load_svensson(path, rates_df):
    sven_raw = pd.read_csv(path)
    date_col = next((c for c in sven_raw.columns if c.lower() == 'date'), None)

    if date_col is None:
        if len(sven_raw) < len(rates_df):
            raise ValueError("svensson-pars.csv has fewer rows than the Treasury curve file.")
        return rates_df, sven_raw.iloc[:len(rates_df)].reset_index(drop=True)

    sven_raw = sven_raw.rename(columns={date_col: 'DATE'})
    sven_raw['DATE'] = pd.to_datetime(sven_raw['DATE'])
    merged = rates_df[['DATE']].merge(sven_raw, on='DATE', how='inner')
    rates_df = rates_df[rates_df['DATE'].isin(merged['DATE'])].reset_index(drop=True)
    return rates_df, merged.drop(columns='DATE').reset_index(drop=True)


def main():
    bonds_path = "bonds-data.csv"
    rates_path = "USTREASURY.csv"
    sven_path = "svensson-pars.csv"

    for path in [bonds_path, rates_path, sven_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required dataset: '{path}' in working directory.")

    bonds_raw = pd.read_csv(bonds_path)
    rates_raw = pd.read_csv(rates_path)

    settle_date = pd.to_datetime("2020-03-10")
    rates_raw['DATE'] = pd.to_datetime(rates_raw['DATE'])
    rates_df = rates_raw[rates_raw['DATE'] <= settle_date].sort_values('DATE').reset_index(drop=True)
    rates_df, sven_df = load_svensson(sven_path, rates_df)

    print(f"Bonds loaded:        {len(bonds_raw)}")
    print(f"Treasury curve days: {rates_df.shape[0]} ({rates_df['DATE'].iloc[0]:%Y-%m-%d} to {rates_df['DATE'].iloc[-1]:%Y-%m-%d})")
    print(f"Svensson parameters: {sven_df.shape[0]} rows")

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
            'CALC_YTM_NOM': b.yield_to_maturity() * 100.0,
            'CALC_YTM_ANN': calc_ann_ytm,
            'DELTA_YTM': calc_ann_ytm - b.posted_ytm,
            'DURATION': b.macaulay_duration(),
            'MOD_DURATION': b.modified_duration()
        })

    df_ytm_comp = pd.DataFrame(ytm_comparisons)
    print("\nPosted vs calculated YTM")
    print(df_ytm_comp.head(5).to_string(index=False))

    pca_ust = YieldCurvePCA(rates_df)
    var_df = pca_ust.variance_explained()
    print("\nVariance explained")
    print(var_df.head(4).to_string(index=False))

    boot_res = pca_ust.run_bootstrap_validation(n_reps=1000, sample_prop=1/6)
    mean_boot_var = np.mean(boot_res['cum_var_pc3']) * 100.0
    print(f"\nBootstrap cumulative variance, 3 PCs: {mean_boot_var:.2f}% (std {np.std(boot_res['cum_var_pc3'])*100:.2f}%)")

    portfolio = FixedIncomePortfolio(bonds_list, settle_date)
    latest_params = sven_df.iloc[-1].values
    risk_cf_df = portfolio.calculate_risk_sensitivities(latest_params, sven_df, rates_df['DATE'])

    pnl_99_var = np.percentile(portfolio.pnl_series, 1.0)
    full_99_var = np.percentile(portfolio.full_pnl_series, 1.0)

    print(f"\nCash flow vertices:    {len(risk_cf_df)}")
    print(f"Portfolio PV:          ${risk_cf_df['Present_Value'].sum():,.2f}")
    print(f"Portfolio PV01:        ${portfolio.total_pv01:.6f} per bp")
    print(f"RF1 (level):           ${portfolio.RF1:8.4f}")
    print(f"RF2 (slope):           ${portfolio.RF2:8.4f}")
    print(f"RF3 (curvature):       ${portfolio.RF3:8.4f}")
    print(f"1-day 99% VaR, 3 factors:      ${abs(pnl_99_var):.2f}")
    print(f"1-day 99% VaR, all vertices:   ${abs(full_99_var):.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    axes[0, 0].bar(df_ytm_comp['NAME'], df_ytm_comp['DELTA_YTM'], color='lightblue', edgecolor='grey')
    axes[0, 0].axhline(0, color='red', linestyle='--', lw=1)
    axes[0, 0].set_title("Calculated vs posted YTM (March 10, 2020)", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Calculated YTM - Posted YTM (%)")
    axes[0, 0].tick_params(axis='x', rotation=90, labelsize=7)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].bar(risk_cf_df['matSven'], risk_cf_df['CF'], width=0.3, color='steelblue', edgecolor='navy', alpha=0.8)
    axes[0, 1].set_title(f"Aggregated portfolio cash flows ({len(risk_cf_df)} vertices)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Years from settlement")
    axes[0, 1].set_ylabel("Cash flow ($)")
    axes[0, 1].grid(True, alpha=0.3)

    mat_x = np.arange(len(pca_ust.tenor_cols))
    axes[1, 0].plot(mat_x, pca_ust.loadings[0], 'b-o', lw=1.8, label=f"PC1 Level ({var_df.iloc[0, 1]:.1f}%)")
    axes[1, 0].plot(mat_x, pca_ust.loadings[1], 'g-s', lw=1.8, label=f"PC2 Slope ({var_df.iloc[1, 1]:.1f}%)")
    axes[1, 0].plot(mat_x, pca_ust.loadings[2], 'r-^', lw=1.8, label=f"PC3 Curvature ({var_df.iloc[2, 1]:.1f}%)")
    axes[1, 0].set_xticks(mat_x)
    axes[1, 0].set_xticklabels(pca_ust.tenor_cols, rotation=45)
    axes[1, 0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].set_title("Treasury curve eigenvector loadings", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Loading")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(portfolio.pnl_series, bins=100, color='lightblue', edgecolor='grey', alpha=0.8)
    axes[1, 1].axvline(pnl_99_var, color='crimson', linestyle='--', lw=2, label=f"99% 1-day VaR (${abs(pnl_99_var):.2f})")
    axes[1, 1].set_title("3-factor historical daily P&L", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Daily P&L ($)")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
