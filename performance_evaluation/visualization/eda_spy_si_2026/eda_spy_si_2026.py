# %% [markdown]
# # EDA - SPY (momentum) vs ES3.SI / STI ETF (mean reversion)
# **Window:** 1 Jan 2026 - 30 Sep 2026 (UTC)
#
# **Question:** is there evidence in the data that **SPY trends** (so a momentum rule fits it) while
# **ES3.SI reverts to a short-term mean** (so a mean-reversion rule fits it)?
#
# **Data:** `Data/market_bars_30m_training_validation.xlsx` (sheet `in`, long format, 30-minute bars).
#
# **Coverage caveat:** the file has **SPY for the full window** but **ES3.SI only from 6 Jul 2026**
# (the merged Yahoo history starts there). SPY results use ~9 months; SI results use ~3 months,
# so SI test statistics have far less power. Read SI p-values as indicative, not conclusive.
#
# **Strategy parameters tested (from `config/selected_strategies.json`, 2-hour decision bars):**
#
# | Asset | Rule | Parameters |
# |---|---|---|
# | SPY | Momentum | EMA 8 > EMA 16 **and** 8-bar return > +0.25% -> long; exit when EMA 8 <= EMA 16 or 8-bar return <= 0 |
# | ES3.SI | Mean reversion | 16-bar z-score < -1.25 -> long; exit z > +0.25 after >= 4 bars; max hold 20 bars; 4-bar cooldown |
#
# **How to use:** open in VS Code and click *Run Cell* above each `# %%` (or Shift+Enter).
# Charts render inline and are also saved as PNGs to `./output/`. Requires
# `pandas numpy matplotlib openpyxl` (`statsmodels` optional - adds ADF tests).

# %% Imports and configuration
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter

try:                                   # inline charts when run as cells in VS Code / Jupyter
    get_ipython()                      # noqa: F821
    INTERACTIVE = True
except NameError:                      # plain `python eda_spy_si_2026.py` -> save PNGs only
    INTERACTIVE = False
    import matplotlib
    matplotlib.use("Agg")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Converting to PeriodArray")

pd.set_option("display.width", 200, "display.max_columns", 30, "display.precision", 4)

WINDOW_START = pd.Timestamp("2026-01-01", tz="UTC")
WINDOW_END = pd.Timestamp("2026-10-01", tz="UTC")           # exclusive -> includes 30 Sep
VALIDATION_START = pd.Timestamp("2026-09-11", tz="UTC")     # joint validation window (EXPERIMENTS.md)

ASSETS = {
    "spy_usd": dict(label="SPY", tz="America/New_York", color="#2a78d6", strategy="Momentum"),
    "sti_es3_sgd": dict(label="ES3.SI (STI ETF)", tz="Asia/Singapore", color="#eb6834", strategy="Mean reversion"),
}
SPY, SI = "spy_usd", "sti_es3_sgd"

DECISION_FACTOR = 4                                   # 2h decision = 4 x 30m bars
SPY_FAST, SPY_SLOW, SPY_LOOKBACK, SPY_ENTRY, SPY_EXIT = 8, 16, 8, 0.0025, 0.0
SI_WINDOW, SI_ENTRY_Z, SI_EXIT_Z, SI_MIN_HOLD, SI_COOLDOWN, SI_MAX_HOLD = 16, -1.25, 0.25, 4, 4, 20


def find_data_file(name="market_bars_30m_training_validation.xlsx") -> Path:
    """Walk up from this file (or the current folder in interactive mode) to find Data/<name>."""
    try:
        start = Path(__file__).resolve().parent
    except NameError:
        start = Path.cwd().resolve()
    for p in [start, *start.parents]:
        if (p / "Data" / name).exists():
            return p / "Data" / name
    raise FileNotFoundError(f"Could not find Data/{name} above {start}. Set DATA_PATH manually.")


DATA_PATH = find_data_file()
try:
    OUT = Path(__file__).resolve().parent / "output"
except NameError:
    OUT = Path.cwd() / "output"
OUT.mkdir(parents=True, exist_ok=True)

# Chart style: light surface, recessive grid, thin marks
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
GOOD, BAD, PURPLE = "#1baf7a", "#e34948", "#6250d6"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.labelsize": 9.5,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "legend.frameon": False, "legend.fontsize": 8.5,
    "font.size": 10, "figure.dpi": 100,
})


def show(fig, name: str, title: str | None = None):
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    if INTERACTIVE:
        plt.show()
    else:
        plt.close(fig)
    print(f"saved -> output/{name}")


print("Data :", DATA_PATH)
print("Out  :", OUT)

# %% [markdown]
# ## 1. Load and filter the data
# Three views of each asset are built:
# * **30m bars** - raw, used for intraday checks;
# * **2h decision bars** - the bar the strategies actually trade on (blocks of four contiguous 30m bars
#   inside a session, last bar kept - same construction as `src/portfolio_simulation`);
# * **daily closes** - for risk context.

# %% Load
raw_all = pd.read_excel(DATA_PATH, sheet_name=0)
raw_all = raw_all[raw_all["instrument_id"].isin(ASSETS)].copy()
raw = raw_all.copy()
raw["t_open"] = pd.to_datetime(raw["bar_open_time_utc"], utc=True)
raw["t_close"] = pd.to_datetime(raw["bar_close_time_utc"], utc=True)
raw = raw[(raw["t_open"] >= WINDOW_START) & (raw["t_open"] < WINDOW_END)]


def prep_bars(g: pd.DataFrame, inst: str) -> pd.DataFrame:
    g = g.sort_values("t_open").reset_index(drop=True)
    g["session"] = pd.to_datetime(g["session_date_local"]).dt.date
    g["t_local"] = g["t_open"].dt.tz_convert(ASSETS[inst]["tz"])
    g["logp"] = np.log(g["adjusted_close"])
    g["ret"] = g["logp"].diff()
    g["intraday"] = g["session"].eq(g["session"].shift())
    g["ret_intraday"] = g["ret"].where(g["intraday"])
    return g


def decision_bars(b: pd.DataFrame) -> pd.DataFrame:
    b = b.copy()
    new_seg = b["t_open"].diff().ne(pd.Timedelta(minutes=30)) | b["session"].ne(b["session"].shift())
    b["_seg"] = new_seg.cumsum()
    b["_blk"] = b.groupby("_seg").cumcount() // DECISION_FACTOR
    d = b.groupby(["_seg", "_blk"]).tail(1).drop(columns=["_seg", "_blk"]).reset_index(drop=True)
    d["ret"] = d["logp"].diff()
    return d


def daily_closes(b: pd.DataFrame) -> pd.DataFrame:
    d = b.groupby("session").agg(close=("adjusted_close", "last"), volume=("volume", "sum"),
                                 t=("t_close", "last")).reset_index()
    d["ret"] = np.log(d["close"]).diff()
    return d


bars = {k: prep_bars(g, k) for k, g in raw.groupby("instrument_id")}
dec = {k: decision_bars(v) for k, v in bars.items()}
daily = {k: daily_closes(v) for k, v in bars.items()}

coverage = pd.DataFrame([{
    "asset": ASSETS[k]["label"],
    "first_bar": bars[k]["t_local"].iloc[0].strftime("%Y-%m-%d %H:%M"),
    "last_bar": bars[k]["t_local"].iloc[-1].strftime("%Y-%m-%d %H:%M"),
    "sessions": len(daily[k]), "bars_30m": len(bars[k]), "decision_bars_2h": len(dec[k]),
    "bars_per_session": round(len(bars[k]) / len(daily[k]), 1),
} for k in ASSETS])
coverage

# %% Data quality
qc = []
for k, b in bars.items():
    qc.append({
        "asset": ASSETS[k]["label"],
        "quality_pass_%": (b["quality_status"] == "pass").mean() * 100,
        "bars_with_synthetic_subbars": int((b["synthetic_subbar_count"].fillna(0) > 0).sum()),
        "bars_with_missing_subbars": int((b["missing_subbar_count"].fillna(0) > 0).sum()),
        "zero_volume_bars": int((b["volume"].fillna(0) == 0).sum()),
        "max_abs_30m_return_%": b["ret_intraday"].abs().max() * 100,
        "bar_origin": ", ".join(sorted(b["bar_origin"].dropna().unique())),
    })
pd.DataFrame(qc)

# %% [markdown]
# ## 2. Price path and headline risk
# First look: how each asset moved through the window, and how volatile it was.

# %% Fig 1 - price overview
fig, axes = plt.subplots(2, 2, figsize=(14, 6.5), gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
for c, k in enumerate(ASSETS):
    m, d = ASSETS[k], daily[k]
    ax = axes[0, c]
    ax.plot(d["t"], d["close"], color=m["color"])
    tot = d["close"].iloc[-1] / d["close"].iloc[0] - 1
    ax.set_title(f"{m['label']} daily adj. close  ({d['session'].iloc[0]:%d %b} - {d['session'].iloc[-1]:%d %b}: {tot:+.1%})",
                 loc="left")
    ax.axvspan(VALIDATION_START, WINDOW_END, color=MUTED, alpha=0.12, lw=0)
    ax.text(VALIDATION_START, ax.get_ylim()[1], " validation", fontsize=8, color=INK2, va="top")
    axes[1, c].bar(d["t"], d["volume"], color=m["color"], alpha=0.55, width=0.8)
    axes[1, c].set_ylabel("Volume")
    axes[1, c].xaxis.set_major_formatter(mdates.DateFormatter("%b" if k == SPY else "%d %b"))
show(fig, "01_price_volume.png", "1. Price and volume, 1 Jan - 30 Sep 2026")

# %% Display Figure 1

from IPython.display import display

display(fig)

# %% Summary statistics
def summary(k):
    d, r2 = daily[k], dec[k]["ret"].dropna()
    rd = d["ret"].dropna()
    dd = d["close"] / d["close"].cummax() - 1
    ann_ret, ann_vol = rd.mean() * 252, rd.std() * math.sqrt(252)
    up_days = (rd > 0).mean()
    return {
        "asset": ASSETS[k]["label"], "total_return": d["close"].iloc[-1] / d["close"].iloc[0] - 1,
        "ann_return_log": ann_ret, "ann_vol": ann_vol, "sharpe_no_rf": ann_ret / ann_vol,
        "max_drawdown": dd.min(), "pct_up_days": up_days,
        "2h_ret_sd": r2.std(), "2h_skew": r2.skew(), "2h_excess_kurt": r2.kurt(),
    }


t_summary = pd.DataFrame([summary(k) for k in ASSETS])
t_summary

# %% Fig 2 - monthly returns
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
monthly_rows = []
for ax, k in zip(axes, ASSETS):
    m, d = ASSETS[k], daily[k].copy()
    d["month"] = pd.to_datetime(d["session"]).dt.to_period("M")
    mr = d.groupby("month")["close"].agg(["first", "last"])
    prev_last = mr["last"].shift()
    mr["ret"] = (mr["last"] / prev_last.fillna(mr["first"])) - 1
    ax.bar(mr.index.strftime("%b"), mr["ret"], color=[GOOD if v > 0 else BAD for v in mr["ret"]], width=0.6)
    for i, v in enumerate(mr["ret"]):
        ax.text(i, v, f"{v:+.1%}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5, color=INK)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title(f"{m['label']} monthly return  ({(mr['ret'] > 0).sum()}/{len(mr)} up months)", loc="left")
    for p, v in mr["ret"].items():
        monthly_rows.append({"asset": m["label"], "month": str(p), "return": v})
show(fig, "02_monthly_returns.png", "2. Monthly returns (first SI month is a partial month from 6 Jul)")
t_monthly = pd.DataFrame(monthly_rows)

# %% [markdown]
# ## 3. Return distribution
# Fat tails matter for both rules: momentum gets whipsawed by single-bar shocks, and z-score thresholds
# are set in standard-deviation units that assume roughly normal noise.

# %% Fig 3 - distribution + QQ
fig, axes = plt.subplots(1, 4, figsize=(15, 4), gridspec_kw={"width_ratios": [1.4, 1, 1.4, 1]})
for i, k in enumerate(ASSETS):
    m, r = ASSETS[k], dec[k]["ret"].dropna()
    mu, sd = r.mean(), r.std()
    ax = axes[2 * i]
    lim = np.quantile(np.abs(r), 0.995)
    ax.hist(r, bins=np.linspace(-lim, lim, 50), density=True, color=m["color"], alpha=0.6, edgecolor=SURFACE, lw=0.4)
    xs = np.linspace(-lim, lim, 300)
    ax.plot(xs, np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi)), color=INK, lw=1.1,
            label="Normal")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax.set_title(f"{m['label']} 2h log returns", loc="left")
    ax.text(0.02, 0.95, f"n={r.size}\nsd={sd:.3%}\nskew={r.skew():.2f}\nex.kurt={r.kurt():.1f}",
            transform=ax.transAxes, va="top", fontsize=8.5, color=INK2)
    ax.legend(loc="upper right")
    # QQ vs normal (no scipy needed)
    z = np.sort((r - mu) / sd)
    p = (np.arange(1, z.size + 1) - 0.5) / z.size
    from statistics import NormalDist
    q = np.array([NormalDist().inv_cdf(v) for v in p])
    axq = axes[2 * i + 1]
    axq.scatter(q, z, s=6, color=m["color"], alpha=0.7)
    axq.plot([-4, 4], [-4, 4], color=INK2, lw=0.8, ls="--")
    axq.set_xlim(-4, 4); axq.set_ylim(min(-4, z.min() - 0.3), max(4, z.max() + 0.3))
    axq.set_title("QQ vs normal", loc="left")
    axq.set_xlabel("Normal quantile"); axq.set_ylabel("Sample (sd units)")
show(fig, "03_return_distribution.png", "3. Return distribution on 2h decision bars")

# %% [markdown]
# ## 4. Core evidence - serial dependence of returns
# The cleanest test for *momentum vs mean reversion* is the sign of return autocorrelation:
# * **positive autocorrelation / variance ratio > 1 / Hurst > 0.5** -> moves persist -> momentum;
# * **negative autocorrelation / variance ratio < 1 / Hurst < 0.5** -> moves reverse -> mean reversion.
#
# Three complementary statistics follow: the ACF, the Lo-MacKinlay variance-ratio profile and the Hurst exponent.

# %% Statistics helpers
def norm_sf(z): return 0.5 * math.erfc(z / math.sqrt(2))


def acf(x, nlags):
    x = pd.Series(x).dropna().to_numpy(); x = x - x.mean(); den = np.dot(x, x)
    return np.array([np.dot(x[k:], x[:-k]) / den for k in range(1, nlags + 1)])


def ljung_box(x, lags):
    n = pd.Series(x).dropna().size; r = acf(x, lags)
    q = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, lags + 1)))
    try:
        from scipy.stats import chi2
        p = float(chi2.sf(q, lags))
    except Exception:
        p = np.nan
    return float(q), p


def variance_ratio(r, q):
    """Lo-MacKinlay (1988) VR(q) with heteroskedasticity-robust z*."""
    r = pd.Series(r).dropna().to_numpy(); n = r.size; mu = r.mean()
    var1 = np.sum((r - mu) ** 2) / (n - 1)
    rq = np.convolve(r, np.ones(q), mode="valid")
    varq = np.sum((rq - q * mu) ** 2) / (q * (n - q + 1) * (1 - q / n))
    vr = varq / var1
    dev2 = (r - mu) ** 2; theta = 0.0
    for j in range(1, q):
        theta += (2 * (q - j) / q) ** 2 * np.sum(dev2[j:] * dev2[:-j]) / np.sum(dev2) ** 2
    z = (vr - 1) / math.sqrt(theta) if theta > 0 else np.nan
    return float(vr), float(z), float(2 * norm_sf(abs(z)))


def hurst(logp, lags=range(1, 25)):
    x = pd.Series(logp).dropna().to_numpy(); lags = np.array(list(lags))
    tau = np.array([np.std(x[l:] - x[:-l]) for l in lags])
    return float(np.polyfit(np.log(lags), np.log(tau), 1)[0]), lags, tau


def adf_p(x):
    try:
        from statsmodels.tsa.stattools import adfuller
        return float(adfuller(pd.Series(x).dropna(), autolag="AIC")[1])
    except Exception:
        return None


# %% Fig 4 - ACF
NLAGS = 12
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
acf_rows = []
for ax, k in zip(axes, ASSETS):
    m, r = ASSETS[k], dec[k]["ret"]
    a = acf(r, NLAGS); n = r.dropna().size; band = 1.96 / math.sqrt(n)
    lags = np.arange(1, NLAGS + 1)
    ax.bar(lags, a, color=[m["color"] if abs(v) > band else MUTED for v in a], width=0.6)
    ax.axhspan(-band, band, color=MUTED, alpha=0.12, lw=0)
    ax.axhline(0, color=INK2, lw=0.8)
    q4, p4 = ljung_box(r, 4)
    ax.text(0.98, 0.95, f"sum ACF lag 1-4 = {a[:4].sum():+.3f}\nsum ACF lag 1-8 = {a[:8].sum():+.3f}\n"
                        f"Ljung-Box Q(4) = {q4:.1f}" + (f" (p={p4:.3f})" if not np.isnan(p4) else "") +
            f"\n95% band = +/-{band:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)
    ax.set_xlabel("Lag (2h decision bars; SPY ~3-4 bars/day, SI ~4 bars/day)")
    ax.set_title(f"{m['label']} autocorrelation of 2h returns", loc="left")
    acf_rows.append({"asset": m["label"], "n": n, **{f"lag{i}": a[i - 1] for i in (1, 2, 3, 4)},
                     "sum_1_4": a[:4].sum(), "sum_1_8": a[:8].sum(), "LB_Q4": q4, "LB_p4": p4,
                     "band95": band, "acf1_30m": acf(bars[k]["ret_intraday"], 1)[0],
                     "acf1_daily": acf(daily[k]["ret"], 1)[0]})
axes[0].set_ylabel("Autocorrelation")
show(fig, "04_autocorrelation.png", "4. Serial correlation (coloured bars are outside the 95% band)")
t_acf = pd.DataFrame(acf_rows)
t_acf

# %% Fig 5 - variance ratio profile + Hurst
QS = [2, 3, 4, 6, 8, 12, 16]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
vr_rows = []
for k in ASSETS:
    m, r = ASSETS[k], dec[k]["ret"]
    res = [variance_ratio(r, q) for q in QS]
    axes[0].plot(QS, [x[0] for x in res], marker="o", ms=5, color=m["color"], label=m["label"])
    for q, (v, z, p) in zip(QS, res):
        vr_rows.append({"asset": m["label"], "q_bars": q, "horizon_h": 2 * q, "VR": v, "z_robust": z, "p": p})
        if p < 0.10:
            axes[0].scatter([q], [v], s=120, facecolors="none", edgecolors=m["color"], lw=1.2)
    H, lags, tau = hurst(dec[k]["logp"])
    axes[1].loglog(lags, tau / tau[0], marker="o", ms=3.5, color=m["color"], label=f"{m['label']}  H = {H:.2f}")
    vr_rows.append({"asset": m["label"], "q_bars": "Hurst", "VR": H})
axes[0].axhline(1, color=INK2, lw=0.9, ls="--")
axes[0].text(QS[-1], 1.0, "random walk ", ha="right", va="bottom", fontsize=8, color=INK2)
axes[0].set_xlabel("q (aggregation in 2h bars)"); axes[0].set_ylabel("VR(q)")
axes[0].set_title("Variance ratio profile (circled: p < 0.10, robust z)", loc="left"); axes[0].legend()
axes[1].loglog(lags, (lags / lags[0]) ** 0.5, color=INK2, ls="--", lw=0.9, label="H = 0.5 random walk")
axes[1].set_xlabel("Lag (2h bars)"); axes[1].set_ylabel("Std of log-price change (normalised)")
axes[1].set_title("Hurst exponent (slope of log-log)", loc="left"); axes[1].legend()
show(fig, "05_variance_ratio_hurst.png", "5. VR > 1 & H > 0.5 = trending;  VR < 1 & H < 0.5 = mean-reverting")
t_vr = pd.DataFrame(vr_rows)
t_vr

# %% [markdown]
# ### 4b. Lead-lag map: does the past *L*-bar return predict the next *H*-bar return?
# This generalises the ACF to the horizons the strategies actually use (SPY looks back 8 bars; SI uses a
# 16-bar window). **Blue (positive) = continuation, red (negative) = reversal.**
# Windows overlap, so treat magnitudes as descriptive rather than as independent tests.

# %% Fig 6 - lead-lag heatmap
LH = [1, 2, 4, 8, 16]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ll_rows = []
for ax, k in zip(axes, ASSETS):
    lp = dec[k]["logp"]
    M = np.full((len(LH), len(LH)), np.nan)
    for i, L in enumerate(LH):
        past = lp - lp.shift(L)
        for j, H_ in enumerate(LH):
            fut = lp.shift(-H_) - lp
            M[i, j] = past.corr(fut)
            ll_rows.append({"asset": ASSETS[k]["label"], "lookback_L": L, "horizon_H": H_, "corr": M[i, j]})
    vmax = 0.3
    im = ax.imshow(M, cmap="RdBu", vmin=-vmax, vmax=vmax)
    for i in range(len(LH)):
        for j in range(len(LH)):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(M[i, j]) > 0.18 else INK)
    ax.set_xticks(range(len(LH)), [f"{h}" for h in LH]); ax.set_yticks(range(len(LH)), [f"{l}" for l in LH])
    ax.set_xlabel("Forward horizon H (2h bars)"); ax.set_ylabel("Lookback L (2h bars)")
    ax.set_title(f"{ASSETS[k]['label']}  corr(past L, next H)", loc="left"); ax.grid(False)
fig.colorbar(im, ax=axes, shrink=0.85, label="correlation")
fig.suptitle("6. Lead-lag correlation map", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
fig.savefig(OUT / "06_lead_lag_heatmap.png", dpi=150, bbox_inches="tight")
plt.show() if INTERACTIVE else plt.close(fig)
print("saved -> output/06_lead_lag_heatmap.png")
t_leadlag = pd.DataFrame(ll_rows)

# %% [markdown]
# ### 4c. Is the behaviour stable month by month?
# A single full-sample statistic can be driven by one episode. Here the lag-1 ACF and VR(4) are
# recomputed for each calendar month.

# %% Fig 7 - monthly stability of dependence
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
stab_rows = []
for k in ASSETS:
    d = dec[k].copy()
    d["month"] = d["t_close"].dt.tz_convert(ASSETS[k]["tz"]).dt.to_period("M")
    for mth, g in d.groupby("month"):
        r = g["ret"].dropna()
        if r.size < 30:
            continue
        stab_rows.append({"asset": ASSETS[k]["label"], "key": k, "month": str(mth), "n": r.size,
                          "acf1": acf(r, 1)[0], "acf_sum_1_4": acf(r, 4).sum(), "VR4": variance_ratio(r, 4)[0]})
t_stab = pd.DataFrame(stab_rows)
months = sorted(t_stab["month"].unique())
w = 0.38
for ax, col, lab, ref in [(axes[0], "acf_sum_1_4", "Sum of ACF lags 1-4", 0), (axes[1], "VR4", "VR(4) - 1", 1)]:
    for i, k in enumerate(ASSETS):
        s = t_stab[t_stab["key"] == k].set_index("month").reindex(months)
        x = np.arange(len(months)) + (i - 0.5) * w
        ax.bar(x, s[col] - ref, width=w, color=ASSETS[k]["color"], label=ASSETS[k]["label"])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(np.arange(len(months)), [pd.Period(mm).strftime("%b") for mm in months])
    ax.set_title(f"{lab} by month (> 0 trending, < 0 reverting)", loc="left"); ax.legend()
show(fig, "07_monthly_stability.png", "7. Stability of serial dependence across months")
t_stab.drop(columns="key")

# %% [markdown]
# ## 5. Signal-level evidence - do the strategy signals predict forward returns?
# Both signals are computed on **both** assets. If the strategy choice is right we expect:
# * the **momentum** signal to slope **upward** for SPY (strong past -> strong future);
# * the **z-score** signal to slope **downward** for SI (oversold -> bounce, overbought -> fade).

# %% Features
def add_features(d):
    d = d.copy(); p = d["adjusted_close"]
    d["ema_fast"] = p.ewm(span=SPY_FAST, adjust=False, min_periods=SPY_FAST).mean()
    d["ema_slow"] = p.ewm(span=SPY_SLOW, adjust=False, min_periods=SPY_SLOW).mean()
    d["mom"] = p.pct_change(SPY_LOOKBACK, fill_method=None)
    roll = p.rolling(SI_WINDOW, min_periods=SI_WINDOW)
    d["ma"], d["sd"] = roll.mean(), roll.std(ddof=0)
    d["z"] = (p - d["ma"]) / d["sd"].replace(0, np.nan)
    for h in (1, 2, 4, 8):
        d[f"fwd{h}"] = np.log(p.shift(-h) / p)
    return d


feat = {k: add_features(v) for k, v in dec.items()}


# %% Strategy rules (simplified replicas of src/trading_strategy)
def mom_positions(d):
    pos, out = 0, []
    for f, s, mm in d[["ema_fast", "ema_slow", "mom"]].to_numpy():
        if np.isnan(f) or np.isnan(s) or np.isnan(mm):
            out.append(0); continue
        if pos == 0 and f > s and mm > SPY_ENTRY: pos = 1
        elif pos == 1 and (f <= s or mm <= SPY_EXIT): pos = 0
        out.append(pos)
    return np.array(out)


def mr_positions(d):
    pos, hold, cool, out = 0, 0, 0, []
    for z in d["z"].to_numpy():
        if pos == 0:
            if cool > 0: cool -= 1
            elif not np.isnan(z) and z < SI_ENTRY_Z: pos, hold = 1, 0
        else:
            hold += 1
            if (hold >= SI_MIN_HOLD and not np.isnan(z) and z > SI_EXIT_Z) or hold >= SI_MAX_HOLD:
                pos, cool = 0, SI_COOLDOWN
        out.append(pos)
    return np.array(out)


# %% Bucket helper
def bucket_table(d, signal, fwd):
    x = d[[signal, fwd]].dropna().copy()
    if signal == "z":
        edges = [-np.inf, -1.25, -0.5, 0.5, 1.25, np.inf]
        x["bucket"] = pd.cut(x[signal], edges, labels=["< -1.25", "-1.25..-0.5", "-0.5..0.5", "0.5..1.25", "> 1.25"])
    else:
        x["bucket"] = pd.qcut(x[signal], 5, labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"])
    g = x.groupby("bucket", observed=False)[fwd]
    t = pd.DataFrame({"mean_fwd": g.mean(), "n": g.size(), "hit_rate": g.apply(lambda s: (s > 0).mean()),
                      "se": g.std() / np.sqrt(g.size())})
    return t, x[signal].corr(x[fwd], method="spearman")


# %% Fig 8 - conditional forward returns
FWD = "fwd4"
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
cond_rows = []
for r_i, (sig, xl) in enumerate([("mom", f"Past {SPY_LOOKBACK}-bar return quintile"),
                                 ("z", f"{SI_WINDOW}-bar z-score bucket")]):
    for c_i, k in enumerate(ASSETS):
        ax = axes[r_i, c_i]
        t, rho = bucket_table(feat[k], sig, FWD)
        xs = np.arange(len(t))
        ax.bar(xs, t["mean_fwd"], color=[GOOD if v > 0 else BAD for v in t["mean_fwd"]], width=0.6, alpha=0.85)
        ax.errorbar(xs, t["mean_fwd"], yerr=1.96 * t["se"], fmt="none", ecolor=INK2, lw=1, capsize=3)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(xs, [f"{b}\nhit {h:.0%} | n={n}" for b, h, n in zip(t.index.astype(str), t["hit_rate"], t["n"])],
                      fontsize=8)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))
        ax.set_xlabel(xl); ax.set_ylabel("Mean next 4-bar (~1 day) log return")
        native = (k == SPY and sig == "mom") or (k == SI and sig == "z")
        ax.set_title(f"{ASSETS[k]['label']} | {'momentum' if sig == 'mom' else 'z-score'} signal"
                     f"{'  (strategy signal)' if native else ''}", loc="left")
        ax.text(0.98, 0.95, f"top - bottom = {t['mean_fwd'].iloc[-1] - t['mean_fwd'].iloc[0]:+.3%}\n"
                            f"Spearman rho = {rho:+.3f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, color=INK)
        for b, row in t.iterrows():
            cond_rows.append({"asset": ASSETS[k]["label"], "signal": sig, "bucket": str(b), **row.to_dict(),
                              "spearman_all": rho})
show(fig, "08_conditional_forward_returns.png",
     "8. Next-day return by signal bucket (95% CI; forward windows overlap so CIs are optimistic)")
t_cond = pd.DataFrame(cond_rows)

# %% [markdown]
# ## 6. SPY - what the momentum rule sees
# Price with the EMA 8 / EMA 16 pair and the 8-bar return filter. Green shading = entry condition true.

# %% Fig 9 - SPY trend regimes
d = feat[SPY]; m = ASSETS[SPY]
long_ok = (d["ema_fast"] > d["ema_slow"]) & (d["mom"] > SPY_ENTRY)
fig, axes = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [2.3, 1]}, sharex=True)
ax = axes[0]
ax.plot(d["t_close"], d["adjusted_close"], color=MUTED, lw=0.9, label="Adj. close (2h)")
ax.plot(d["t_close"], d["ema_fast"], color=m["color"], lw=1.3, label=f"EMA {SPY_FAST}")
ax.plot(d["t_close"], d["ema_slow"], color=PURPLE, lw=1.3, label=f"EMA {SPY_SLOW}")
y0, y1 = d["adjusted_close"].min() * 0.985, d["adjusted_close"].max() * 1.01
ax.fill_between(d["t_close"], y0, y1, where=long_ok, color=GOOD, alpha=0.12, lw=0, label="Entry condition true")
ax.set_ylim(y0, y1); ax.legend(loc="upper left", ncol=4)
ax.set_title(f"SPY trend regimes - entry condition true on {long_ok.mean():.0%} of decision bars", loc="left")
axes[1].plot(d["t_close"], d["mom"], color=m["color"], lw=0.8)
axes[1].axhline(SPY_ENTRY, color=GOOD, ls="--", lw=0.9); axes[1].axhline(0, color=INK2, lw=0.8)
axes[1].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0)); axes[1].set_ylabel("8-bar return")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
show(fig, "09_spy_trend_regimes.png", "9. SPY: EMA 8/16 + 8-bar momentum filter")

# %% [markdown]
# ### 6b. The momentum rule as a regime filter
# Momentum on an index ETF earns money in two ways: (i) return *autocorrelation* (tested in section 4), and
# (ii) *regime selection*: staying long while the trend is up, so you keep most of the drift, and stepping
# aside in down-trends, which cuts volatility and drawdowns. This cell tests (ii): next-bar returns when the
# rule is long vs flat, and how much of the asset's drawdown time the rule spends out of the market.

# %% Fig 9b - regime filter
def rule_pos(k):
    return mom_positions(feat[k]) if k == SPY else mr_positions(feat[k])


fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
regime_rows = []
for ax, k in zip(axes, ASSETS):
    dd_ = feat[k]
    pos = mom_positions(dd_)
    r_next = dd_["ret"].shift(-1)
    ok = dd_["mom"].notna() & r_next.notna()
    bpy = len(dd_) / len(daily[k]) * 252
    eq = np.exp(dd_["logp"] - dd_["logp"].iloc[0])
    in_dd = (eq / eq.cummax() - 1) < -0.02                         # bars where the asset is >2% below its peak
    for lab, mask in [("Momentum ON (long)", (pos == 1) & ok), ("Momentum OFF (flat)", (pos == 0) & ok)]:
        s = r_next[mask]
        regime_rows.append({"asset": ASSETS[k]["label"], "regime": lab, "share_of_bars": mask.sum() / ok.sum(),
                            "ann_mean_next_bar": s.mean() * bpy, "ann_vol_next_bar": s.std() * math.sqrt(bpy),
                            "sharpe": s.mean() / s.std() * math.sqrt(bpy),
                            "share_of_bars_in_>2%_drawdown": in_dd[mask].mean()})
    t = pd.DataFrame([r for r in regime_rows if r["asset"] == ASSETS[k]["label"]])
    xx = np.arange(2)
    ax.bar(xx - 0.2, t["ann_mean_next_bar"], 0.38, color=ASSETS[k]["color"], label="Ann. mean")
    ax.bar(xx + 0.2, t["ann_vol_next_bar"], 0.38, color=ASSETS[k]["color"], alpha=0.4, label="Ann. vol")
    for xi in xx:
        for off, col in [(-0.2, "ann_mean_next_bar"), (0.2, "ann_vol_next_bar")]:
            v = t[col].iloc[xi]
            ax.text(xi + off, v, f"{v:.0%}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5, color=INK)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(xx, [f"{r}\n{s:.0%} of bars | Sharpe {sh:.2f}" for r, s, sh in
                       zip(t["regime"], t["share_of_bars"], t["sharpe"])])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title(f"{ASSETS[k]['label']} - next-bar return by momentum regime", loc="left"); ax.legend(loc="upper right")
show(fig, "09b_momentum_regime_filter.png", "9b. Momentum rule as a regime filter (next 2h bar, annualised)")
t_regime = pd.DataFrame(regime_rows)
t_regime

# %% Trend-run statistics: how long do SPY trend regimes last vs SI?
def run_lengths(flag: pd.Series) -> np.ndarray:
    f = flag.astype(int).to_numpy()
    runs, cur = [], 0
    for v in f:
        if v: cur += 1
        elif cur: runs.append(cur); cur = 0
    if cur: runs.append(cur)
    return np.array(runs)


run_rows = []
for k in ASSETS:
    dd_ = feat[k]
    up = (dd_["ema_fast"] > dd_["ema_slow"]).where(dd_["ema_slow"].notna(), False)
    ru, rd = run_lengths(up), run_lengths(~up & dd_["ema_slow"].notna())
    sign = np.sign(dd_["ret"].dropna())
    same_sign = (sign == sign.shift()).iloc[1:].mean()
    run_rows.append({"asset": ASSETS[k]["label"], "up_regimes": len(ru), "median_up_run_bars": np.median(ru),
                     "mean_up_run_bars": ru.mean(), "median_down_run_bars": np.median(rd),
                     "pct_time_ema_up": up.mean(), "P(next 2h same sign)": same_sign})
t_runs = pd.DataFrame(run_rows)
t_runs

# %% [markdown]
# ## 7. ES3.SI - what the mean-reversion rule sees
# Price around its 16-bar rolling mean with the +/-1.25 sd band, and the z-score with entry/exit levels.
# The x-axis is *trading time* (no overnight/weekend gaps) so SGX sessions are readable.

# %% Fig 10 - SI z-score bands
d = feat[SI]; m = ASSETS[SI]
x = np.arange(len(d))
loc_dates = d["t_close"].dt.tz_convert(m["tz"])
wk = loc_dates.dt.strftime("%G-%V"); ticks = np.where(wk.ne(wk.shift()))[0]
fig, axes = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [2.3, 1]}, sharex=True)
ax = axes[0]
ax.plot(x, d["adjusted_close"], color=m["color"], lw=1.3, label="Adj. close (2h)")
ax.plot(x, d["ma"], color=INK2, lw=1.0, ls="--", label=f"{SI_WINDOW}-bar mean")
ax.fill_between(x, d["ma"] - 1.25 * d["sd"], d["ma"] + 1.25 * d["sd"], color=MUTED, alpha=0.15, lw=0,
                label="+/-1.25 sd band")
cross = ((d["z"] < SI_ENTRY_Z) & (d["z"].shift() >= SI_ENTRY_Z)).to_numpy()
ax.scatter(x[cross], d.loc[cross, "adjusted_close"], s=45, color=GOOD, zorder=3, edgecolors=SURFACE, lw=1.3,
           label=f"z crosses below {SI_ENTRY_Z}")
i0 = int(np.argmax((d["t_close"] >= VALIDATION_START).to_numpy()))
ax.axvspan(i0, len(d) - 1, color=MUTED, alpha=0.12, lw=0)
ax.legend(loc="upper left", ncol=4)
inside = ((d["z"].abs() <= 1.25) & d["z"].notna()).sum() / d["z"].notna().sum()
ax.set_title(f"ES3.SI price vs rolling mean - inside the +/-1.25 sd band {inside:.0%} of the time", loc="left")
axz = axes[1]
axz.plot(x, d["z"], color=m["color"], lw=0.9)
axz.axhline(SI_ENTRY_Z, color=GOOD, ls="--", lw=0.9); axz.axhline(SI_EXIT_Z, color=BAD, ls="--", lw=0.9)
axz.axhline(0, color=INK2, lw=0.6); axz.set_ylabel("z-score")
axz.set_xticks(ticks, [loc_dates.iloc[i].strftime("%d %b") for i in ticks], rotation=0)
axz.set_xlabel("2h decision bars, trading time (ticks = first bar of each week)")
show(fig, "10_si_zscore_bands.png", "10. ES3.SI: 16-bar z-score bands (entry -1.25, exit +0.25)")

# %% Reversion speed and stationarity
hl_rows = []
for k in ASSETS:
    dd_ = feat[k]
    dev = np.log(dd_["adjusted_close"] / dd_["ma"]).dropna()
    y, xl = dev.diff().iloc[1:].to_numpy(), dev.shift(1).iloc[1:].to_numpy()
    b = np.linalg.lstsq(np.column_stack([np.ones_like(xl), xl]), y, rcond=None)[0][1]
    hl = -math.log(2) / math.log(1 + b) if -1 < b < 0 else np.inf
    # half-life of the log price itself (AR(1) on level) - long for a trending asset
    lp = dd_["logp"].dropna()
    b_lvl = np.linalg.lstsq(np.column_stack([np.ones(len(lp) - 1), lp.shift(1).iloc[1:]]),
                            lp.diff().iloc[1:], rcond=None)[0][1]
    hl_lvl = -math.log(2) / math.log(1 + b_lvl) if -1 < b_lvl < 0 else np.inf
    hl_rows.append({"asset": ASSETS[k]["label"], "half_life_dev_bars": hl, "half_life_dev_hours": 2 * hl,
                    "half_life_price_level_bars": hl_lvl,
                    "ADF_p_log_price": adf_p(dd_["logp"]), "ADF_p_dev_from_16bar_mean": adf_p(dev)})
t_hl = pd.DataFrame(hl_rows)
t_hl

# %% [markdown]
# ### 7b. Event studies - what happens *after* each strategy's entry signal?
# Average cumulative log return over the next 12 decision bars after (a) a fresh momentum entry and
# (b) a fresh z < -1.25 crossing, net of the asset's unconditional average drift over the same
# horizon (so SPY's general uptrend does not flatter the momentum event).

# %% Fig 11 - event studies
HORIZON = 12
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=False)
ev_rows = []
for ax, (ev_name, ev_fn) in zip(axes, [
        ("Momentum entry (EMA8>EMA16 & 8-bar ret > 0.25%, fresh)",
         lambda d: ((d["ema_fast"] > d["ema_slow"]) & (d["mom"] > SPY_ENTRY)) &
                   ~((d["ema_fast"].shift() > d["ema_slow"].shift()) & (d["mom"].shift() > SPY_ENTRY))),
        ("Oversold entry (z crosses below -1.25)",
         lambda d: (d["z"] < SI_ENTRY_Z) & (d["z"].shift() >= SI_ENTRY_Z))]):
    for k in ASSETS:
        dd_ = feat[k]; ev = np.where(ev_fn(dd_).fillna(False).to_numpy())[0]
        p = dd_["logp"].to_numpy(); drift = dd_["ret"].mean()
        paths = np.array([p[i:i + HORIZON + 1] - p[i] - drift * np.arange(HORIZON + 1)
                          for i in ev if i + HORIZON < len(p)])
        if len(paths) < 3:
            continue
        mu, se = paths.mean(0), paths.std(0, ddof=1) / np.sqrt(len(paths))
        kk = np.arange(HORIZON + 1)
        ax.plot(kk, mu, marker="o", ms=3.5, color=ASSETS[k]["color"], label=f"{ASSETS[k]['label']} (n={len(paths)})")
        ax.fill_between(kk, mu - 1.96 * se, mu + 1.96 * se, color=ASSETS[k]["color"], alpha=0.12, lw=0)
        ev_rows.append({"event": ev_name.split(" (")[0], "asset": ASSETS[k]["label"], "n_events": len(paths),
                        "excess_ret_4bars": mu[4], "excess_ret_8bars": mu[8], "excess_ret_12bars": mu[12],
                        "t_stat_4bars": mu[4] / se[4] if se[4] > 0 else np.nan,
                        "hit_rate_4bars": (paths[:, 4] > 0).mean()})
    ax.axhline(0, color=INK2, lw=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))
    ax.set_xlabel("2h bars after signal"); ax.set_ylabel("Cumulative excess log return")
    ax.set_title(ev_name, loc="left", fontsize=10.5); ax.legend(loc="upper left")
show(fig, "11_event_studies.png", "11. Event studies: drift-adjusted path after each entry signal (95% band)")
t_events = pd.DataFrame(ev_rows)
t_events

# %% [markdown]
# ## 8. Putting it together - each rule on each asset (gross, illustrative)
# Simplified versions of the two rules, executed with a one-bar delay (signal at bar close, return of the
# next bar), **no costs**. This is a sanity check that the statistical evidence translates into P&L in the
# expected direction - the official after-cost results come from `run_strategy_backtest.py`.
# The *break-even cost* column shows how much round-trip cost per trade the gross edge could absorb.

# %% Fig 12 - rule comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
rule_rows = []
styles = {"Buy & hold": dict(color=MUTED, lw=1.2), "Momentum rule": dict(color="#2a78d6"),
          "Mean-reversion rule": dict(color="#eb6834")}
for ax, k in zip(axes, ASSETS):
    dd_ = feat[k]
    r_next = dd_["ret"].shift(-1).fillna(0).to_numpy()
    xv = np.arange(len(dd_)) if k == SI else dd_["t_close"]
    for name, pos in {"Buy & hold": np.ones(len(dd_)), "Momentum rule": mom_positions(dd_),
                      "Mean-reversion rule": mr_positions(dd_)}.items():
        pnl = pos * r_next; eq = np.exp(np.cumsum(pnl))
        ax.plot(xv, eq, label=name, **styles[name])
        s = pd.Series(pnl); bars_per_yr = len(dd_) / len(daily[k]) * 252
        ann, vol = s.mean() * bars_per_yr, s.std() * math.sqrt(bars_per_yr)
        trades = int(np.sum(np.diff(np.r_[0, pos]) == 1))
        in_mkt = pnl[pos == 1]
        rule_rows.append({"asset": ASSETS[k]["label"], "rule": name, "gross_total": eq[-1] - 1,
                          "ann_ret": ann, "ann_vol": vol, "sharpe_gross": ann / vol if vol else np.nan,
                          "max_dd": (eq / np.maximum.accumulate(eq) - 1).min(),
                          "time_in_market": pos.mean(), "round_trips": trades,
                          "bar_hit_rate_in_mkt": (in_mkt > 0).mean() if in_mkt.size else np.nan,
                          "breakeven_cost_bps_per_round_trip":
                              (np.log(eq[-1]) / trades * 1e4) if (name != "Buy & hold" and trades) else np.nan})
    ax.set_title(f"{ASSETS[k]['label']} - gross growth of $1", loc="left"); ax.legend(loc="upper left")
    if k == SPY:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    else:
        ld = dd_["t_close"].dt.tz_convert(ASSETS[k]["tz"]); wk = ld.dt.strftime("%G-%V")
        tk = np.where(wk.ne(wk.shift()))[0][::2]
        ax.set_xticks(tk, [ld.iloc[i].strftime("%d %b") for i in tk])
show(fig, "12_rule_comparison_gross.png", "12. Both rules on both assets (gross, 1-bar delay, simplified)")
t_rules = pd.DataFrame(rule_rows)
t_rules

# %% [markdown]
# ### 8b. Context - SPY on the 2023-25 period the momentum rule was selected on
# Is the lack of 2h autocorrelation in SPY specific to 2026? Same statistics on SPY's earlier history
# (out of the requested window; shown for context only).

# %% SPY context: earlier history
ctx_rows = []
for lab, a, b_ in [("SPY 18 Sep 2023 - 31 Dec 2025", "2023-01-01", "2026-01-01"),
                   ("SPY 1 Jan - 30 Sep 2026 (this EDA)", "2026-01-01", "2026-10-01")]:
    g_ = raw_all[raw_all["instrument_id"] == SPY].copy()
    g_["t_open"] = pd.to_datetime(g_["bar_open_time_utc"], utc=True)
    g_["t_close"] = pd.to_datetime(g_["bar_close_time_utc"], utc=True)
    g_ = g_[(g_["t_open"] >= pd.Timestamp(a, tz="UTC")) & (g_["t_open"] < pd.Timestamp(b_, tz="UTC"))]
    b2 = prep_bars(g_, SPY); d2 = decision_bars(b2); dl = daily_closes(b2)
    ctx_rows.append({"period": lab, "decision_bars": len(d2), "acf_sum_1_4": acf(d2["ret"], 4).sum(),
                     "VR(4) 2h": variance_ratio(d2["ret"], 4)[0], "VR(16) 2h": variance_ratio(d2["ret"], 16)[0],
                     "Hurst 2h": hurst(d2["logp"])[0], "VR(5) daily": variance_ratio(dl["ret"], 5)[0],
                     "VR(20) daily": variance_ratio(dl["ret"], 20)[0],
                     "ann_return": dl["ret"].mean() * 252, "ann_vol": dl["ret"].std() * math.sqrt(252)})
t_ctx = pd.DataFrame(ctx_rows)
t_ctx

# %% [markdown]
# ## 9. Evidence scorecard

# %% Scorecard
def g(tab, asset_key, col, **filt):
    t = tab[tab["asset"] == ASSETS[asset_key]["label"]]
    for c, v in filt.items():
        t = t[t[c] == v]
    return t[col].iloc[0]


score = pd.DataFrame([
    ["Sum ACF lags 1-4 (2h)", "> 0", "< 0", g(t_acf, SPY, "sum_1_4"), g(t_acf, SI, "sum_1_4")],
    ["VR(4)", "> 1", "< 1", g(t_vr, SPY, "VR", q_bars=4), g(t_vr, SI, "VR", q_bars=4)],
    ["VR(8)", "> 1", "< 1", g(t_vr, SPY, "VR", q_bars=8), g(t_vr, SI, "VR", q_bars=8)],
    ["Hurst exponent", "> 0.5", "< 0.5", g(t_vr, SPY, "VR", q_bars="Hurst"), g(t_vr, SI, "VR", q_bars="Hurst")],
    ["corr(past 8, next 4)", "> 0", "< 0",
     g(t_leadlag, SPY, "corr", lookback_L=8, horizon_H=4), g(t_leadlag, SI, "corr", lookback_L=8, horizon_H=4)],
    ["Spearman: momentum signal vs next 4 bars", "> 0", "-",
     g(t_cond, SPY, "spearman_all", signal="mom", bucket="Q1 low"), g(t_cond, SI, "spearman_all", signal="mom", bucket="Q1 low")],
    ["Spearman: z-score vs next 4 bars", "-", "< 0",
     g(t_cond, SPY, "spearman_all", signal="z", bucket="< -1.25"), g(t_cond, SI, "spearman_all", signal="z", bucket="< -1.25")],
    ["Half-life of deviation from 16-bar mean (h)", "longer", "shorter",
     g(t_hl, SPY, "half_life_dev_hours"), g(t_hl, SI, "half_life_dev_hours")],
    ["Gross Sharpe - own rule", "momentum", "mean rev.",
     g(t_rules, SPY, "sharpe_gross", rule="Momentum rule"), g(t_rules, SI, "sharpe_gross", rule="Mean-reversion rule")],
    ["Gross Sharpe - other rule", "", "",
     g(t_rules, SPY, "sharpe_gross", rule="Mean-reversion rule"), g(t_rules, SI, "sharpe_gross", rule="Momentum rule")],
], columns=["metric", "SPY momentum needs", "SI mean-rev needs", "SPY", "SI"])
score

# %% Save all tables to Excel
with pd.ExcelWriter(OUT / "eda_spy_si_2026_stats.xlsx") as xw:
    for name, tab in [("scorecard", score), ("coverage", coverage), ("quality", pd.DataFrame(qc)),
                      ("summary", t_summary), ("monthly_returns", t_monthly), ("autocorrelation", t_acf),
                      ("variance_ratio_hurst", t_vr), ("lead_lag", t_leadlag),
                      ("monthly_stability", t_stab.drop(columns="key")), ("conditional_fwd", t_cond),
                      ("regime_filter", t_regime), ("spy_context", t_ctx), ("trend_runs", t_runs), ("half_life_adf", t_hl), ("event_studies", t_events),
                      ("rule_comparison", t_rules)]:
        tab.to_excel(xw, sheet_name=name, index=False)
print("saved -> output/eda_spy_si_2026_stats.xlsx")

# %% [markdown]
# ## 10. Observations and interpretation
# _(written from the run on the 1 Jan - 30 Sep 2026 data; numbers match the tables above)_
#
# ### 10.1 Market backdrop (sections 1-3)
# * **SPY (+12.5%, 2 Jan - 30 Sep):** three distinct phases. (1) Jan-Mar: drift lower into a ~9% drawdown
#   that bottomed at the start of April (March -5.0%). (2) April-May: a sharp V-shaped recovery,
#   **+10.5% in April and +5.3% in May**, a persistent one-directional move. (3) June-Sep: a slower grind higher
#   with a step up in early August, then a range near the highs. Annualised vol is 13%, Sharpe (no rf) 1.2,
#   and only 5/9 months were up: most of the year's return came from one trending episode.
# * **ES3.SI (+7.6%, 6 Jul - 30 Sep):** a strong rally in the first two weeks of July (July +7.3%), then a
#   **sideways range of roughly SGD 5.70-5.90** for August and September (+1.2%, -0.9%). Volatility is lower
#   (11%) and max drawdown only -2.9%. In the part of the sample that drives the statistics, price keeps
#   returning to the same band rather than trending.
# * **Distributions:** both have fat tails on 2h bars. SI is more extreme (excess kurtosis 10, skew -0.9,
#   driven by a few sharp down-bars) than SPY (3.3, +0.3). For SI this means z-score entries will sometimes
#   catch a real gap down rather than noise, which supports the max-hold (20 bars) exit in the rule.
# * **Data quality:** SPY is 100% `pass`. About 5% of SI bars are partial or synthetic-boundary backfills (Yahoo
#   merge); no zero-volume bars. This doesn't change the conclusions but adds to the small-sample caution.
#
# ### 10.2 ES3.SI: the evidence supports mean reversion
# | Lens | Result | Reads as |
# |---|---|---|
# | ACF (2h) | lag-1 = -0.11; sum lags 1-8 = -0.18 | reversal |
# | Variance ratio | VR(2)=0.90 (p=0.02), VR(3)=0.85 (p=0.04), VR(4)=0.84 (p=0.09); falls to 0.70 at 32h | reversal at the strategy's horizon, significant at short q |
# | Hurst | **0.41** (vs 0.5) | anti-persistent |
# | Lead-lag map | **every** cell with L >= 8 is negative; corr(past 16, next 8) = -0.19 | the longer it has moved, the more it gives back |
# | Month by month | VR(4) < 1 in all 3 months; lag-1 ACF of about -0.11 to -0.12 in each month | consistent, not one episode |
# | Stationarity | ADF p = **0.001** on the log *price* (SPY: 0.75); deviation from 16-bar mean half-life **8.4h** (~4 bars) | price is range-bound; deviations close in about a day |
# | Signal buckets | z < -1.25 -> next-day +0.23% (61% hit); z between 0.5 and 1.25 -> -0.13% | oversold bounces, overbought fades |
# | Event study | after z crosses below -1.25: +0.32% excess after 4 bars, +0.53% after 12, 70% hit (t ~ 1.8, n=10) | the entry signal does lead to a rebound |
# | Rules head-to-head | mean-reversion rule gross Sharpe **3.9**, max DD -1.0%; momentum rule on SI **-1.6%**, Sharpe -0.7 | the wrong rule loses money |
# | Regime filter (9b) | when the momentum filter is ON, next-bar SI return is -15% ann.; when OFF, +53% | buying strength in SI is penalised |
#
# **Assessment:** several independent lenses agree, and the opposite rule clearly fails. The 16-bar window and
# the 4-bar minimum hold also fit the measured ~4-bar half-life. **Main weakness: sample size.** There are
# 248 decision bars, 7 round trips and 10 entry events, and most bucket CIs overlap zero. The ADF result is
# also helped by the August-September plateau after the July rally. So the evidence is consistent and
# directionally strong, but not statistically conclusive on its own. The 60 bp break-even cost per round trip
# leaves a comfortable cushion over SGX ETF costs, which matters more than the exact Sharpe.
#
# ### 10.3 SPY: momentum is supported as a trend/regime filter, not by return autocorrelation
# **What the data does *not* show.** On 2h bars SPY is statistically a random walk in this window: ACF sum
# lags 1-4 = -0.02 (Ljung-Box p = 0.92), VR(q) is about 1.00 for q = 2-8 with every p-value above 0.6, and
# Hurst is 0.48. The lead-lag map is flat to slightly negative. In signal buckets, the **lowest** 8-bar
# momentum quintile had the best next-day return (+0.23%), and z < -1.25 also bounced (+0.33%, 67% hit). At
# the 1-day horizon, SPY in 2026 behaved more like a buy-the-dip market than a continuation market. The
# context cell (8b) shows the same was true for 2023-25 (2h VR(4) = 1.00, Hurst 0.49). **The SPY momentum
# choice cannot be justified by short-horizon autocorrelation, and the write-up should not claim it is.**
#
# **What the data *does* show (the defensible case):**
# 1. **SPY trends in its level; SI does not.** The log price is non-stationary (ADF p = 0.75, AR(1) half-life of
#    the level about 255 bars, versus 15 for SI). Deviations from the 16-bar mean decay more slowly (12.7h versus
#    8.4h for SI), and up-regimes are long: 17 EMA-up regimes with a **median of 18 bars (about 4.5 trading days)**.
#    The April-May rally shows this: once the trend turned, it ran for about 8 weeks with only short pullbacks
#    (figure 9).
# 2. **The momentum filter separates good regimes from bad ones (figure 9b).** When the rule is long, next-bar
#    returns average **19% annualised at 12% vol (Sharpe 1.58)**. When it is flat, **12% at 14% vol (Sharpe 0.89)**.
#    Only **11%** of "ON" bars fall inside a >2% drawdown, versus **42%** of "OFF" bars, so the rule tends to be
#    out of the market during drawdowns. The same filter on SI picks the *worst* regimes (-15% annualised),
#    which is the clearest single picture of why the two assets need opposite rules.
# 3. **Risk profile:** momentum rule max DD **-5.8% vs -9.3%** buy & hold, ann. vol 7.6% vs 13%, 40% time in market.
#
# **The honest trade-off:** in this window the momentum rule's gross Sharpe (1.00) is **below buy & hold (1.24)**
# and below the mean-reversion rule applied to SPY (1.85, max DD -3.0%). Momentum on SPY is lagging: the EMA
# 8/16 cross confirmed the April rebound only after the first ~4% of it, and in the Feb-March chop it took
# several small losses (31 round trips, 53% bar hit rate). The break-even cost is about 18 bp per round trip,
# which is thin but above SPY's realistic cost of ~1-3 bp. The improvement in the ON-versus-OFF means is also
# not statistically significant over 9 months.
#
# ### 10.4 Bottom line
# | | ES3.SI -> mean reversion | SPY -> momentum |
# |---|---|---|
# | Statistical evidence for the premise | **Consistent across all lenses** (VR<1 significant at 4-6h, H=0.41, negative lead-lag, stationary price) | **Absent at 2h-1d** (random walk); present only as regime/trend persistence in the price level |
# | Opposite rule on the same asset | Loses money (-1.6%) | Outperforms in this window (Sharpe 1.85 vs 1.00) |
# | Sample | Small (3 months, 7 trades) | Adequate (9 months, 31 trades) |
# | Strength of justification | **Moderate-to-strong**, limited by sample | **Weak-to-moderate**: framed as drawdown control and trend participation, not as an edge |
#
# **Suggested framing for the report:** SI is a low-vol, range-bound ETF whose short-term deviations close
# within about a day, so fading the 16-bar z-score fits its dynamics. SPY has a strong positive drift and long
# directional regimes. The momentum rule is a way to *participate in those regimes while sidestepping
# drawdowns*: it gives up some return for roughly 40% lower drawdown and vol. Acknowledge that 2026 SPY shows
# short-horizon dip-buying, which is a known risk to this choice. A natural robustness check is a longer
# daily-bar trend filter, since SPY's trend persistence lives at multi-day horizons rather than 2h.
#
# *All P&L figures in this EDA are gross and use simplified rule replicas. Official after-cost results
# come from `run_strategy_backtest.py`.*
