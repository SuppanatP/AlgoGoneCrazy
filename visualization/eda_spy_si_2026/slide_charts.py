# %% [markdown]
# # Slide charts - SPY momentum & STI (ES3.SI) mean reversion
# Re-uses every calculation in `eda_spy_si_2026.py` (runs it first), then draws six single-asset,
# presentation-sized charts into `./slide_charts/`. Run as cells in VS Code or `python slide_charts.py`.

# %% Run the EDA to get data, features and tables
from pathlib import Path
import re

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()

_src = (HERE / "eda_spy_si_2026.py").read_text(encoding="utf-8")
for _cell in re.split(r"^# %%.*$", _src, flags=re.M):
    exec(compile(_cell, "eda_spy_si_2026.py", "exec"), globals())

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

SLIDE_OUT = HERE / "slide_charts"
SLIDE_OUT.mkdir(exist_ok=True)
BG, DARK, SOFT, RULE = "#fbfbf8", "#14213d", "#4a5568", "#dfe2e6"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9aa0a6"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": RULE, "axes.labelcolor": SOFT, "axes.titlecolor": DARK,
    "axes.titlesize": 17, "axes.titleweight": "bold", "axes.labelsize": 15,
    "xtick.color": SOFT, "ytick.color": SOFT, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "grid.color": RULE, "legend.fontsize": 14, "font.size": 14, "lines.linewidth": 2.0,
})


def save_slide(fig, name):
    fig.tight_layout()
    fig.savefig(SLIDE_OUT / name, dpi=200, bbox_inches="tight")
    plt.show() if INTERACTIVE else plt.close(fig)
    print("saved -> slide_charts/" + name)


def trading_time_ticks(ax, d, tz, every=1):
    ld = d["t_close"].dt.tz_convert(tz)
    wk = ld.dt.strftime("%G-%V")
    tk = np.where(wk.ne(wk.shift()))[0][::every]
    ax.set_xticks(tk, [ld.iloc[i].strftime("%d %b") for i in tk])


# %% SPY slide 1 - trend regimes
d = feat[SPY]
pos = mom_positions(d)
fig, ax = plt.subplots(figsize=(11, 6.2))
y0, y1 = d["adjusted_close"].min() * 0.985, d["adjusted_close"].max() * 1.012
ax.fill_between(d["t_close"], y0, y1, where=pos == 1, color=BLUE, alpha=0.10, lw=0, label="Strategy long")
ax.plot(d["t_close"], d["adjusted_close"], color=GREY, lw=1.2, label="SPY (2h close)")
ax.plot(d["t_close"], d["ema_fast"], color=BLUE, lw=2.0, label="EMA 8")
ax.plot(d["t_close"], d["ema_slow"], color=DARK, lw=2.0, label="EMA 16")
ax.set_ylim(y0, y1)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.set_ylabel("USD")
ax.legend(loc="upper left", ncol=4, frameon=False)
for txt, x, y in [("Jan-Mar: drift lower,\n-9% drawdown", "2026-01-08", 0.10),
                  ("Apr-May: V-shaped\ntrend, +16%", "2026-04-28", 0.12),
                  ("Jun-Sep: grind higher", "2026-06-25", 0.42)]:
    ax.text(pd.Timestamp(x, tz="UTC"), y0 + (y1 - y0) * y, txt, fontsize=13.5, color=DARK)
save_slide(fig, "spy_1_trend_regimes.png")

# %% SPY slide 2 - regime filter
t = t_regime[t_regime["asset"] == ASSETS[SPY]["label"]].reset_index(drop=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axes[0]
x = np.arange(2)
for off, col, a, lab in [(-0.2, "ann_mean_next_bar", 1, "Annualised return"), (0.2, "ann_vol_next_bar", 0.35, "Annualised volatility")]:
    ax.bar(x + off, t[col], 0.38, color=BLUE, alpha=a, label=lab)
    for xi, v in zip(x, t[col]):
        ax.text(xi + off, v + 0.004, f"{v:.0%}", ha="center", va="bottom", fontsize=15, color=DARK, fontweight="bold")
ax.set_xticks(x, [f"Long (momentum ON)\nSharpe {t['sharpe'][0]:.2f}", f"Flat (momentum OFF)\nSharpe {t['sharpe'][1]:.2f}"])
ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.set_ylim(0, 0.24); ax.legend(loc="upper right", frameon=False)
ax.set_title("SPY next-2h return by regime", loc="left")
ax2 = axes[1]
vals = t["share_of_bars_in_>2%_drawdown"].to_numpy()
ax2.bar(x, vals, 0.55, color=[BLUE, GREY])
for xi, v in zip(x, vals):
    ax2.text(xi, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=15, color=DARK, fontweight="bold")
ax2.set_xticks(x, ["Long", "Flat"])
ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0)); ax2.set_ylim(0, 0.52)
ax2.set_title("Share of bars spent in\na >2% drawdown", loc="left")
save_slide(fig, "spy_2_regime_filter.png")

# %% SPY slide 3 - equity and drawdown vs buy & hold
r_next = d["ret"].shift(-1).fillna(0).to_numpy()
eq_bh, eq_m = np.exp(np.cumsum(r_next)), np.exp(np.cumsum(pos * r_next))
fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
axes[0].plot(d["t_close"], eq_bh, color=GREY, label="Buy & hold")
axes[0].plot(d["t_close"], eq_m, color=BLUE, label="Momentum rule (gross)")
axes[0].set_ylabel("Growth of $1"); axes[0].legend(loc="upper left", frameon=False)
for eq, c in [(eq_bh, GREY), (eq_m, BLUE)]:
    dd = eq / np.maximum.accumulate(eq) - 1
    axes[1].fill_between(d["t_close"], dd, 0, color=c, alpha=0.35 if c == BLUE else 0.25, lw=0)
    axes[1].plot(d["t_close"], dd, color=c, lw=1.4)
axes[1].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0)); axes[1].set_ylabel("Drawdown")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
save_slide(fig, "spy_3_equity_drawdown.png")

# %% STI slide 1 - z-score bands
d = feat[SI]
xx = np.arange(len(d))
fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
ax = axes[0]
ax.fill_between(xx, d["ma"] - 1.25 * d["sd"], d["ma"] + 1.25 * d["sd"], color=GREY, alpha=0.22, lw=0, label="±1.25 sd band")
ax.plot(xx, d["adjusted_close"], color=ORANGE, label="ES3.SI (2h close)")
ax.plot(xx, d["ma"], color=DARK, lw=1.4, ls="--", label="16-bar mean")
cross = ((d["z"] < SI_ENTRY_Z) & (d["z"].shift() >= SI_ENTRY_Z)).to_numpy()
ax.scatter(xx[cross], d.loc[cross, "adjusted_close"], s=70, color=BLUE, zorder=3, edgecolors=BG, lw=1.5, label="Buy signal (z < -1.25)")
ax.set_ylabel("SGD"); ax.legend(loc="lower right", ncol=2, frameon=False)
axz = axes[1]
axz.plot(xx, d["z"], color=ORANGE, lw=1.3)
axz.axhline(SI_ENTRY_Z, color=BLUE, ls="--", lw=1.4); axz.axhline(SI_EXIT_Z, color=DARK, ls=":", lw=1.4)
axz.text(2, SI_ENTRY_Z - 0.15, "entry -1.25", va="top", fontsize=13, color=BLUE)
axz.text(2, SI_EXIT_Z + 0.15, "exit +0.25", va="bottom", fontsize=13, color=DARK)
axz.set_ylabel("z-score")
trading_time_ticks(axz, d, ASSETS[SI]["tz"], every=2)
save_slide(fig, "sti_1_zscore_bands.png")

# %% STI slide 2 - variance ratio + Hurst (SPY as reference)
fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
for k, c in [(SPY, GREY), (SI, ORANGE)]:
    v = t_vr[(t_vr["asset"] == ASSETS[k]["label"]) & (t_vr["q_bars"] != "Hurst")]
    ax.plot(v["horizon_h"], v["VR"], marker="o", ms=7, color=c, lw=2.4, label=ASSETS[k]["label"].split(" (")[0])
    sig = v[v["p"] < 0.10]
    ax.scatter(sig["horizon_h"], sig["VR"], s=230, facecolors="none", edgecolors=c, lw=1.6)
ax.axhline(1, color=DARK, lw=1, ls="--")
ax.text(32, 1.005, "random walk = 1.0", ha="right", va="bottom", fontsize=13, color=DARK)
ax.set_xlabel("Holding horizon (hours)"); ax.set_ylabel("Variance ratio")
ax.set_title("Variance ratio (circled: p < 0.10)", loc="left"); ax.legend(loc="lower left", frameon=False)
ax2 = axes[1]
H = [t_vr[(t_vr["asset"] == ASSETS[k]["label"]) & (t_vr["q_bars"] == "Hurst")]["VR"].iloc[0] for k in (SPY, SI)]
ax2.bar([0, 1], H, 0.55, color=[GREY, ORANGE])
for i, h in enumerate(H):
    ax2.text(i, h + 0.008, f"{h:.2f}", ha="center", va="bottom", fontsize=16, fontweight="bold", color=DARK)
ax2.axhline(0.5, color=DARK, lw=1, ls="--")
ax2.text(1.42, 0.512, "0.5 = random walk", ha="right", va="bottom", fontsize=13, color=DARK)
ax2.set_xticks([0, 1], ["SPY", "ES3.SI"]); ax2.set_ylim(0, 0.62)
ax2.set_title("Hurst exponent", loc="left")
save_slide(fig, "sti_2_variance_ratio_hurst.png")

# %% STI slide 3 - event study + rule comparison
fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
p = d["logp"].to_numpy(); drift = d["ret"].mean()
ev = np.where(((d["z"] < SI_ENTRY_Z) & (d["z"].shift() >= SI_ENTRY_Z)).to_numpy())[0]
paths = np.array([p[i:i + 13] - p[i] - drift * np.arange(13) for i in ev if i + 12 < len(p)])
for pth in paths:
    ax.plot(np.arange(13), pth, color=ORANGE, alpha=0.18, lw=1)
mu = paths.mean(0)
ax.plot(np.arange(13), mu, color=ORANGE, lw=3, marker="o", ms=5, label=f"Average of {len(paths)} signals")
ax.axhline(0, color=DARK, lw=1)
ax.text(12, mu[-1] + 0.001, f"{mu[-1]:+.2%}", ha="right", va="bottom", fontsize=15, fontweight="bold", color=DARK)
ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
ax.set_xticks(range(0, 13, 2)); ax.set_xlabel("2h bars after buy signal"); ax.set_ylabel("Excess return vs drift")
ax.set_title("After an oversold signal", loc="left"); ax.legend(loc="upper left", frameon=False)
ax2 = axes[1]
r_next = d["ret"].shift(-1).fillna(0).to_numpy()
for lab, pp, c, lw in [("Buy & hold", np.ones(len(d)), GREY, 1.6), ("Momentum rule", mom_positions(d), BLUE, 2.2),
                       ("Mean-reversion rule", mr_positions(d), ORANGE, 2.6)]:
    ax2.plot(xx, np.exp(np.cumsum(pp * r_next)), color=c, lw=lw, label=lab)
ax2.set_ylabel("Growth of $1 (gross)"); ax2.set_ylim(0.97, 1.16); ax2.legend(loc="upper left", frameon=False)
ax2.set_title("Same asset, both rules", loc="left")
trading_time_ticks(ax2, d, ASSETS[SI]["tz"], every=3)
save_slide(fig, "sti_3_event_study_rules.png")
