

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:55:00 2026

@author: zariageorge
"""

"""
USV call analysis: whether number of calls predicts approach.

Usage:  python call_approach.py Wav_File_Count_1.xlsx [output_folder]

Input sheet columns: File Names, Call_Count, Approach, Approach_Amount,
Non_Social, Social. File names look like  ANIMAL.DAY.STIM_PLAYBACK,
e.g. "AFD.D2.Diss_S" -> animal AFD, day 2, Diss stimulus, Social playback.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ---------------------------------------------------------------- settings
XLSX = sys.argv[1] if len(sys.argv) > 1 else "/Users/zariageorge/Downloads/Wav_File_Count_1.xlsx"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
OUT.mkdir(parents=True, exist_ok=True)

COND_ORDER = ["WN-NS", "WN-S", "Diss-NS", "Diss-S"]
# Fixed categorical order (colorblind-validated): blue, orange, aqua, yellow
COND_COLORS = {"WN-NS": "#2a78d6", "WN-S": "#eb6834",
               "Diss-NS": "#1baf7a", "Diss-S": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


# ---------------------------------------------------------------- load data
def load(path):
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={df.columns[0]: "File"})
    df = df.dropna(subset=["File"]).copy()
    df["File"] = df["File"].str.strip()

    parts = df["File"].str.extract(r"^([^.]+)\.(D\d)\.(WN|Diss)[._](S|NS)$")
    df["Animal"], df["Day"], df["Stim"], df["Playback"] = (
        parts[0], parts[1], parts[2], parts[3])
    if df["Stim"].isna().any():
        bad = df.loc[df["Stim"].isna(), "File"].tolist()
        raise ValueError(f"Could not parse condition from: {bad}")
    df["Condition"] = df["Stim"] + "-" + df["Playback"]

    for c in ["Call_Count", "Non_Social", "Social", "Approach_Amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")   # 'Don't_Have_Missing' -> NaN
    df["Approached"] = (df["Approach"].str.strip().str.lower() == "y").astype(int)
    return df


df = load(XLSX)
calls = df.dropna(subset=["Call_Count"]).copy()          # sessions with a recording
typed = df.dropna(subset=["Social", "Non_Social"]).copy()  # sessions with call types

lines = []
def report(s=""):
    print(s); lines.append(s)

report(f"Sessions in sheet: {len(df)}  |  with call count: {len(calls)}  |  "
       f"with call types: {len(typed)}  |  animals: {df['Animal'].nunique()}")
mismatch = typed[(typed["Social"] + typed["Non_Social"]) != typed["Call_Count"]]
if len(mismatch):
    report("NOTE: Social + Non_Social != Call_Count for: "
           + ", ".join(f"{r.File} ({r.Social:.0f}+{r.Non_Social:.0f} vs {r.Call_Count:.0f})"
                       for r in mismatch.itertuples()))


def jitter(n, width=0.10, seed=0):
    return np.random.default_rng(seed).uniform(-width, width, n)


def ptxt(p):
    return "p < .001" if p < .001 else f"p = {p:.3f}".replace("0.", ".")


# ================================================================ PART 2
# Does the number of calls predict approach?
report("\n=== PART 2: Does number of calls predict approach? ===")
calls["logCalls"] = np.log10(calls["Call_Count"] + 1)
report(calls.groupby("Approached")["Call_Count"]
       .agg(["count", "mean", "median", "min", "max"]).round(1)
       .rename(index={0: "No approach", 1: "Approach"}).to_string())

# (a) Mann-Whitney: calls in approachers vs non-approachers
yes = calls.loc[calls["Approached"] == 1, "Call_Count"]
no = calls.loc[calls["Approached"] == 0, "Call_Count"]
U, p_mw = stats.mannwhitneyu(yes, no, alternative="two-sided")
r_rb = 1 - 2 * U / (len(yes) * len(no))  # rank-biserial (sign: + means approach > no)
r_rb = -r_rb
report(f"\n(a) Mann-Whitney U = {U:.1f}, {ptxt(p_mw)}, rank-biserial r = {r_rb:.2f} "
       f"(n approach = {len(yes)}, n no approach = {len(no)})")

# (b) Logistic regression: P(approach) ~ log10(calls + 1)
logit = smf.logit("Approached ~ logCalls", data=calls).fit(disp=0)
b, se_b = logit.params["logCalls"], logit.bse["logCalls"]
ci = np.exp(logit.conf_int().loc["logCalls"])
report(f"\n(b) Logistic regression, approach ~ log10(calls+1):")
report(f"    b = {b:.2f} (SE {se_b:.2f}), OR per 10x more calls = {np.exp(b):.2f} "
       f"[95% CI {ci[0]:.2f}, {ci[1]:.2f}], Wald {ptxt(logit.pvalues['logCalls'])}, "
       f"LR chi2(1) = {logit.llr:.2f}, LR {ptxt(logit.llr_pvalue)}, "
       f"McFadden R2 = {logit.prsquared:.2f}")

# (c) Same, controlling for condition (social vs non-social playback)
logit2 = smf.logit("Approached ~ logCalls + C(Playback)", data=calls).fit(disp=0)
report(f"\n(c) Logistic regression, approach ~ log10(calls+1) + playback (S vs NS):")
for term in logit2.params.index[1:]:
    report(f"    {term}: b = {logit2.params[term]:.2f}, OR = {np.exp(logit2.params[term]):.2f}, "
           f"{ptxt(logit2.pvalues[term])}")

# (d) GEE: same model, accounting for each animal contributing 2 sessions
gee = smf.gee("Approached ~ logCalls", groups="Animal", data=calls,
              family=sm.families.Binomial(),
              cov_struct=sm.cov_struct.Exchangeable()).fit()
report(f"\n(d) GEE (clustered by animal), approach ~ log10(calls+1): "
       f"b = {gee.params['logCalls']:.2f}, OR = {np.exp(gee.params['logCalls']):.2f}, "
       f"robust {ptxt(gee.pvalues['logCalls'])}")

# (e) Spearman: calls vs number of approaches
rho, p_sp = stats.spearmanr(calls["Call_Count"], calls["Approach_Amount"])
report(f"\n(e) Spearman, calls vs approach amount: rho = {rho:.2f}, {ptxt(p_sp)}, N = {len(calls)}")

# Optional: ROC AUC (how well call count separates approachers)
auc = U / (len(yes) * len(no))
report(f"    AUC (calls discriminating approach) = {auc:.2f}  (0.5 = chance)")


# --- Figure 2: (A) calls by approach outcome, (B) fitted logistic curve
fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4.2),
                               gridspec_kw={"width_ratios": [1, 1.4]})

# Panel A
for i, (lab, v) in enumerate([("No approach", no), ("Approach", yes)]):
    sub = calls.loc[calls["Approached"] == i]
    axA.boxplot(v, positions=[i], widths=0.45, showfliers=False,
                medianprops=dict(color=INK, lw=2),
                boxprops=dict(color=INK2), whiskerprops=dict(color=INK2),
                capprops=dict(color=INK2), zorder=2)
    axA.scatter(i + jitter(len(sub), seed=10 + i), sub["Call_Count"],
                c=[COND_COLORS[c] for c in sub["Condition"]], s=36,
                edgecolor="white", linewidth=1, zorder=4)
axA.set_xticks([0, 1], [f"No approach\n(n = {len(no)})", f"Approach\n(n = {len(yes)})"])
axA.set_yscale("symlog", linthresh=10)
axA.set_ylim(bottom=0)
axA.set_ylabel("Total calls per session (symlog scale)")
axA.set_title(f"A  Calls by approach outcome\nMann-Whitney U = {U:.0f}, {ptxt(p_mw)}, "
              f"r = {r_rb:.2f}", loc="left", fontsize=10, color=INK)

# Panel B
xg = np.linspace(0, calls["logCalls"].max() * 1.02, 200)
X = sm.add_constant(xg)
eta = X @ logit.params.values
se_eta = np.sqrt(np.einsum("ij,jk,ik->i", X, logit.cov_params().values, X))
lo, hi = [1 / (1 + np.exp(-(eta + z * se_eta))) for z in (-1.96, 1.96)]
axB.fill_between(xg, lo, hi, color=INK2, alpha=0.12, lw=0, zorder=1)
axB.plot(xg, 1 / (1 + np.exp(-eta)), color=INK, lw=2, zorder=2)
yj = calls["Approached"] + jitter(len(calls), 0.03, seed=3)
for c in COND_ORDER:
    s = calls["Condition"] == c
    axB.scatter(calls.loc[s, "logCalls"], yj[s], color=COND_COLORS[c], s=36,
                edgecolor="white", linewidth=1, zorder=3, label=c)
ticks = [0, 10, 100, 1000]
axB.set_xticks(np.log10(np.array(ticks) + 1), [str(t) for t in ticks])
axB.set_xlabel("Total calls per session (log scale)")
axB.set_ylabel("P(approach)")
axB.set_ylim(-0.08, 1.08)
axB.grid(axis="x", color=GRID, lw=0.8)
axB.set_title(f"B  Logistic regression: approach ~ log(calls)\n"
              f"OR per 10× calls = {np.exp(b):.2f}, LR {ptxt(logit.llr_pvalue)}, "
              f"AUC = {auc:.2f}", loc="left", fontsize=10, color=INK)
axB.legend(title="Condition", frameon=True, facecolor="white", edgecolor=GRID,
           fontsize=8, title_fontsize=8, loc="upper left")
fig.text(0.06, -0.04, "Dots = sessions, colored by condition. Shaded band = 95% CI "
         "of fitted probability. Sessions with missing recordings excluded.",
         fontsize=8, color=INK2)
fig.tight_layout()
fig.savefig(OUT / "calls_predict_approach.png")
plt.show()


# --- Figure 3: mean calls, approach vs no approach (t-test + Wilcoxon rank-sum)
t, p_t = stats.ttest_ind(yes, no, equal_var=False)            # Welch's t-test
t_log, p_tlog = stats.ttest_ind(np.log10(yes + 1), np.log10(no + 1), equal_var=False)
report(f"\n(f) Welch t-test on raw calls: t = {t:.2f}, {ptxt(p_t)}")
report(f"    Welch t-test on log10(calls+1): t = {t_log:.2f}, {ptxt(p_tlog)}")
report(f"    Wilcoxon rank-sum (= Mann-Whitney): U = {U:.0f}, {ptxt(p_mw)}")

fig, ax = plt.subplots(figsize=(5, 4.5))
for i, (lab, v) in enumerate([("No approach", no), ("Approach", yes)]):
    sub = calls.loc[calls["Approached"] == i]
    m, se = v.mean(), stats.sem(v)
    ax.bar(i, m, width=0.55, color="#d9d8d3", edgecolor=INK2, lw=1, zorder=2)
    ax.errorbar(i, m, yerr=se, color=INK, capsize=5, lw=1.3, zorder=3)
    ax.scatter(i + jitter(len(sub), seed=20 + i), sub["Call_Count"],
               c=[COND_COLORS[c] for c in sub["Condition"]], s=36,
               edgecolor="white", linewidth=1, zorder=4)
ax.set_xticks([0, 1], [f"No approach\n(n = {len(no)})", f"Approach\n(n = {len(yes)})"])
ax.set_xlim(-0.6, 1.6)
ax.set_ylim(bottom=0)
ax.set_ylabel("Calls per session (mean ± SEM)")
ax.set_title(f"Average calls by approach outcome\n"
             f"Welch t-test: t = {t:.2f}, {ptxt(p_t)}\n"
             f"Wilcoxon rank-sum: U = {U:.0f}, {ptxt(p_mw)}",
             loc="left", fontsize=10, color=INK)
handles = [plt.Line2D([], [], marker="o", ls="", color=COND_COLORS[c], label=c)
           for c in COND_ORDER]
ax.legend(handles=handles, title="Condition", frameon=False, fontsize=8,
          title_fontsize=8, loc="upper left")
fig.text(0.02, -0.03, "Bars = mean, whiskers = SEM, dots = sessions.",
         fontsize=8, color=INK2)
fig.tight_layout()
fig.savefig(OUT / "mean_calls_by_approach.png")
plt.show()

(OUT / "stats_results.txt").write_text("\n".join(lines))
print(f"\nSaved figures and stats_results.txt to {OUT.resolve()}")