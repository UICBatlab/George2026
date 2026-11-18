#!/usr/bin/env python3
"""
plot_closest_position_cm.py
---------------------------
For each bat, finds the minimum x position (closest approach to the speaker)
reached during stimulus windows, converted from pixels to centimetres.

Pixel -> cm conversion
    The video frame (1600 x 1000 px) is larger than the arena, and the arena
    is square while the frame is not. So you can't just divide by 1600.
    Instead, measure where the arena's LEFT and RIGHT inner walls sit in
    pixels (open one frame in ImageJ / any image viewer and hover the cursor
    over each wall), and use:

        cm = (x_px - ARENA_LEFT_PX) * ARENA_WIDTH_CM / (ARENA_RIGHT_PX - ARENA_LEFT_PX)

    Result: 0 cm = speaker-side wall, ARENA_WIDTH_CM = far wall.
    Use ONE scale factor for everything (data, threshold, axis limits, shading).

Output: <BASE_DIR>/plots/closest_position_cm_boxplot.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats

# ── CONFIG ────────────────────────────────────────────────────────────────────

BASE_DIR = Path("/Users/zariageorge/Desktop/PhD/LOR/CSV_Tracking/Batch1.Man1")

FRAME_WIDTH  = 1600
FRAME_HEIGHT = 1000

# ── ARENA CALIBRATION (EDIT THESE) ───────────────────────────────────────────
# The full 1600 x 1000 px frame covers the behavioral box: 138.747 x 91.757 cm.
# x scale: 1600 / 138.747 = 11.53 px/cm
# y scale: 1000 /  91.757 = 10.90 px/cm   (not the same → use x scale for x only)
ARENA_WIDTH_CM  = 138.747
ARENA_HEIGHT_CM = 91.757

ARENA_LEFT_PX  = 0.0
ARENA_RIGHT_PX = float(FRAME_WIDTH)

PX_PER_CM = (ARENA_RIGHT_PX - ARENA_LEFT_PX) / ARENA_WIDTH_CM


def px_to_cm(x_px):
    """Convert pixel x-coordinate(s) to cm from the speaker-side wall."""
    return (np.asarray(x_px, dtype=float) - ARENA_LEFT_PX) / PX_PER_CM


# Speaker is on the LEFT side of the arena (low x values)
SPEAKER_X = 362.14
SPEAKER_W = 144.29

# Near-speaker zone threshold, originally defined in pixels
ZONE_THRESHOLD_PX = 500
ZONE_THRESHOLD_CM = float(px_to_cm(ZONE_THRESHOLD_PX))

# Stimulus windows (seconds)
STIMULUS_WINDOWS = [
    (120, 150),   # 2:00–2:30
    (270, 300),   # 4:30–5:00
    (420, 450),   # 7:00–7:30
    (570, 600),   # 9:30–10:00
]

DPI = 180

# ── CONDITIONS ────────────────────────────────────────────────────────────────

condition_files = {
    "Distress · Nonsocial": [
        BASE_DIR / "tracked.BIB.Diss.D1.NS.csv",
        BASE_DIR / "tracked.AF8.Diss.D1.NS.csv",
        BASE_DIR / "tracked.AFE.D2.Diss.NS.csv",
        BASE_DIR / "tracked.B31.D1.Diss.NS.F.csv",
        BASE_DIR / "tracked.B0C.D2.Diss.NS.csv",
        BASE_DIR / "tracked.B06.NS.Diss.D2.F.csv",
    ],
    "Distress · Social": [
        BASE_DIR / "tracked.B07.Diss.D1.S.csv",
        BASE_DIR / "tracked.B0F.Diss.D1.S.csv",
        BASE_DIR / "tracked.AFD.D2.Diss.S.csv",
        BASE_DIR / "tracked.ADB.D1.Diss.Sl.F.csv",
        BASE_DIR / "tracked.B26.D2.Diss.S.F.csv",
        BASE_DIR / "tracked.AE7.Diss.D2.S.csv",
    ],
    "WN · Nonsocial": [
        BASE_DIR / "tracked.AFD.WN.D1.NS.csv",
        BASE_DIR / "tracked.AE7.WN.D1.NS.csv",
        BASE_DIR / "tracked.B0F.D2.WN.NS.csv",
        BASE_DIR / "tracked.B07.D2.WN.NS.csv",
        BASE_DIR / "tracked.B26.D1.WN.NS.F.csv",
        BASE_DIR / "tracked.ADB.D2.WN.NS.F.csv",
    ],
    "WN · Social": [
        BASE_DIR / "tracked.AFE.WN.D1.S.csv",
        BASE_DIR / "tracked.B0C.WN.D1.S.csv",
        BASE_DIR / "tracked.AF8.D2.WN.S.csv",
        BASE_DIR / "tracked.BIB.D2.WN.S.csv",
        BASE_DIR / "tracked.B06.D1.WS.S.F.csv",
        BASE_DIR / "tracked.B31.D2.WN.S.csv",
    ],
}

# Bottom → top on horizontal plot
CONDITION_ORDER = [
    "WN · Social",
    "WN · Nonsocial",
    "Distress · Social",
    "Distress · Nonsocial",
]

CONDITION_COLORS = {
    "Distress · Nonsocial": "#221449",
    "Distress · Social":    "#862781",
    "WN · Nonsocial":       "#e34e64",
    "WN · Social":          "#feb77f",
}

# ── CORE METRIC ───────────────────────────────────────────────────────────────

def closest_position_cm(csv_path: Path) -> float | None:
    """
    Returns the minimum x position (in cm from the speaker-side wall) the bat
    reached during any stimulus window. Lower = closer to speaker.
    """
    if not csv_path.exists():
        print(f"  [MISSING]  {csv_path.name}")
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [ERROR]    {csv_path.name}: {e}")
        return None

    det = (df[df["detected"] == True]
             .dropna(subset=["x_px", "time_sec"])
             .sort_values("time_sec")
             .reset_index(drop=True))

    if det.empty:
        print(f"  [WARNING]  No detected frames: {csv_path.name}")
        return None

    stim_mask = pd.Series(False, index=det.index)
    for t0, t1 in STIMULUS_WINDOWS:
        stim_mask |= (det["time_sec"] >= t0) & (det["time_sec"] <= t1)
    det = det[stim_mask]

    if det.empty:
        print(f"  [WARNING]  No stimulus-window frames: {csv_path.name}")
        return None

    min_x_px = float(det["x_px"].min())
    min_x_cm = float(px_to_cm(min_x_px))
    print(f"  {csv_path.name:<45}  min x = {min_x_px:,.1f} px = {min_x_cm:.2f} cm")
    return min_x_cm


# ── COLLECT ───────────────────────────────────────────────────────────────────

def collect_data() -> dict:
    data = {}
    for cond, files in condition_files.items():
        print(f"\n[{cond}]")
        vals = []
        for fp in files:
            v = closest_position_cm(fp)
            if v is not None:
                vals.append(v)
        data[cond] = vals
        print(f"  → n={len(vals)}  values (cm): {[f'{v:.2f}' for v in vals]}")
    return data


# ── PLOT ──────────────────────────────────────────────────────────────────────

def plot(data: dict):
    fig, ax = plt.subplots(figsize=(11, 5), facecolor="white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#e4e4e4", linewidth=0.6, zorder=0)
    ax.yaxis.grid(False)

    n_conds   = len(CONDITION_ORDER)
    positions = list(range(n_conds))

    thr = ZONE_THRESHOLD_CM

    # Zone threshold + shading, all in cm
    ax.axvline(x=thr, color="#e63946", linewidth=1.2, linestyle="--",
               alpha=0.6, zorder=3)
    ax.axvspan(0, thr, color="#e63946", alpha=0.05, zorder=1)
    ax.axvspan(thr, ARENA_WIDTH_CM, color="#457b9d", alpha=0.04, zorder=1)

    for i, cond in enumerate(CONDITION_ORDER):
        vals  = data.get(cond, [])
        color = CONDITION_COLORS[cond]

        if len(vals) >= 2:
            ax.boxplot(
                vals,
                positions=[i],
                vert=False,
                widths=0.42,
                patch_artist=True,
                manage_ticks=False,
                zorder=5,
                boxprops=dict(facecolor=color, alpha=0.25, linewidth=1.4,
                              edgecolor=color),
                medianprops=dict(color=color, linewidth=2.4),
                whiskerprops=dict(color=color, linewidth=1.4),
                capprops=dict(color=color, linewidth=1.6),
                showfliers=False,  # every point is already drawn as a dot
            )

        # Individual dots with jitter
        rng    = np.random.default_rng(seed=42 + i)
        jitter = rng.uniform(-0.14, 0.14, size=len(vals))
        ax.scatter(vals, np.array([i] * len(vals)) + jitter,
                   color=color, s=55, zorder=7, alpha=0.9,
                   edgecolors="white", linewidths=0.7)

    # x axis = full arena width in cm
    ax.set_xlim(0, ARENA_WIDTH_CM)

    ax.set_ylim(-0.6, n_conds - 0.4)
    ax.set_yticks(positions)
    ax.set_yticklabels(CONDITION_ORDER, fontsize=10.5, color="#222222")
    ax.set_xlabel(
        "Closest x position reached during stimulus windows (cm from speaker wall)",
        fontsize=9, color="#444444", labelpad=8
    )
    ax.set_title(
        "Closest Position to Speaker During Stimulus Windows",
        fontsize=13, fontweight="bold", color="#111111", pad=10
    )

    ax.text(0.01, 0.02, "← Closer to speaker", transform=ax.transAxes,
            fontsize=8, color="#e63946", alpha=0.8, va="bottom")
    ax.text(0.99, 0.02, "Further away →", transform=ax.transAxes,
            fontsize=8, color="#457b9d", alpha=0.8, va="bottom", ha="right")

    legend_handles = [
        patches.Patch(facecolor="#e63946", alpha=0.2, edgecolor="#e63946",
                      label=f"Near speaker zone (≤ {thr:.1f} cm)"),
        patches.Patch(facecolor="#457b9d", alpha=0.2, edgecolor="#457b9d",
                      label=f"Away from speaker (> {thr:.1f} cm)"),
        plt.Line2D([0], [0], color="#e63946", linewidth=1.2, linestyle="--",
                   label=f"Zone threshold ({thr:.1f} cm)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8.5, loc="upper right",
              framealpha=0.9, edgecolor="#dddddd")

    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(0.8)
    ax.tick_params(axis="x", colors="#555555", labelsize=8)
    ax.tick_params(axis="y", length=0)

    plt.tight_layout()
    return fig


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Calibration: {PX_PER_CM:.2f} px/cm  "
          f"(arena {ARENA_LEFT_PX:.0f}–{ARENA_RIGHT_PX:.0f} px = {ARENA_WIDTH_CM} cm)")
    print(f"Zone threshold: {ZONE_THRESHOLD_PX} px = {ZONE_THRESHOLD_CM:.2f} cm")

    out_dir = BASE_DIR / "plots"
    out_dir.mkdir(exist_ok=True)

    data = collect_data()

    print("\n── Summary (cm) ──────────────────────────────────────")
    for cond in CONDITION_ORDER:
        vals = data.get(cond, [])
        if vals:
            print(f"  {cond}: mean={np.mean(vals):.2f} cm  "
                  f"sem={stats.sem(vals):.2f}  n={len(vals)}")
        else:
            print(f"  {cond}: no data")
    print("──────────────────────────────────────────────────────\n")

    fig = plot(data)
    out_path = out_dir / "closest_position_cm_boxplot.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out_path}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()