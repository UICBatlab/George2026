import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ── Font: Arial, 8 pt everywhere ─────────────────────────────
FONT_SIZE = 8
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],  # Arial first
    'font.size': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE,
    'legend.title_fontsize': FONT_SIZE,
    'figure.titlesize': FONT_SIZE,
    'pdf.fonttype': 42,   # keep text editable in Illustrator
    'svg.fonttype': 'none',
})


# ─────────────────────────────────────────────────────────────
# FILE PARSING
# ─────────────────────────────────────────────────────────────

def read_boris(path):
    """Read a BORIS export; .tsv files are tab-separated, everything else comma."""
    return pd.read_csv(path, sep='\t' if str(path).lower().endswith('.tsv') else ',')


def calculate_behavior_durations(csv_path):
    """
    Calculate durations for each behavior in a BORIS CSV file.
    Returns a dictionary with behavior names as keys and list of durations as values.
    """
    try:
        df = read_boris(csv_path)
    except Exception as e:
        print(f"  ERROR reading file: {e}")
        return {}

    behaviors = df['Behavior'].dropna().unique()
    behavior_durations = {}

    for behavior in behaviors:
        behavior_df = df[df['Behavior'] == behavior].copy()
        behavior_df = behavior_df.sort_values('Time')

        durations = []
        i = 0
        while i < len(behavior_df):
            row = behavior_df.iloc[i]
            if row['Behavior type'] == 'START':
                j = i + 1
                while j < len(behavior_df):
                    next_row = behavior_df.iloc[j]
                    if next_row['Behavior type'] == 'STOP':
                        duration = next_row['Time'] - row['Time']
                        durations.append(duration)
                        i = j
                        break
                    j += 1
            i += 1

        if durations:
            behavior_durations[behavior] = durations

    # IMMOBILE = time in each 30 s window NOT accounted for by any scored behavior:
    #   immobile per window = (n_windows x 30 s - time covered by behaviors) / n_windows
    # Overlapping behaviors are only counted once. Max = WINDOW_SECONDS.
    try:
        intervals, length, _, _ = get_behavior_intervals(csv_path)
        covered, _ = find_gaps(intervals, length)
        # windows needed to hold the scored span (2 s tolerance for slight overrun)
        n_windows = max(1, int(np.ceil((length - 2) / WINDOW_SECONDS)))
        total = n_windows * WINDOW_SECONDS
        immobile = max(0.0, total - covered) / n_windows
        behavior_durations['Immobile'] = [immobile]
        if VERBOSE_SPAN:
            print(f"      -> {n_windows} window(s) x {WINDOW_SECONDS}s = {total:.0f}s, "
                  f"behaviors cover {covered:.1f}s, immobile = {immobile:.1f}s per window")
    except Exception as e:
        print(f"  Could not compute Immobile for {Path(csv_path).name}: {e}")

    return behavior_durations


def process_csv_files(csv_files, condition_name):
    """
    Process a list of CSV files for a given condition.
    Returns a DataFrame with columns: Condition, Behavior, Duration, File
    """
    all_data = []

    for csv_file in csv_files:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            print(f"  WARNING: File not found: {csv_file}")
            continue

        print(f"  Processing {csv_path.name}...")
        behavior_durations = calculate_behavior_durations(csv_path)

        for behavior, durations in behavior_durations.items():
            for duration in durations:
                all_data.append({
                    'Condition': condition_name,
                    'Behavior': behavior,
                    'Duration': duration,
                    'File': csv_path.name
                })

    return pd.DataFrame(all_data)


def process_csv_files_per_file(csv_files, condition_name):
    """
    Process CSV files and calculate MEAN duration per file (n=1 per file).
    This treats each file as an independent observation.
    """
    all_data = []

    for csv_file in csv_files:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            print(f"  WARNING: File not found: {csv_file}")
            continue

        print(f"  Processing {csv_path.name}...")
        behavior_durations = calculate_behavior_durations(csv_path)

        for behavior, durations in behavior_durations.items():
            mean_duration = np.mean(durations) if len(durations) > 0 else 0
            count = len(durations)

            all_data.append({
                'Condition': condition_name,
                'Behavior': behavior,
                'MeanDuration': mean_duration,
                'Count': count,
                'File': csv_path.name,
                'Subject': csv_path.stem
            })

    return pd.DataFrame(all_data)


# ─────────────────────────────────────────────────────────────
# SHARED CONSTANTS
# ─────────────────────────────────────────────────────────────

CONDITIONS = ['Distress_Nonsocial', 'Distress_Social', 'WN_Nonsocial', 'WN_Social']
# Panels are drawn in THIS order (left to right)
BEHAVIORS_TO_KEEP = ['Grooming', 'Pointing', 'Touching Box', 'Immobile']

CONDITION_COLORS = {
    'Distress_Nonsocial': '#1a0b2e',
    'Distress_Social':    '#9b4f96',
    'WN_Nonsocial':       '#ff6b9d',
    'WN_Social':          '#ffd23f',
}

# All plots and CSVs are saved straight to your Desktop.
OUTPUT_DIR = Path.home() / 'Desktop'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Length of one time window (s). Immobile is measured per window, so its max = this.
WINDOW_SECONDS = 30

# True  = only analyse the part of each video you scored (first START -> last STOP)
# False = use the full media duration from BORIS (counts unscored video as immobile)
USE_SCORED_SPAN = True
VERBOSE_SPAN = True   # print media length vs scored span for every file

# Unscored stretches shorter than this (seconds) are NOT added to Immobile time
# (this filters out tiny scoring slop between one behavior's STOP and the next START)
MIN_GAP_SECONDS = 1.0


def immobile_per_window(gaps, length):
    """
    Split the scored span into consecutive WINDOW_SECONDS windows and return
    the unscored seconds in each one (e.g. a 29 s span = 1 window, 61 s = 2).
    """
    n = max(1, int(round(length / WINDOW_SECONDS)))
    bins = [(i * WINDOW_SECONDS, min((i + 1) * WINDOW_SECONDS, length)) for i in range(n)]
    bins = [(b0, b1) for b0, b1 in bins if b1 > b0]
    return [sum(max(0, min(e, b1) - max(s, b0)) for s, e in gaps) for b0, b1 in bins]


# Extra spellings that should count as each plotted behavior (lowercase).
BEHAVIOR_ALIASES = {
    'Touching Box': ['touching box', 'touch box', 'touching the box'],
}


def normalize_behaviors(data):
    """
    Map behavior names onto BEHAVIORS_TO_KEEP, ignoring capitals and extra
    spaces (so 'immobile', 'Immobile ', 'Immobility' all become 'Immobile'),
    then keep only those behaviors. Warns about any that weren't found.
    """
    data = data.copy()
    data['Behavior'] = data['Behavior'].replace({'Flapping': 'Flying'})
    lookup = {}
    for b in BEHAVIORS_TO_KEEP:
        lookup[b.lower()] = b
        for alias in BEHAVIOR_ALIASES.get(b, []):
            lookup[alias.strip().lower()] = b
    raw = data['Behavior'].astype(str)
    data['Behavior'] = raw.str.strip().str.lower().map(lookup).fillna(raw)

    found = set(data['Behavior'])
    missing = [b for b in BEHAVIORS_TO_KEEP if b not in found]
    if missing:
        print(f"\n!!! WARNING: no data found for {missing}.")
        print(f"!!! Behavior names in your files are: {sorted(set(raw))}")
        print("!!! Add the right spelling to BEHAVIOR_ALIASES near the top of the script.")
    return data[data['Behavior'].isin(BEHAVIORS_TO_KEEP)]


# ─────────────────────────────────────────────────────────────
# TIME BUDGET / GAP ANALYSIS
# ─────────────────────────────────────────────────────────────

WINDOW_COLUMNS = ['Media duration (s)', 'Media duration', 'Total length',
                  'Observation duration', 'Observation duration (s)']


def get_window_length(df):
    """
    Length of the observation window. Uses BORIS's media-duration column
    when present; otherwise falls back to the last scored event time.
    Returns (seconds, source_label).
    """
    for col in WINDOW_COLUMNS:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(vals) > 0 and vals.max() > 0:
                return float(vals.max()), col
    return float(pd.to_numeric(df['Time'], errors='coerce').max()), 'last event (estimated)'


def get_behavior_intervals(csv_path):
    """
    Returns (intervals, window_length, window_source, unmatched)
      intervals : list of (behavior, start, stop) for ALL behaviors
      unmatched : list of (behavior, time, 'START'/'STOP') events with no partner
    """
    df = read_boris(csv_path)
    type_col = 'Behavior type' if 'Behavior type' in df.columns else 'Status'
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')

    intervals, unmatched = [], []
    for behavior in df['Behavior'].dropna().unique():
        bdf = df[df['Behavior'] == behavior].sort_values('Time')
        open_start = None
        for _, row in bdf.iterrows():
            etype = str(row[type_col]).upper()
            if etype == 'START':
                if open_start is not None:
                    unmatched.append((behavior, open_start, 'START'))
                open_start = row['Time']
            elif etype == 'STOP':
                if open_start is None:
                    unmatched.append((behavior, row['Time'], 'STOP'))
                else:
                    intervals.append((behavior, open_start, row['Time']))
                    open_start = None
        if open_start is not None:
            unmatched.append((behavior, open_start, 'START'))

    media_len, media_src = get_window_length(df)

    if USE_SCORED_SPAN and intervals:
        # Only analyse the stretch you actually scored: first START -> last STOP.
        # Times are shifted so the scored span starts at 0.
        t0 = min(s for _, s, _ in intervals)
        t1 = max(e for _, _, e in intervals)
        intervals = [(b, s - t0, e - t0) for b, s, e in intervals]
        unmatched = [(b, t - t0, k) for b, t, k in unmatched]
        window, source = t1 - t0, f'scored span {t0:.1f}-{t1:.1f}s'
        if VERBOSE_SPAN:
            print(f"    {Path(csv_path).name}: media = {media_len:.1f}s, "
                  f"scored span = {t0:.1f}s to {t1:.1f}s ({window:.1f}s, "
                  f"~{max(1, int(np.ceil((window - 2) / WINDOW_SECONDS)))} x {WINDOW_SECONDS}s windows)")
    else:
        window, source = media_len, media_src
    return intervals, window, source, unmatched


def find_gaps(intervals, window):
    """Merge all behavior intervals and return (covered_seconds, gaps list)."""
    spans = sorted((max(0, s), min(window, e)) for _, s, e in intervals if e > s)
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    covered = sum(e - s for s, e in merged)
    gaps, cursor = [], 0.0
    for s, e in merged:
        if s > cursor:
            gaps.append((cursor, s))
        cursor = max(cursor, e)
    if window > cursor:
        gaps.append((cursor, window))
    return covered, gaps


def build_time_budget(file_lists):
    """
    One row per file: seconds spent in each behavior, covered time,
    unaccounted time, overlap, and gap info. Also returns a long-form
    table of every gap and the raw intervals (for the timeline plot).
    """
    rows, gap_rows, timeline = [], [], []
    for condition, files in file_lists.items():
        for f in files:
            p = Path(f)
            if not p.exists():
                print(f"  WARNING: File not found: {f}")
                continue
            try:
                intervals, window, source, unmatched = get_behavior_intervals(p)
            except Exception as e:
                print(f"  ERROR reading {p.name}: {e}")
                continue

            covered, gaps = find_gaps(intervals, window)
            real_gaps = [(s, e) for s, e in gaps if e - s >= MIN_GAP_SECONDS]

            row = {'Condition': condition, 'File': p.name,
                   'Window_s': window, 'WindowSource': source}
            for b, s, e in intervals:
                row[b] = row.get(b, 0) + (e - s)
            behavior_total = sum(e - s for _, s, e in intervals)
            row.update({
                'Covered_s': covered,
                'Unaccounted_s': window - covered,
                'Unaccounted_pct': 100 * (window - covered) / window if window else np.nan,
                'Overlap_s': behavior_total - covered,
                'N_gaps': len(real_gaps),
                'Longest_gap_s': max((e - s for s, e in real_gaps), default=0),
                'Unmatched_events': len(unmatched),
            })
            rows.append(row)

            for s, e in real_gaps:
                gap_rows.append({'Condition': condition, 'File': p.name,
                                 'GapStart_s': round(s, 2), 'GapEnd_s': round(e, 2),
                                 'GapLength_s': round(e - s, 2)})
            for b, t, kind in unmatched:
                print(f"  WARNING {p.name}: unmatched {kind} for '{b}' at {t:.2f}s "
                      f"(this bout was not counted)")
            if source.startswith('last event'):
                print(f"  NOTE {p.name}: no media-duration column found; window "
                      f"estimated from last event ({window:.1f}s) — trailing gaps can't be detected")

            timeline.append({'Condition': condition, 'File': p.name,
                             'Window': window, 'Intervals': intervals, 'Gaps': real_gaps})

    budget = pd.DataFrame(rows)
    if len(budget):
        behavior_cols = [c for c in budget.columns if c not in (
            'Condition', 'File', 'Window_s', 'WindowSource', 'Covered_s', 'Unaccounted_s',
            'Unaccounted_pct', 'Overlap_s', 'N_gaps', 'Longest_gap_s', 'Unmatched_events')]
        budget[behavior_cols] = budget[behavior_cols].fillna(0)
    return budget, pd.DataFrame(gap_rows), timeline


def _behavior_palette(behaviors):
    # Okabe-Ito colorblind-safe palette (+ extras if many behaviors)
    base = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9',
            '#D55E00', '#F0E442', '#000000', '#7F3C8D', '#11A579']
    return {b: base[i % len(base)] for i, b in enumerate(behaviors)}


def create_time_budget_plot(budget, output_path=None):
    """
    Stacked horizontal bar per condition: average seconds spent in each
    behavior per observation window, plus the average unaccounted time (grey).
    Black tick = average window length.
    """
    output_path = output_path or OUTPUT_DIR / 'time_budget_by_condition.png'
    fixed = {'Condition', 'File', 'Window_s', 'WindowSource', 'Covered_s', 'Unaccounted_s',
             'Unaccounted_pct', 'Overlap_s', 'N_gaps', 'Longest_gap_s', 'Unmatched_events'}
    behaviors = sorted(c for c in budget.columns if c not in fixed)
    colors = _behavior_palette(behaviors)

    conds = [c for c in CONDITIONS if c in budget['Condition'].values]
    means = budget.groupby('Condition')[behaviors + ['Unaccounted_s', 'Window_s',
                                                      'Unaccounted_pct', 'Overlap_s']].mean()

    fig, ax = plt.subplots(figsize=(12, 1.2 * len(conds) + 2))
    for y, cond in enumerate(conds):
        left = 0
        for b in behaviors:
            w = means.loc[cond, b]
            if w > 0:
                ax.barh(y, w, left=left, color=colors[b], edgecolor='white', height=0.6)
                if w > means['Window_s'].max() * 0.04:
                    ax.text(left + w / 2, y, f"{w:.0f}", ha='center', va='center',
                            fontsize=FONT_SIZE, color='white', fontweight='bold')
            left += w
        u = means.loc[cond, 'Unaccounted_s']
        ax.barh(y, u, left=left, color='#BDBDBD', edgecolor='white', height=0.6,
                hatch='//')
        win = means.loc[cond, 'Window_s']
        ax.plot([win, win], [y - 0.38, y + 0.38], color='black', lw=2)
        label = f"{means.loc[cond, 'Unaccounted_pct']:.1f}% immobile"
        if means.loc[cond, 'Overlap_s'] > 0.5:
            label += f"\n({means.loc[cond, 'Overlap_s']:.0f}s overlapping behaviors)"
        ax.text(max(left + u, win) + win * 0.01, y, label, va='center', fontsize=FONT_SIZE)

    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels(conds, fontsize=FONT_SIZE)
    ax.invert_yaxis()
    ax.set_xlabel('Average time per observation window (s)', fontsize=FONT_SIZE, fontweight='bold')
    ax.set_title('Average Time Budget per Window — where did the time go?',
                 fontsize=FONT_SIZE, fontweight='bold')
    ax.set_xlim(0, means['Window_s'].max() * 1.35)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[b], label=b) for b in behaviors]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor='#BDBDBD', hatch='//',
                                 label='Immobile (no behavior scored)'))
    handles.append(plt.Line2D([0], [0], color='black', lw=2, label='Avg window length'))
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=min(4, len(handles)), fontsize=FONT_SIZE, frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nTime budget plot saved to {output_path}")
    plt.show()
    return fig


def create_gap_timeline_plot(timeline, output_path=None):
    """
    One row per file showing every scored bout across the window.
    Red shading = gaps where nothing was scored. Good for spotting
    exactly where/when coding is missing.
    """
    output_path = output_path or OUTPUT_DIR / 'gap_timeline_per_file.png'
    behaviors = sorted({b for t in timeline for b, _, _ in t['Intervals']})
    colors = _behavior_palette(behaviors)
    nb = max(len(behaviors), 1)
    lane_h = 0.8 / nb

    fig, ax = plt.subplots(figsize=(14, 0.45 * len(timeline) + 2))
    labels = []
    for y, t in enumerate(timeline):
        for s, e in t['Gaps']:
            ax.broken_barh([(s, e - s)], (y - 0.45, 0.9), facecolors='#F4A6A6', alpha=0.6)
        for b, s, e in t['Intervals']:
            lane = behaviors.index(b)
            ax.broken_barh([(s, e - s)], (y - 0.4 + lane * lane_h, lane_h),
                           facecolors=colors[b])
        ax.plot([t['Window'], t['Window']], [y - 0.45, y + 0.45], color='black', lw=1)
        labels.append(f"{t['Condition']}  |  {t['File']}")

    ax.set_yticks(range(len(timeline)))
    ax.set_yticklabels(labels, fontsize=FONT_SIZE)
    ax.invert_yaxis()
    ax.set_xlabel('Time in observation (s)', fontsize=FONT_SIZE, fontweight='bold')
    ax.set_title('Scored Behavior Timeline per File (red = Immobile)',
                 fontsize=FONT_SIZE, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[b], label=b) for b in behaviors]
    handles.append(plt.Rectangle((0, 0), 1, 1, color='#F4A6A6', label=f'Immobile (≥ {MIN_GAP_SECONDS:g}s unscored)'))
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.04 - 0.6 / max(len(timeline), 1)),
              ncol=min(6, len(handles)), fontsize=FONT_SIZE, frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gap timeline plot saved to {output_path}")
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────
# BASIC PLOTS (no statistics)
# ─────────────────────────────────────────────────────────────

def create_duration_plot(data, output_path=None):
    """
    Create subplots for each behavior showing mean duration across conditions.
    """
    output_path = output_path or OUTPUT_DIR / 'behavior_duration_barplot.png'
    print(f"\n*** ALL BEHAVIORS FOUND IN DATA: {sorted(data['Behavior'].unique())}")
    data = normalize_behaviors(data)
    print(f"*** BEHAVIORS AFTER FILTERING: {sorted(data['Behavior'].unique())}")

    behaviors = [b for b in BEHAVIORS_TO_KEEP if b in set(data['Behavior'])]
    n_behaviors = len(behaviors)

    if n_behaviors == 0:
        print("No data found for the specified behaviors!")
        return None

    fig, axes = plt.subplots(1, n_behaviors, figsize=(5 * n_behaviors, 6))
    plt.subplots_adjust(wspace=0.4)

    if n_behaviors == 1:
        axes = [axes]

    # Shared y-axis: tallest bar + error across ALL behaviors
    g = data.groupby(['Behavior', 'Condition'])['Duration']
    y_top = max(40, (g.mean() + g.sem().fillna(0)).max() * 1.1)

    for idx, behavior in enumerate(behaviors):
        ax = axes[idx]
        means, sems = [], []

        for condition in CONDITIONS:
            subset = data[(data['Condition'] == condition) & (data['Behavior'] == behavior)]
            means.append(subset['Duration'].mean() if len(subset) > 0 else 0)
            sems.append(subset['Duration'].sem()  if len(subset) > 0 else 0)

        x_positions = np.arange(len(CONDITIONS))

        for i, condition in enumerate(CONDITIONS):
            ax.bar(x_positions[i], means[i],
                   color=CONDITION_COLORS[condition],
                   alpha=0.8, edgecolor='black', linewidth=1.5,
                   yerr=sems[i], capsize=5,
                   error_kw={'linewidth': 2, 'elinewidth': 1.5})

        ax.set_xticks(x_positions)
        ax.set_ylim(0, y_top)
        ax.set_xticklabels(CONDITIONS, fontsize=FONT_SIZE, rotation=45, ha='right')
        if idx == 0:  # shared y-axis, so only label the first panel
            ax.set_ylabel('Duration (s)', fontsize=FONT_SIZE, fontweight='bold')
        ax.set_title(behavior, fontsize=FONT_SIZE, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

    fig.suptitle('Behavior Durations Across Conditions', fontsize=FONT_SIZE, fontweight='bold', y=0.98)


    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    
    print(f"\nDuration plot saved to {output_path}")
    plt.show()
    return fig


def create_frequency_plot(data, output_path=None):
    """
    Create subplots for each behavior showing frequency across conditions.
    """
    output_path = output_path or OUTPUT_DIR / 'behavior_frequency_barplot.png'
    data = normalize_behaviors(data)
    data = data[data['Behavior'] != 'Immobile']  # per-window total, no bout count

    count_data = data.groupby(['Condition', 'File', 'Behavior']).size().reset_index(name='Count')

    behaviors = [b for b in BEHAVIORS_TO_KEEP if b in set(count_data['Behavior'])]
    n_behaviors = len(behaviors)

    if n_behaviors == 0:
        print("No data found for the specified behaviors!")
        return None

    fig, axes = plt.subplots(1, n_behaviors, figsize=(5 * n_behaviors, 6))
    plt.subplots_adjust(wspace=0.4)

    if n_behaviors == 1:
        axes = [axes]

    g = count_data.groupby(['Behavior', 'Condition'])['Count']
    y_top = max(6, (g.mean() + g.sem().fillna(0)).max() * 1.1)

    for idx, behavior in enumerate(behaviors):
        ax = axes[idx]
        means, sems = [], []

        for condition in CONDITIONS:
            subset = count_data[(count_data['Condition'] == condition) &
                                (count_data['Behavior'] == behavior)]
            means.append(subset['Count'].mean() if len(subset) > 0 else 0)
            sems.append(subset['Count'].sem()  if len(subset) > 0 else 0)

        x_positions = np.arange(len(CONDITIONS))

        for i, condition in enumerate(CONDITIONS):
            ax.bar(x_positions[i], means[i],
                   color=CONDITION_COLORS[condition],
                   alpha=0.8, edgecolor='black', linewidth=1.5,
                   yerr=sems[i], capsize=5,
                   error_kw={'linewidth': 2, 'elinewidth': 1.5})

        ax.set_xticks(x_positions)
        ax.set_xticklabels(CONDITIONS, fontsize=FONT_SIZE, rotation=45, ha='right')
        if idx == 0:  # shared y-axis, so only label the first panel
            ax.set_ylabel('Frequency (count)', fontsize=FONT_SIZE, fontweight='bold')
        ax.set_title(behavior, fontsize=FONT_SIZE, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.set_ylim(0, y_top)

    fig.suptitle('Behavior Frequency Across Conditions', fontsize=FONT_SIZE, fontweight='bold', y=0.98)


    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFrequency plot saved to {output_path}")
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────
# STATISTICS (scipy only — no statsmodels, no scikit-posthocs)
# ─────────────────────────────────────────────────────────────

def perform_kruskal_wallis(data, behavior):
    """
    Kruskal-Wallis H-test across all 4 conditions.
    Returns (result_table, behavior_data) or None.
    """
    behavior_data = data[data['Behavior'] == behavior].copy()

    groups = [
        behavior_data[behavior_data['Condition'] == c]['MeanDuration'].values
        for c in CONDITIONS
        if len(behavior_data[behavior_data['Condition'] == c]) > 0
    ]

    if len(groups) < 2:
        return None

    try:
        h_stat, p_val = stats.kruskal(*groups)
        result_table = pd.DataFrame({
            'H-statistic': [round(h_stat, 4)],
            'p-value':     [round(p_val, 4)],
            'significant': [p_val < 0.05]
        })
        print("\nKruskal-Wallis Test:")
        print(result_table.to_string(index=False))
        print(f"\nSample sizes per condition:")
        print(behavior_data.groupby('Condition').size().to_string())
        return result_table, behavior_data
    except Exception as e:
        print(f"    Kruskal-Wallis failed for {behavior}: {e}")
        return None


def perform_posthoc_mannwhitney(data, behavior):
    """
    Pairwise Mann-Whitney U tests with Bonferroni correction.
    Returns a symmetric p-value DataFrame (same shape as scikit-posthocs output).
    """
    behavior_data = data[data['Behavior'] == behavior].copy()
    available = [c for c in CONDITIONS if c in behavior_data['Condition'].values]

    if len(available) < 2:
        return None

    try:
        n = len(available)
        p_matrix = pd.DataFrame(np.ones((n, n)), index=available, columns=available)
        pairs = list(combinations(available, 2))
        n_comparisons = len(pairs)

        for c1, c2 in pairs:
            g1 = behavior_data[behavior_data['Condition'] == c1]['MeanDuration']
            g2 = behavior_data[behavior_data['Condition'] == c2]['MeanDuration']

            if len(g1) > 1 and len(g2) > 1:
                _, p_raw = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                p_bonf = min(p_raw * n_comparisons, 1.0)
            else:
                p_bonf = 1.0

            p_matrix.loc[c1, c2] = p_bonf
            p_matrix.loc[c2, c1] = p_bonf

        return p_matrix
    except Exception as e:
        print(f"    Post-hoc failed for {behavior}: {e}")
        return None


def get_significance_stars(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    return 'ns'


def add_significance_bracket(ax, x1, x2, y, p_value, height_multiplier=0.05):
    sig_text = get_significance_stars(p_value)
    if sig_text == 'ns':
        return
    bracket_h = y * height_multiplier
    ax.plot([x1, x1, x2, x2], [y, y + bracket_h, y + bracket_h, y],
            lw=1.5, c='black')
    ax.text((x1 + x2) / 2, y + bracket_h * 1.2, sig_text,
            ha='center', va='bottom', fontsize=FONT_SIZE, fontweight='bold')


def create_duration_plot_with_stats(data, output_path=None):
    """
    Duration bar plots with Kruskal-Wallis + Bonferroni-corrected
    Mann-Whitney post-hoc significance brackets.
    """
    output_path = output_path or OUTPUT_DIR / 'behavior_duration_with_stats.png'
    data = normalize_behaviors(data)

    behaviors = [b for b in BEHAVIORS_TO_KEEP if b in set(data['Behavior'])]
    n_behaviors = len(behaviors)

    if n_behaviors == 0:
        print("No data found for the specified behaviors!")
        return None

    fig, axes = plt.subplots(1, n_behaviors, figsize=(5 * n_behaviors, 7))
    plt.subplots_adjust(wspace=0.4)

    if n_behaviors == 1:
        axes = [axes]

    # Shared y-axis (leaves room for significance brackets)
    g = data.groupby(['Behavior', 'Condition'])['MeanDuration']
    y_top = max((g.mean() + g.sem().fillna(0)).max() * 1.8, 1)

    print("\n" + "=" * 70)
    print("STATISTICAL RESULTS — DURATION (File-Level Analysis)")
    print("=" * 70)

    for idx, behavior in enumerate(behaviors):
        ax = axes[idx]

        # Compute per-condition summaries
        summary = []
        for condition in CONDITIONS:
            subset = data[(data['Condition'] == condition) & (data['Behavior'] == behavior)]
            if len(subset) > 0:
                summary.append({
                    'condition': condition,
                    'mean': subset['MeanDuration'].mean(),
                    'sem':  subset['MeanDuration'].sem(),
                    'n':    len(subset)
                })
            else:
                summary.append({'condition': condition, 'mean': 0, 'sem': 0, 'n': 0})

        summary_df = pd.DataFrame(summary)
        x_positions = np.arange(len(CONDITIONS))

        for i, row in summary_df.iterrows():
            ax.bar(x_positions[i], row['mean'],
                   color=CONDITION_COLORS[row['condition']],
                   alpha=0.8, edgecolor='black', linewidth=1.5,
                   yerr=row['sem'], capsize=5,
                   error_kw={'linewidth': 2, 'elinewidth': 1.5})

        print(f"\n{'─'*50}")
        print(f"BEHAVIOR: {behavior}")
        print(f"{'─'*50}")

        # Kruskal-Wallis
        kw_result = perform_kruskal_wallis(data, behavior)

        # Post-hoc pairwise tests
        posthoc = perform_posthoc_mannwhitney(data, behavior)
        if posthoc is not None:
            print("\nPost-hoc Mann-Whitney U (Bonferroni-corrected p-values):")
            print(posthoc.round(4).to_string())

            y_max = summary_df['mean'].max() + summary_df['sem'].max()
            comparisons = [
                ('Distress_Nonsocial', 'Distress_Social', 0),
                ('WN_Nonsocial',       'WN_Social',       1),
                ('Distress_Nonsocial', 'WN_Nonsocial',    2),
                ('Distress_Social',    'WN_Social',       3),
            ]
            bracket_y_start = y_max * 1.1

            for c1, c2, offset in comparisons:
                if c1 in posthoc.index and c2 in posthoc.columns:
                    p_val = posthoc.loc[c1, c2]
                    if p_val < 0.05:
                        i1 = CONDITIONS.index(c1)
                        i2 = CONDITIONS.index(c2)
                        bracket_y = bracket_y_start + offset * y_max * 0.15
                        add_significance_bracket(ax, i1, i2, bracket_y, p_val)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(CONDITIONS, fontsize=FONT_SIZE, rotation=45, ha='right')
        if idx == 0:  # shared y-axis, so only label the first panel
            ax.set_ylabel('Mean Duration (s)', fontsize=FONT_SIZE, fontweight='bold')
        ax.set_title(behavior, fontsize=FONT_SIZE, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        ax.set_ylim(bottom=0, top=y_top)

    fig.suptitle('Behavior Durations Across Conditions',
                 fontsize=FONT_SIZE, fontweight='bold', y=0.98)


    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\n{'='*70}")
    print(f"Duration plot saved to {output_path}")
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    file_lists = {
        'Distress_Nonsocial': [
            '/Volumes/T7/Day2.Redo.2weekstent/Batch1.Day2/AEF.D2.Diss.NS.csv',
            '/Volumes/T7/Day2.Redo.2weekstent/Batch1.Day2/BOC.D2.Diss.NS.csv',
            '/Volumes/T7/UPDATEBORIS/BIB.D1.Diss.Nonsocial.csv',
            '/Volumes/T7/UPDATEBORIS/AF8.Diss.D1.Nonsocial.csv',
            '/Volumes/T7/UPDATEBORIS/B06.D2.NS.Diss.csv',
            '/Volumes/T7/UPDATEBORIS/Batch2.B31.F.D1.NS.Diss.csv',
        ],
        'Distress_Social': [
            '/Volumes/T7/Day2.Redo.2weekstent/Batch1.Day2/AFD.D2.Diss.S.csv',
            '/Volumes/T7/UPDATEBORIS/B0F.D1.Diss.S.csv',
            '/Volumes/T7/UPDATEBORIS/B07.D1.Diss.S.csv',
            '/Volumes/T7/UPDATEBORIS/ADB.D1.S.Diss.Batch2.csv',
            '/Volumes/T7/UPDATEBORIS/B26.D2.Soc.Diss.csv',
            '/Volumes/T7/UPDATEBORIS/B31.WN.D2.Social.csv',
            '/Volumes/T7/ae7.d2.diss.s.tsv'
            ],
        'WN_Nonsocial': [
            '/Volumes/T7/Day2.Redo.2weekstent/Batch1.Day2/B07.D2.Wn.NS.csv',
            '/Volumes/T7/Day2.Redo.2weekstent/Batch1.Day2/B0F.WN.D2.NS.csv',
            '/Volumes/T7/UPDATEBORIS/AE7.WN.D1.Nonsocial.csv',
            '/Volumes/T7/UPDATEBORIS/AFD.D1.WN.Nonsocial.csv',
            '/Volumes/T7/UPDATEBORIS/ADB.D2.NS.Wn.csv',
            '/Volumes/T7/UPDATEBORIS/Batch2.F.D1.WN.NS.B26.csv',
        ],
        'WN_Social': [
            '/Volumes/T7/Day2.Redo.2weekstent/Batch1.Day2/AF8.WN.D2.S.csv',
            '/Volumes/T7/Day2.Redo.2weekstent/Batch1.Day2/BIB.WN.D2.S.csv',
            '/Volumes/T7/UPDATEBORIS/AFE.WN.D1.Social.csv',
            '/Volumes/T7/UPDATEBORIS/BOC.WN.D1.S.csv',
            '/Volumes/T7/UPDATEBORIS/B06.D1.S.WN.batch2.csv',
            '/Volumes/T7/UPDATEBORIS/B31.WN.D2.Social.csv'
        ],
    }

    # ── 4-behavior duration bar plot only ─────────────────────
    all_data = []
    for condition_name, csv_files in file_lists.items():
        print(f"\nProcessing condition: {condition_name}")
        condition_data = process_csv_files(csv_files, condition_name)
        if len(condition_data) > 0:
            all_data.append(condition_data)

    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        create_duration_plot(combined_data)
    else:
        print("\nNo data was processed. Please check your file paths.")