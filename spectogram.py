"""
Publication-ready spectrograms + power spectra (FFT), Arial 11 pt.

Adds what the reviewer asked for ("show the FFTs for each call type"):

1. plot_spectrogram()  - your original spectrogram, now with an optional
   power-spectrum panel on the right that shares the frequency axis, so
   spectral peaks line up with the bright bands in the spectrogram.

2. plot_mean_spectra() - mean power spectrum (± SD) for each call type,
   averaged over several example calls, all overlaid on one plot so
   call types can be compared directly. This is the most direct answer
   to the reviewer's comment.

Both print peak frequency and -10 dB bandwidth so you can report them.

Spectrum method: power |STFT|^2 averaged over time frames using the
same FFT size / window / overlap as the spectrogram (equivalent to
Welch's method). Each call is normalized to its own maximum (0 dB).
"""
import librosa
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # registers the 'rocket' colormap

# ── Global style: Arial 11 pt everywhere ──────────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,   # keep text editable in Illustrator/Inkscape
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

FIG_WIDTHS = {"single": 3.5, "onehalf": 5.5, "double": 7.2}  # inches


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_stft(wav_file, n_fft, overlap, window):
    """Load at native sample rate and return magnitude STFT + axes."""
    y, sr = librosa.load(wav_file, sr=None, mono=True)
    hop = int(n_fft * (1 - overlap))
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, window=window))
    freqs_khz = librosa.fft_frequencies(sr=sr, n_fft=n_fft) / 1000
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)
    return S, sr, hop, freqs_khz, times


def _power_spectrum_db(S, times, tmin=None, tmax=None):
    """Time-averaged power spectrum in dB re: its own max (0 dB peak)."""
    mask = np.ones_like(times, dtype=bool)
    if tmin is not None:
        mask &= times >= tmin
    if tmax is not None:
        mask &= times <= tmax
    P = np.mean(S[:, mask] ** 2, axis=1)
    return 10 * np.log10(P / P.max() + 1e-20)


def _spectral_summary(freqs_khz, spec_db, drop_db=10, fmin_khz=None, fmax_khz=None):
    """Peak frequency and bandwidth at `drop_db` below the peak (kHz)."""
    band = np.ones_like(freqs_khz, dtype=bool)
    if fmin_khz is not None:
        band &= freqs_khz >= fmin_khz
    if fmax_khz is not None:
        band &= freqs_khz <= fmax_khz
    f, s = freqs_khz[band], spec_db[band]
    i_pk = np.argmax(s)
    thresh = s[i_pk] - drop_db
    # walk outward from the peak until the spectrum falls below threshold
    lo = i_pk
    while lo > 0 and s[lo - 1] >= thresh:
        lo -= 1
    hi = i_pk
    while hi < len(s) - 1 and s[hi + 1] >= thresh:
        hi += 1
    return f[i_pk], f[lo], f[hi]


# ── 1. Spectrogram (+ optional side spectrum) ─────────────────────────────────
def plot_spectrogram(wav_file,
                     save_path=None,
                     n_fft=1024,          # FFT window length (samples)
                     overlap=0.75,        # fraction of window overlap
                     window="hann",
                     db_floor=-80,        # dB below max shown as darkest color
                     fmin_khz=None, fmax_khz=None,
                     tmin=None, tmax=None,
                     width="single", height=2.5,
                     cmap="rocket",
                     colorbar=True,
                     panel_label=None,    # e.g. "A"
                     title=None,          # e.g. "Aggression"
                     show_spectrum=True,  # add power-spectrum panel on the right
                     spec_floor=-60,      # dB limit of the spectrum panel x-axis
                     spec_color="black"):
    S, sr, hop, freqs_khz, times = _load_stft(wav_file, n_fft, overlap, window)
    S_dB = librosa.amplitude_to_db(S, ref=np.max)

    fig_w = FIG_WIDTHS.get(width, width)
    if show_spectrum:
        ratios = [4, 1.3] + ([0.18] if colorbar else [])
        fig = plt.figure(figsize=(fig_w, height))
        gs = fig.add_gridspec(1, len(ratios), width_ratios=ratios, wspace=0.08)
        ax = fig.add_subplot(gs[0])
        ax_spec = fig.add_subplot(gs[1], sharey=ax)
        cax = fig.add_subplot(gs[2]) if colorbar else None
    else:
        fig, ax = plt.subplots(figsize=(fig_w, height))
        ax_spec, cax = None, None

    mesh = ax.pcolormesh(times, freqs_khz, S_dB, shading="auto", cmap=cmap,
                         vmin=db_floor, vmax=0, rasterized=True)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_ylim(fmin_khz if fmin_khz is not None else 0,
                fmax_khz if fmax_khz is not None else sr / 2000)
    if tmin is not None or tmax is not None:
        ax.set_xlim(tmin if tmin is not None else times[0],
                    tmax if tmax is not None else times[-1])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title)

    if show_spectrum:
        # spectrum over the same time window that is displayed
        spec_db = _power_spectrum_db(S, times, tmin, tmax)
        ax_spec.plot(spec_db, freqs_khz, color=spec_color, lw=0.9)
        ax_spec.set_xlim(spec_floor, 0)
        ax_spec.set_xlabel("Power\n(dB)")
        ax_spec.tick_params(axis="y", labelleft=False)
        for side in ("top", "right"):
            ax_spec.spines[side].set_visible(False)

        pk, lo, hi = _spectral_summary(freqs_khz, spec_db, 10, fmin_khz, fmax_khz)
        print(f"  peak = {pk:.1f} kHz, -10 dB band = {lo:.1f}-{hi:.1f} kHz")

    if colorbar:
        if cax is not None:
            cb = fig.colorbar(mesh, cax=cax)
        else:
            cb = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.05)
        cb.set_label("Relative amplitude (dB)")
        cb.outline.set_linewidth(0.8)

    if panel_label:
        ax.text(-0.30, 1.06, panel_label, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom")

    if not show_spectrum:
        fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches="tight")

    # Print parameters for your figure legend
    print(f"{wav_file}: sr = {sr} Hz, FFT = {n_fft} pts ({window}), "
          f"overlap = {overlap:.0%}, freq res = {sr / n_fft:.1f} Hz, "
          f"time res = {hop / sr * 1000:.2f} ms")
    plt.show()
    return fig, ax


# ── 2. Mean spectrum per call type (overlay) ──────────────────────────────────
def plot_mean_spectra(call_files,          # {"Aggression": [wav, wav, ...], ...}
                      save_path=None,
                      n_fft=1024, overlap=0.75, window="hann",
                      fmin_khz=None, fmax_khz=None,
                      db_floor=-60,
                      show_sd=True,
                      width="single", height=2.5,
                      colors=None,
                      panel_label=None):
    """
    Each wav should contain ONE call, trimmed tightly (little silence),
    otherwise background noise is averaged into the spectrum.
    """
    fig, ax = plt.subplots(figsize=(FIG_WIDTHS.get(width, width), height))
    if colors is None:
        colors = sns.color_palette("colorblind", len(call_files))

    ref_sr, ref_f = None, None
    for (name, files), color in zip(call_files.items(), colors):
        specs = []
        for wav in files:
            S, sr, hop, freqs_khz, times = _load_stft(wav, n_fft, overlap, window)
            if ref_sr is None:
                ref_sr, ref_f = sr, freqs_khz
            elif sr != ref_sr:
                raise ValueError(f"{wav} has sr={sr}, expected {ref_sr}. "
                                 "All files must share one sample rate.")
            specs.append(_power_spectrum_db(S, times))
        specs = np.vstack(specs)
        mean, sd = specs.mean(axis=0), specs.std(axis=0)

        ax.plot(ref_f, mean, color=color, lw=1.0, label=f"{name} (n = {len(files)})")
        if show_sd and len(files) > 1:
            ax.fill_between(ref_f, mean - sd, mean + sd, color=color,
                            alpha=0.25, lw=0)

        pk, lo, hi = _spectral_summary(ref_f, mean, 10, fmin_khz, fmax_khz)
        print(f"{name}: n = {len(files)}, peak = {pk:.1f} kHz, "
              f"-10 dB band = {lo:.1f}-{hi:.1f} kHz")

    ax.set_xlim(fmin_khz if fmin_khz is not None else 0,
                fmax_khz if fmax_khz is not None else ref_sr / 2000)
    ax.set_ylim(db_floor, 3)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Relative power (dB)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)

    if panel_label:
        ax.text(-0.25, 1.06, panel_label, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches="tight")

    print(f"sr = {ref_sr} Hz, FFT = {n_fft} pts ({window}), overlap = {overlap:.0%}, "
          f"freq res = {ref_sr / n_fft:.1f} Hz; mean ± SD of per-call spectra, "
          f"each normalized to its own peak")
    plt.show()
    return fig, ax


# === Example usage ===
if __name__ == "__main__":
    wav_file = "/Users/zariageorge/Desktop/spectrogram/echotest.wav"

    # Spectrogram with the spectrum panel beside it (shared frequency axis)
    plot_spectrogram(wav_file, save_path="/Users/zariageorge/Desktop/Echo1.pdf",
                     width="single", height=2.5)

    