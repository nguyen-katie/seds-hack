"""
Real-data validation of the "FIND" stage.
SatNOGS decoded frames have timestamps -> we check whether our blind burst
detector fires where real SONATE-2 packets were actually received.

Two detectors:
  energy   : in-band power rises above the noise floor (classic)
  quieting : FM QUIETING - when a carrier is present, an FM discriminator's
             high-frequency noise drops sharply. Packets = dips in 7-10.5 kHz noise.
             (Physics-based, no thresholds hand-set: median - k*MAD of the recording.)

Run from repo root:  python3 src/validate_bursts.py
"""
import sys, json
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, spectrogram
from frontend import detect_bursts

OBS = ["15052729", "15039241", "15104225"]
CSV = "data/satnogs/sonate2_frames.csv"


def band_power_db(audio, fs, band, win_s=0.02):
    sos = butter(4, band, btype="bandpass", fs=fs, output="sos")
    x = sosfiltfilt(sos, audio.astype(np.float64))
    w = int(win_s * fs); n = len(x) // w
    return 10 * np.log10((x[:n * w].reshape(n, w) ** 2).mean(axis=1) + 1e-20), win_s


def detect_quieting(audio, fs, band=(7000, 10500), k=3.0, min_len_s=0.06, merge_s=0.1):
    p, win_s = band_power_db(audio, fs, band)
    # smooth over ~3 windows to suppress single-window noise
    p = np.convolve(p, np.ones(3) / 3, mode="same")
    med = np.median(p)
    mad = np.median(np.abs(p - med)) + 1e-9
    on = p < med - k * mad                      # noise DROPS when a carrier is present
    segs, start = [], None
    for i, v in enumerate(np.append(on, False)):
        if v and start is None: start = i
        elif not v and start is not None: segs.append([start, i]); start = None
    merged = []
    for s in segs:
        if merged and (s[0] - merged[-1][1]) * win_s < merge_s: merged[-1][1] = s[1]
        else: merged.append(s)
    w = int(win_s * fs)
    return [(a * w, b * w) for a, b in merged if (b - a) * win_s >= min_len_s]


def recall(bursts, fs, t_frames, tol=1.0):
    if len(t_frames) == 0:
        return float("nan")
    spans = np.array([(a / fs, b / fs) for a, b in bursts]) if bursts else np.zeros((0, 2))
    hit = [np.any((spans[:, 0] - tol <= t) & (spans[:, 1] + tol >= t)) for t in t_frames]
    return float(np.mean(hit))


def frame_times(obs, meta):
    df = pd.read_csv(CSV)
    df = df[df.observation_id.astype("Int64").astype(str) == obs]
    start = pd.to_datetime(meta["start"], utc=True)
    t = (pd.to_datetime(df.timestamp, utc=True) - start).dt.total_seconds().values
    return np.sort(t)


def plot(audio, fs, bursts, t_frames, title, path):
    import matplotlib.pyplot as plt
    f, t, S = spectrogram(audio, fs, nperseg=1024, noverlap=512)
    sel = f <= 12000
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.pcolormesh(t, f[sel], 10 * np.log10(S[sel] + 1e-12), shading="auto", cmap="viridis")
    for a, b in bursts:
        ax.axvspan(a / fs, b / fs, ymin=0, ymax=0.08, color="red", alpha=0.9)
    ax.vlines(t_frames, 11200, 12000, color="white", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time since observation start (s)"); ax.set_ylabel("Audio frequency (Hz)")
    ax.text(0.01, 0.97, "white ticks = SatNOGS decoded frames   red bars = our blind detections",
            transform=ax.transAxes, color="white", fontsize=8, va="top")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    rows = []
    for obs in OBS:
        d = f"data/satnogs/sonate2_{obs}"
        audio, fs = sf.read(f"{d}/audio.ogg", dtype="float32")
        if audio.ndim > 1: audio = audio.mean(axis=1)
        with open(f"{d}/obs.json") as fh:
            meta = json.load(fh)
        if isinstance(meta, list): meta = meta[0]
        tf = frame_times(obs, meta)
        tf = tf[(tf >= 0) & (tf <= len(audio) / fs)]
        b_energy = detect_bursts(audio, fs, band=(300, 8000), merge_s=0.1)
        b_quiet = detect_quieting(audio, fs)
        r = dict(obs=obs, satnogs_frames=len(tf),
                 energy_bursts=len(b_energy), energy_recall=recall(b_energy, fs, tf),
                 quieting_bursts=len(b_quiet), quieting_recall=recall(b_quiet, fs, tf))
        rows.append(r)
        print(f"obs {obs}: SatNOGS frames {len(tf):4} | energy: {len(b_energy):4} bursts, "
              f"recall {r['energy_recall']:.0%} | FM-quieting: {len(b_quiet):4} bursts, "
              f"recall {r['quieting_recall']:.0%}")
        best = b_quiet if r["quieting_recall"] >= r["energy_recall"] else b_energy
        name = "FM-quieting" if best is b_quiet else "energy"
        plot(audio, fs, best, tf,
             f"SONATE-2 obs {obs}: blind {name} detector vs SatNOGS decoded frames "
             f"(recall {max(r['quieting_recall'], r['energy_recall']):.0%})",
             f"results/validate_{obs}.png")
    pd.DataFrame(rows).to_csv("results/burst_validation.csv", index=False)
    print("Saved results/validate_*.png and results/burst_validation.csv")