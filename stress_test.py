"""Stress-test the decoder on a real SONATE-2 recording with injected impairments.

1. Noise sweep:   add increasing noise; hard-decision decoding vs. soft-decision rescue
2. Dropouts:      replace stretches of the pass with signal-free noise; does it re-acquire?
3. Freq. offset:  add constant and drifting frequency offsets (extra Doppler)

    python stress_test.py
Outputs: results/stress_test.png, results/stress_*.csv
"""
import json
import zlib
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

import decoder

PASS = "data/satnogs/sonate2_15104225"
SEG = (40.0, 108.0)          # the part of the pass where the satellite is audible
NOISE_REF = (2.0, 25.0)      # satellite below the horizon: receiver noise only
DROPOUTS = [(50, 53), (60, 65), (75, 77), (88, 93)]   # seconds into the recording
NOISE_DB = [None, -30, -24, -21, -18, -16, -14, -12]
OFFSETS = [("none", 0, 0), ("constant +0.5", 0.5, 0), ("constant +2", 2.0, 0),
           ("drift +-1 @ 0.5 Hz", 1.0, 0.5), ("drift +-1 @ 2 Hz", 1.0, 2.0)]
RESULTS = "results"
BLUE, ORANGE, GRAY = "#2a6fdb", "#e8743b", "#9aa0a6"


def load():
    x, fs = decoder.load_audio(f"{PASS}/audio.ogg")
    return x, fs


def segment(x, fs):
    return x[int(SEG[0] * fs):int(SEG[1] * fs)]


def run(job):
    """Decode one impaired copy of the segment. Returns (job name, list of (time, frame))."""
    name, kind, param, flips = job
    x, fs = load()
    rng = np.random.default_rng(zlib.crc32(name.encode()))   # reproducible per job
    s = segment(x, fs).copy()
    if kind == "noise" and param is not None:
        n = rng.standard_normal(len(s))
        n = signal.sosfiltfilt(signal.butter(6, 11000, "lowpass", fs=fs, output="sos"), n)   # same band as the recording
        n *= np.std(s) * 10 ** (param / 20) / np.std(n)
        s = s + n
    elif kind == "dropout":
        ref = x[int(NOISE_REF[0] * fs):int(NOISE_REF[1] * fs)]
        for a, b in DROPOUTS:
            i0, i1 = int((a - SEG[0]) * fs), int((b - SEG[0]) * fs)
            k = rng.integers(0, len(ref) - (i1 - i0))
            s[i0:i1] = ref[k:k + (i1 - i0)]
    elif kind == "offset":
        amp, rate = param
        level = np.std(s)
        t = np.arange(len(s)) / fs
        s = s + amp * level * (np.sin(2 * np.pi * rate * t) if rate else 1.0)
    frames = decoder.decode_audio(s, fs, max_flips=flips)
    return name, [(r["t"] + SEG[0], r["frame"].hex()) for r in frames]


def main():
    jobs = []
    for db in NOISE_DB:
        for flips in (0, 2):
            jobs.append((f"noise {db} flips {flips}", "noise", db, flips))
    jobs.append(("dropout", "dropout", None, 2))
    for label, amp, rate in OFFSETS[1:]:
        jobs.append((f"offset {label}", "offset", (amp, rate), 2))
    with Pool(8) as pool:
        out = dict(pool.map(run, jobs))

    base = out["noise None flips 2"]
    base_frames = {f for _, f in base}

    # 1. Noise sweep
    rows = []
    for db in NOISE_DB:
        for flips in (0, 2):
            got = {f for _, f in out[f"noise {db} flips {flips}"]}
            rows.append({"added_noise_db": db, "rescue": flips > 0, "decoded": len(got),
                         "pct_of_clean": round(100 * len(got & base_frames) / len(base_frames), 1),
                         "not_in_clean_run": len(got - base_frames)})
    noise = pd.DataFrame(rows)
    noise.to_csv(f"{RESULTS}/stress_noise.csv", index=False)
    print(noise.to_string(index=False))

    # 2. Dropouts: how soon after each dropout ends do we decode again?
    drop = out["dropout"]
    drop_rows = []
    for a, b in DROPOUTS:
        nxt_base = min((t for t, _ in base if t >= b), default=np.nan)
        nxt = min((t for t, _ in drop if t >= b), default=np.nan)
        drop_rows.append({"dropout": f"{a}-{b} s", "length_s": b - a,
                          "first_packet_after_s": round(nxt - b, 2),
                          "first_packet_after_in_clean_run_s": round(nxt_base - b, 2)})
    drops = pd.DataFrame(drop_rows)
    drops.to_csv(f"{RESULTS}/stress_dropouts.csv", index=False)
    print(drops.to_string(index=False))

    # 3. Frequency offsets
    off = pd.DataFrame([{"offset": label, "decoded": len(out["noise None flips 2"] if amp == 0 else out[f"offset {label}"])}
                        for label, amp, rate in OFFSETS])
    off.to_csv(f"{RESULTS}/stress_offset.csv", index=False)
    print(off.to_string(index=False))

    plot(noise, base, drop, drops, off)


def plot(noise, base, drop, drops, off):
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.8])

    ax = fig.add_subplot(gs[0, 0])
    xs = [d for d in NOISE_DB if d is not None]
    for rescue, color, label in [(False, GRAY, "hard decision only"), (True, BLUE, "with soft-decision rescue")]:
        n = noise[(noise["rescue"] == rescue) & noise["added_noise_db"].notna()]
        ax.plot(n["added_noise_db"], n["decoded"], marker="o", lw=2, color=color, label=label)
    clean = noise[noise["added_noise_db"].isna() & noise["rescue"]]["decoded"].iloc[0]
    ax.axhline(clean, color=GRAY, ls="--", lw=1, label=f"no added noise ({clean})")
    ax.set_xticks(xs)
    ax.set_xlabel("Added noise power, dB relative to the recording (audio domain)")
    ax.set_ylabel("Packets decoded (CRC-valid)")
    ax.set_title("1. Sensitivity: decoding as noise increases", loc="left")
    ax.grid(alpha=0.2)
    ax.legend()

    ax = fig.add_subplot(gs[0, 1])
    ax.barh(off["offset"][::-1], off["decoded"][::-1], color=BLUE)
    for i, v in enumerate(off["decoded"][::-1]):
        ax.text(v + 1, i, str(v), va="center")
    ax.set_xlabel("Packets decoded")
    ax.set_title("3. Frequency offset / drift (× signal swing)", loc="left")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    ax = fig.add_subplot(gs[1, :])
    for a, b in DROPOUTS:
        ax.axvspan(a, b, color=ORANGE, alpha=0.25, lw=0)
    ax.plot([t for t, _ in base], np.ones(len(base)), "|", color=GRAY, ms=18, mew=2, label="clean recording")
    ax.plot([t for t, _ in drop], np.zeros(len(drop)), "|", color=BLUE, ms=18, mew=2, label="with dropouts")
    ax.set_yticks([0, 1], ["with dropouts", "clean"])
    ax.set_ylim(-0.7, 1.7)
    ax.set_xlim(*SEG)
    ax.set_xlabel("Time into pass (s)   ·   orange = signal replaced by receiver noise")
    delays = ", ".join(f"{d:.1f}s" for d in drops["first_packet_after_s"])
    ax.set_title(f"2. Re-acquisition after dropouts: first packet decoded {delays} after each dropout ends",
                 loc="left")
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)

    fig.suptitle("Stress test on a real SONATE-2 recording (pass 15104225)", x=0.01, ha="left", fontsize=13)
    plt.tight_layout()
    fig.savefig(f"{RESULTS}/stress_test.png", dpi=120)
    plt.close(fig)
    print(f"saved {RESULTS}/stress_test.png")


if __name__ == "__main__":
    main()
