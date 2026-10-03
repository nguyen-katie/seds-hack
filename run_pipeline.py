"""End-to-end pipeline: decode every downloaded SONATE-2 pass, write telemetry logs,
compare against SatNOGS ground truth and make the result plots.

Usage:
    python run_pipeline.py              # all passes in data/satnogs/sonate2_*/
    python run_pipeline.py 15104225     # one pass
    python run_pipeline.py --force      # re-decode even if a telemetry log already exists
    python run_pipeline.py --no-open    # don't pop up the summary graph at the end
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

import decoder

DATA = "data/satnogs"
RESULTS = "results"
BLUE, ORANGE, GRAY = "#2a6fdb", "#e8743b", "#9aa0a6"


def pass_dirs(only=None):
    dirs = sorted(glob.glob(f"{DATA}/sonate2_*/"))
    return [d for d in dirs if not only or any(o in d for o in only)]


def obs_meta(d):
    return json.load(open(os.path.join(d, "obs.json")))


# ---------- 1. Decode -> telemetry log ----------

def decode_pass(d, force=False):
    """Decode one pass and save results/telemetry_<id>.csv. Cached unless force=True."""
    o = obs_meta(d)
    out = f"{RESULTS}/telemetry_{o['id']}.csv"
    if os.path.exists(out) and not force:
        return pd.read_csv(out)
    print(f"Decoding pass {o['id']} ...", flush=True)
    frames = decoder.decode(os.path.join(d, "audio.ogg"))
    start = pd.Timestamp(o["start"])
    rows = []
    for r in frames:
        p = decoder.parse_ax25(r["frame"])
        rows.append({
            "pass_id": o["id"],
            "audio_time_s": r["t"],
            "utc_approx": (start + pd.Timedelta(seconds=r["t"])).isoformat(),
            "src": p["src"],
            "dst": p["dst"],
            "length": len(r["frame"]),
            "corrected_bits": r["corrected_bits"],
            "frame": r["frame"].hex().upper(),
            "payload_hex": p["payload"].hex(),
        })
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"  {len(df)} CRC-valid packets -> {out}", flush=True)
    return df


# ---------- 2. Before/after spectra ----------

def remodulate(tlog, fs, n_samples, amplitude):
    """Rebuild the noise-free baseband signal from the decoded packets.

    Each verified packet is re-encoded exactly as the satellite sends it
    (FCS, bit stuffing, flags, NRZI, G3RUH scrambling, Gaussian pulse shaping)
    and placed at the time it was received. Silence where nothing was decoded.
    """
    sps = int(fs / decoder.BAUD)
    out = np.zeros(n_samples)
    for r in tlog.itertuples():
        frame = bytes.fromhex(r.frame)
        crc = decoder.crc16_x25(frame)
        bits = "".join(format(b, "08b")[::-1] for b in frame + bytes([crc & 0xFF, crc >> 8]))
        stuffed, ones = [], 0
        for b in bits:
            stuffed.append(b)
            ones = ones + 1 if b == "1" else 0
            if ones == 5:
                stuffed.append("0")
                ones = 0
        d = np.array([int(b) for b in decoder.FLAG * 4 + "".join(stuffed) + decoder.FLAG], np.uint8)
        nrzi = np.cumsum(d == 0) % 2   # 0 -> transition, 1 -> no transition
        s = np.zeros(len(nrzi) + 17, np.uint8)
        for k in range(len(nrzi)):     # G3RUH scrambler 1 + x^12 + x^17
            s[k + 17] = nrzi[k] ^ s[k + 5] ^ s[k]
        wave = np.repeat(2.0 * s[17:] - 1, sps)
        start = int((r.audio_time_s - 24 / decoder.BAUD) * fs)   # frame time marks its first flag
        if 0 <= start and start + len(wave) <= n_samples:
            out[start:start + len(wave)] = wave
    taps = signal.windows.gaussian(4 * sps + 1, std=sps * 0.45)   # GMSK BT ~ 0.5
    return amplitude * np.convolve(out, taps / taps.sum(), "same")


def plot_spectra(d, tlog):
    o = obs_meta(d)
    x, fs = decoder.load_audio(os.path.join(d, "audio.ogg"))
    x = x - x.mean()
    y = remodulate(tlog, fs, len(x), np.std(x))

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1.2])

    def spec(sig):
        f, t, S = signal.spectrogram(sig, fs, nperseg=1024, noverlap=512)
        return f, t, 10 * np.log10(S + 1e-12)

    f, t, Sx = spec(x)
    vmin, vmax = np.percentile(Sx, 5), np.percentile(Sx, 99.5)
    for row, (sig, label) in enumerate([
            (x, "Before: raw pass audio from the ground station"),
            (y, f"After: clean signal rebuilt from the {len(tlog)} decoded, CRC-verified packets")]):
        ax = fig.add_subplot(gs[row, 0])
        f, t, S = spec(sig)
        ax.pcolormesh(t, f / 1000, S, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_ylim(0, 12)
        ax.set_ylabel("Frequency (kHz)")
        ax.set_title(label, loc="left")
    ax.set_xlabel("Time into pass (s)")

    ax = fig.add_subplot(gs[:, 1])
    for sig, label, color in [(x, "before (signal + noise)", GRAY), (y, "after (recovered signal)", BLUE)]:
        f, P = signal.welch(sig[np.abs(y) > 0] if label.startswith("after") else sig, fs, nperseg=4096)
        ax.plot(f / 1000, 10 * np.log10(P + 1e-20), color=color, lw=2, label=label)
    ax.axvline(decoder.BAUD / 2000, color=ORANGE, ls="--", lw=1.5, label="4.8 kHz (half the bit rate)")
    ax.set_xlim(0, 12)
    ax.set_ylim(-110, None)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title("Average spectrum", loc="left")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)

    fig.suptitle(f"SONATE-2 pass {o['id']} ({o['start'][:10]}, {o['station_name']})", x=0.01, ha="left")
    plt.tight_layout()
    out = f"{RESULTS}/spectrum_{o['id']}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  saved {out}")


# ---------- 3. Compare with SatNOGS ----------

def compare(dirs, logs):
    truth = pd.read_csv(f"{DATA}/sonate2_frames_clean.csv")
    rows = []
    for d, tlog in zip(dirs, logs):
        o = obs_meta(d)
        ours = set(tlog["frame"])
        same_station = set(truth.loc[truth["observation_id"] == o["id"], "frame"])
        all_stations = set(truth.loc[truth["pass_id"] == o["id"], "frame"])
        rows.append({
            "pass_id": o["id"],
            "date": o["start"][:10],
            "station": o["station_name"],
            "satnogs_same_recording": len(same_station),
            "ours": len(ours),
            "both": len(ours & same_station),
            "only_ours": len(ours - same_station),
            "only_ours_confirmed_by_other_stations": len((ours - same_station) & all_stations),
            "only_satnogs": len(same_station - ours),
            "repaired_by_soft_decision": int((tlog["corrected_bits"] > 0).sum()),
            "all_stations_combined": len(all_stations),
        })
    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS}/comparison.csv", index=False)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 4))
    xpos = np.arange(len(df))
    w = 0.38
    b1 = ax.bar(xpos - w / 2, df["satnogs_same_recording"], w, color=GRAY, label="SatNOGS decoder")
    b2 = ax.bar(xpos + w / 2, df["ours"], w, color=BLUE, label="Our decoder")
    ax.bar_label(b1, padding=3)
    ax.bar_label(b2, padding=3)
    ax.set_xticks(xpos, [f"{r.pass_id}\n{r.date}, {r.station.split()[0]}" for r in df.itertuples()])
    ax.set_ylabel("Unique CRC-valid packets")
    ax.set_title("Packets decoded from the same recording", loc="left")
    ax.legend(loc="upper left")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    fig.savefig(f"{RESULTS}/comparison.png", dpi=120)
    plt.close(fig)
    print(f"saved {RESULTS}/comparison.png")
    return df


# ---------- 4. Packet loss from the satellite's own counter ----------

def unwrap_counter(times, payloads):
    """Turn the 8-bit packet counter (payload byte 2) into a running packet number.

    Steps are taken as signed (-128..127), so wraps at 255 -> 0 and small reorderings are handled.
    """
    order = np.argsort(times, kind="stable")
    c = np.array([p[2] for p in payloads], int)[order]
    steps = ((np.diff(c) + 128) % 256) - 128
    seq = np.empty(len(c), int)
    seq[order] = np.concatenate([[0], np.cumsum(steps)])
    return seq


def is_telemetry(p):
    return len(p) > 3 and p[0] == 0x27   # all SONATE-2 telemetry payloads start with 0x27


def counter_loss(dirs, logs):
    """Estimate packet loss from gaps in the counter, and check it against all SatNOGS stations."""
    truth_path = f"{DATA}/sonate2_frames_clean.csv"
    truth = pd.read_csv(truth_path, parse_dates=["timestamp"]) if os.path.exists(truth_path) else None
    rows, strips = [], []
    for d, tlog in zip(dirs, logs):
        o = obs_meta(d)
        g = tlog.assign(p=tlog["payload_hex"].apply(bytes.fromhex))
        g = g[g["p"].apply(is_telemetry)].copy()
        g["seq"] = unwrap_counter(g["audio_time_s"].values, g["p"].tolist())
        sent = int(g["seq"].max() - g["seq"].min() + 1)
        got = g["seq"].nunique()
        row = {"pass_id": o["id"], "decoded": got, "counter_says_sent": sent,
               "estimated_loss_pct": round(100 * (1 - got / sent), 1)}

        if truth is not None:
            # Align our packet numbers with the numbering of all stations combined
            u = truth[truth["pass_id"] == o["id"]].drop_duplicates("frame")
            u = u.assign(p=u["payload_hex"].apply(bytes.fromhex))
            u = u[u["p"].apply(is_telemetry)].copy()
            u["seq"] = unwrap_counter(u["timestamp"].values.astype("int64"), u["p"].tolist())
            shared = g.merge(u[["frame", "seq"]], on="frame", suffixes=("", "_all"))
            offset = int((shared["seq_all"] - shared["seq"]).mode()[0])
            ours = set(g["seq"] + offset)
            others = set(u["seq"])
            lo, hi = min(ours), max(ours)
            row.update({"all_stations_in_same_span": len({s for s in others if lo <= s <= hi}),
                        "offset_consistent": f"{(shared['seq_all'] - shared['seq'] == offset).sum()}/{len(shared)}"})
            span = range(min(others | ours), max(others | ours) + 1)
            strips.append((o, [("ours" if s in ours else "others" if s in others else "nobody") for s in span]))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS}/counter_loss.csv", index=False)
    print(df.to_string(index=False))
    if strips:
        plot_counter_strips(strips, df)
    return df


def plot_counter_strips(strips, df):
    colors = {"ours": BLUE, "others": ORANGE, "nobody": GRAY}
    labels = {"ours": "decoded by us",
              "others": "missed by us, received by another station",
              "nobody": "received by no station (counter gap)"}
    fig, axes = plt.subplots(len(strips), 1, figsize=(13, 1.6 * len(strips) + 1.2))
    for ax, (o, cats), r in zip(np.atleast_1d(axes), strips, df.itertuples()):
        for cat, color in colors.items():
            xs = [k for k, c in enumerate(cats) if c == cat]
            ax.bar(xs, 1, width=1.0, color=color, linewidth=0, label=labels[cat])
        ax.set_xlim(-0.5, len(cats) - 0.5)
        ax.set_yticks([])
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.set_title(f"Pass {o['id']} ({o['start'][:10]}, {o['station_name'].split()[0]}): "
                     f"decoded {r.decoded} of {r.counter_says_sent} packets the counter says were sent "
                     f"in our window, {r.estimated_loss_pct:.0f}% loss", loc="left", fontsize=10)
    axes[-1].set_xlabel("Packet number (from the satellite's own counter)")
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0))
    fig.suptitle("Every packet SONATE-2 sent during each pass", x=0.01, ha="left")
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(f"{RESULTS}/packet_counter.png", dpi=120)
    plt.close(fig)
    print(f"saved {RESULTS}/packet_counter.png")


def main():
    args = sys.argv[1:]
    force = "--force" in args
    only = [a for a in args if not a.startswith("--")]
    dirs = pass_dirs(only)
    if not dirs:
        sys.exit(f"No passes found in {DATA}/. Run: python fetch_data.py")
    os.makedirs(RESULTS, exist_ok=True)

    logs = []
    for d in dirs:
        tlog = decode_pass(d, force)
        plot_spectra(d, tlog)
        logs.append(tlog)

    pd.concat(logs).to_csv(f"{RESULTS}/telemetry_log.csv", index=False)
    print(f"saved {RESULTS}/telemetry_log.csv")
    if os.path.exists(f"{DATA}/sonate2_frames_clean.csv"):
        compare(dirs, logs)
    counter_loss(dirs, logs)
    if os.path.exists(f"{DATA}/sonate2_frames_clean.csv"):
        import review_results
        review_results.main(open_png="--no-open" not in args)


if __name__ == "__main__":
    main()
