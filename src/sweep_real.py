"""
Money-chart experiment: REAL SONATE-2 packets (from SatNOGS) through a simulated
noisy AFSK 1200 channel. Compares plain decoding vs soft-decision Chase rescue.

Usage:  python src/sweep_real.py data/sonate2_frames.csv
Output: results/sweep_real.csv, results/sweep_real.png
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from afsk import crc_x25, frame_to_bits, afsk_modulate, decode_audio, decode_frames

CSV = sys.argv[1] if len(sys.argv) > 1 else "data/sonate2_frames.csv"
SNRS = [14, 12, 11, 10, 9, 8, 7, 6]
N_FRAMES = 40
FS = 48000
os.makedirs("results", exist_ok=True)

# ---- load real frames (SatNOGS stores them WITHOUT the 2-byte FCS -> add it back)
df = pd.read_csv(CSV).dropna(subset=["frame"])
df = df[df.frame.str.startswith("86A240404040E088A060A69CB061")]   # CQ <- DP0SNX beacon
payloads = [bytes.fromhex(h) for h in df.frame.drop_duplicates()]
print(f"{len(payloads)} unique real SONATE-2 frames loaded")

def with_fcs(p):
    c = crc_x25(p)
    return p + bytes([c & 0xFF, c >> 8])

rng = np.random.default_rng(42)
rows = []
for snr in SNRS:
    pick = rng.choice(len(payloads), N_FRAMES, replace=False)
    frames = [with_fcs(payloads[i]) for i in pick]
    truth = {f.hex().upper() for f in frames}
    chunks = []
    for f in frames:
        chunks += [np.zeros(int(0.2 * FS), np.float32), afsk_modulate(frame_to_bits(f), FS)]
    sig = np.concatenate(chunks + [np.zeros(int(0.2 * FS), np.float32)])
    noise_p = 0.5 / 10 ** (snr / 10) * (FS / 2 / 1800)          # SNR in ~1.8 kHz band
    audio = sig + rng.normal(0, np.sqrt(noise_p), len(sig)).astype(np.float32)

    hard, tones = decode_audio(audio, FS)
    soft = decode_frames(tones, rescue=True, max_flips=3)
    h = len({f["hex"] for f in hard} & truth)
    s = len({f["hex"] for f in soft} & truth)
    false = len({f["hex"] for f in soft} - truth)
    rows.append(dict(snr_db=snr, sent=N_FRAMES, hard=h, rescue=s, false_accepts=false,
                     hard_pct=100 * h / N_FRAMES, rescue_pct=100 * s / N_FRAMES))
    print(f"SNR {snr:3} dB | hard {h:3}/{N_FRAMES} | rescue {s:3}/{N_FRAMES} | false {false}")

r = pd.DataFrame(rows)
r.to_csv("results/sweep_real.csv", index=False)

plt.figure(figsize=(7, 4.5))
plt.plot(r.snr_db, r.hard_pct, "o-", label="Standard decoder (hard decision)")
plt.plot(r.snr_db, r.rescue_pct, "s-", label="GHOST FRAMES (soft-decision Chase rescue)")
plt.fill_between(r.snr_db, r.hard_pct, r.rescue_pct, alpha=0.15)
plt.xlabel("In-band SNR (dB, simulated channel)")
plt.ylabel("Packets recovered (%)")
plt.title("Real SONATE-2 packets through a noisy AFSK 1200 channel")
plt.grid(alpha=0.3); plt.legend(); plt.gca().invert_xaxis()
plt.tight_layout(); plt.savefig("results/sweep_real.png", dpi=150)
print("Saved results/sweep_real.png and results/sweep_real.csv")
