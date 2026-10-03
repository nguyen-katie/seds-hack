"""
Symbol-rate estimation on REAL SONATE-2 SatNOGS audio (approximate).

Why the first attempt failed: in SatNOGS audio the 9600-baud GMSK data is buried in
FM-discriminator noise (which grows with frequency), and we averaged over whole
noisy recordings. Fixes here:
  1. use only the packet bursts found by the FM-quieting detector (where the carrier is)
  2. low-pass to the data band (< 7 kHz) where the signal beats the noise
  3. average the cyclic spectrum of (dx/dt)^2 over MANY short chunks (noise averages out)
  4. score each candidate rate by how much it stands out from its neighbourhood
Reports (a) the free blind estimate and (b) the best of the standard amateur rates.

Run from repo root:  python3 src/real_baud.py
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt
from validate_bursts import detect_quieting

OBS = ["15052729", "15039241", "15104225"]
STANDARD = [1200, 2400, 4800, 9600, 19200]
NFFT = 4096          # 85 ms: shorter than one SONATE-2 packet (~0.13-0.25 s)


def cyclic_spectrum(audio, fs, bursts, cutoff=7000):
    sos = butter(6, cutoff, fs=fs, output="sos")
    acc, n = np.zeros(NFFT // 2 + 1), 0
    win = np.hanning(NFFT)
    for a, b in bursts:
        x = sosfiltfilt(sos, audio[a:b].astype(np.float64))
        d2 = np.diff(x) ** 2
        if len(d2) < NFFT:
            d2 = np.pad(d2, (0, NFFT - len(d2)))
        for k in range(0, len(d2) - NFFT + 1, NFFT // 2):
            seg = d2[k:k + NFFT]
            seg = seg - seg.mean()
            acc += np.abs(np.fft.rfft(seg * win)) ** 2
            n += 1
    return np.fft.rfftfreq(NFFT, 1 / fs), acc / max(n, 1), n


def line_score(f, P, f0, rel=0.08, guard=3):
    i = int(np.argmin(np.abs(f - f0)))
    lo, hi = max(0, int(i * (1 - rel))), min(len(P), int(i * (1 + rel)) + 1)
    neigh = np.r_[P[lo:max(lo, i - guard)], P[min(hi, i + guard + 1):hi]]
    j = i - 2 + int(np.argmax(P[max(0, i - 2):i + 3]))      # allow +-2 bins of drift
    return P[j] / (np.median(neigh) + 1e-20), f[j]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(OBS), 1, figsize=(9, 7), sharex=True)
    for ax, obs in zip(axes, OBS):
        audio, fs = sf.read(f"data/satnogs/sonate2_{obs}/audio.ogg", dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        bursts = [b for b in detect_quieting(audio, fs) if b[1] - b[0] > NFFT // 2]
        if not bursts:
            bursts = [(0, len(audio))]
        f, P, n = cyclic_spectrum(audio, fs, bursts)
        # (a) free blind search 800-12000 Hz
        sel = np.where((f > 800) & (f < 12000))[0]
        scores = [line_score(f, P, f[i])[0] for i in sel[::2]]
        k = int(np.argmax(scores)); free = f[sel[::2][k]]
        # (b) best standard rate
        std = {r: line_score(f, P, r) for r in STANDARD if r < fs / 2}
        best = max(std, key=lambda r: std[r][0])
        print(f"obs {obs}: {len(bursts)} packet bursts, {n} chunks | free blind estimate "
              f"{free:.0f} Hz (score {max(scores):.1f}) | best standard rate {best} baud "
              f"(score {std[best][0]:.1f}; " +
              ", ".join(f"{r}:{s[0]:.1f}" for r, s in std.items()) + ")")
        m = (f > 500) & (f < 12000)
        ax.semilogy(f[m], P[m], lw=0.7)
        ax.axvline(9600, color="r", ls="--", lw=0.8, label="9600 (published)")
        ax.axvline(free, color="g", ls=":", lw=1.2, label=f"our estimate {free:.0f}")
        ax.set_title(f"obs {obs}: cyclic spectrum of packet bursts (best standard rate: {best})",
                     fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Cycle frequency (Hz)")
    fig.tight_layout(); fig.savefig("results/real_baud.png", dpi=150)
    print("Saved results/real_baud.png")