"""
GHOST FRAMES - blind front end ("find" + "no human tuning").

Given raw audio, with NO prior knowledge:
  1. detect_bursts   : where are the packets?   (adaptive noise floor, median + k*MAD)
  2. estimate_tones  : which two FSK tones?      (spectral peaks inside bursts)
  3. estimate_baud   : what symbol rate?         (spectral line of the squared
                                                  derivative of instantaneous frequency)
Run demo:  python3 src/frontend.py      -> results/frontend.png
"""
import os
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, welch, find_peaks, spectrogram


def detect_bursts(audio, fs, win_s=0.01, k=6.0, min_len_s=0.05, merge_s=0.05,
                  band=(300, 3000)):
    """Return list of (start_sample, end_sample). Threshold = noise median + k*MAD."""
    sos = butter(4, band, btype="bandpass", fs=fs, output="sos")
    x = sosfiltfilt(sos, audio.astype(np.float64))
    w = max(1, int(win_s * fs))
    n = len(x) // w
    p = 10 * np.log10((x[:n * w].reshape(n, w) ** 2).mean(axis=1) + 1e-20)
    # noise floor from the quietest 30% of windows (packets may dominate the recording)
    q = p[p <= np.percentile(p, 30)]
    med = np.median(q)
    mad = np.median(np.abs(q - med)) + 1e-9
    on = p > med + max(k * mad, 3.0)
    bursts, start = [], None
    for i, v in enumerate(np.append(on, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            bursts.append([start, i]); start = None
    merged = []
    for b in bursts:                                   # merge close segments
        if merged and (b[0] - merged[-1][1]) * win_s < merge_s:
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    return [(a * w, b * w) for a, b in merged if (b - a) * win_s >= min_len_s]


def _burst_audio(audio, bursts):
    return np.concatenate([audio[a:b] for a, b in bursts]) if bursts else audio


def estimate_tones(audio, fs, bursts, band=(300, 3500)):
    """Two strongest spectral peaks inside bursts -> (low_tone, high_tone) in Hz."""
    x = _burst_audio(audio, bursts)
    f, P = welch(x, fs, nperseg=4096)
    sel = (f >= band[0]) & (f <= band[1])
    f, P = f[sel], P[sel]
    pk, _ = find_peaks(P, distance=max(1, int(300 / (f[1] - f[0]))))
    top = pk[np.argsort(P[pk])[-2:]]
    lo, hi = sorted(f[top])
    return float(lo), float(hi)


def estimate_baud(audio, fs, bursts, tones, fmax=12000):
    """Symbol rate from the spectral line of (d/dt inst. frequency)^2."""
    lo, hi = tones
    sos = butter(4, [max(50, lo - 600), hi + 600], btype="bandpass", fs=fs, output="sos")
    lines = []
    for a, b in bursts:
        seg = sosfiltfilt(sos, audio[a:b].astype(np.float64))
        if len(seg) < fs * 0.05:
            continue
        inst_f = np.diff(np.unwrap(np.angle(hilbert(seg)))) * fs / (2 * np.pi)
        sos_lp = butter(4, (lo + hi) / 2, fs=fs, output="sos")   # smooth inst. freq
        inst_f = sosfiltfilt(sos_lp, inst_f)
        d2 = np.diff(inst_f) ** 2
        d2 -= d2.mean()
        spec = np.abs(np.fft.rfft(d2 * np.hanning(len(d2)), n=1 << 18))
        freqs = np.fft.rfftfreq(1 << 18, 1 / fs)
        sel = (freqs > 200) & (freqs < fmax)
        lines.append(freqs[sel][np.argmax(spec[sel])])
    return float(np.median(lines)) if lines else float("nan")


def analyze(audio, fs):
    bursts = detect_bursts(audio, fs)
    tones = estimate_tones(audio, fs, bursts)
    baud = estimate_baud(audio, fs, bursts, tones)
    return dict(bursts=bursts, tones=tones, baud=baud)


def plot_frontend(audio, fs, result, path="results/frontend.png", title=""):
    import matplotlib.pyplot as plt
    f, t, S = spectrogram(audio, fs, nperseg=1024, noverlap=768)
    sel = f <= 4000
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.pcolormesh(t, f[sel], 10 * np.log10(S[sel] + 1e-12), shading="auto", cmap="viridis")
    for a, b in result["bursts"]:
        ax.axvspan(a / fs, b / fs, color="red", alpha=0.18)
    for tone in result["tones"]:
        ax.axhline(tone, color="white", ls="--", lw=0.8)
    lo, hi = result["tones"]
    ax.set_title(title or f"Blind front end: {len(result['bursts'])} bursts | tones "
                 f"{lo:.0f}/{hi:.0f} Hz | baud {result['baud']:.0f}  (no human tuning)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Audio frequency (Hz)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    from afsk import synth_pass
    print("SNR | bursts found/true | tones (Hz)      | baud est")
    for snr in [20, 12, 9]:
        audio, fs, truth = synth_pass(n_frames=12, snr_db=snr, seed=snr)
        r = analyze(audio, fs)
        print(f"{snr:3} | {len(r['bursts']):5}/{len(truth):<5}      | "
              f"{r['tones'][0]:6.0f} / {r['tones'][1]:6.0f} | {r['baud']:7.1f}")
        if snr == 12:
            plot_frontend(audio, fs, r)
    print("Saved results/frontend.png")


# =====================================================================
# v2: modulation-family-aware estimation (AFSK tones vs. direct FSK/GMSK)
# In FM audio, AFSK shows up as two audio tones; 9600-baud GMSK shows up as
# the baseband data waveform itself. We test both hypotheses and keep the one
# whose symbol-rate spectral line is most prominent -> fully blind.
# =====================================================================
NFFT = 1 << 16


def _line_spectrum(signals, fs):
    acc = None
    for d2 in signals:
        d2 = d2 - d2.mean()
        if len(d2) < 256:
            continue
        seg = d2[:NFFT] if len(d2) >= NFFT else np.pad(d2, (0, NFFT - len(d2)))
        P = np.abs(np.fft.rfft(seg * np.hanning(NFFT))) ** 2
        acc = P if acc is None else acc + P
    return np.fft.rfftfreq(NFFT, 1 / fs), acc


def _best_line(freqs, P, fmin=300, fmax=12000):
    if P is None:
        return float("nan"), 0.0
    sel = (freqs > fmin) & (freqs < fmax)
    f, p = freqs[sel], P[sel]
    # a symbol-rate LINE is narrow: score each bin against its local neighbourhood
    # (median of +-15% in frequency, excluding the bin's own +-3 bins), not the global median
    from scipy.ndimage import median_filter
    df = f[1] - f[0]
    win = np.maximum(7, (0.15 * f / df).astype(int))
    w = int(np.median(win)) | 1
    local = median_filter(p, size=w, mode="nearest") + 1e-20
    score = p / local
    i = int(np.argmax(score))
    return float(f[i]), float(score[i])                        # line freq, prominence


def _longest(bursts, n=40):
    return sorted(bursts, key=lambda b: b[1] - b[0], reverse=True)[:n]


def baud_hypothesis_baseband(audio, fs, bursts):
    """Direct FSK/GMSK: data waveform is in the audio. Line of (dx/dt)^2."""
    sos = butter(4, min(0.45 * fs, 10000), fs=fs, output="sos")
    sigs = [np.diff(sosfiltfilt(sos, audio[a:b].astype(np.float64))) ** 2
            for a, b in _longest(bursts)]
    return _best_line(*_line_spectrum(sigs, fs))


def baud_hypothesis_afsk(audio, fs, bursts):
    """AFSK: data is in the instantaneous frequency of audio tones."""
    tones = estimate_tones(audio, fs, bursts)
    lo, hi = tones
    sos = butter(4, [max(50, lo - 600), hi + 600], btype="bandpass", fs=fs, output="sos")
    sos_lp = butter(4, (lo + hi) / 2, fs=fs, output="sos")
    sigs = []
    for a, b in _longest(bursts):
        seg = sosfiltfilt(sos, audio[a:b].astype(np.float64))
        if len(seg) < 512:
            continue
        inst = np.diff(np.unwrap(np.angle(hilbert(seg)))) * fs / (2 * np.pi)
        sigs.append(np.diff(sosfiltfilt(sos_lp, inst)) ** 2)
    f, prom = _best_line(*_line_spectrum(sigs, fs))
    return f, prom, tones


def analyze_v2(audio, fs):
    bursts = detect_bursts(audio, fs, band=(300, 8000), merge_s=0.2, min_len_s=0.05)
    if not bursts:                                  # continuous signal: use whole file
        bursts = [(0, len(audio))]
    fb, pb = baud_hypothesis_baseband(audio, fs, bursts)
    fa, pa, tones = baud_hypothesis_afsk(audio, fs, bursts)
    if pa > pb:
        return dict(bursts=bursts, family="AFSK (audio tones)", baud=fa, tones=tones,
                    prominence=pa, alt=(fb, pb))
    return dict(bursts=bursts, family="direct FSK/GMSK (baseband)", baud=fb, tones=None,
                prominence=pb, alt=(fa, pa))


def plot_frontend_v2(audio, fs, r, path, title=""):
    import matplotlib.pyplot as plt
    f, t, S = spectrogram(audio, fs, nperseg=1024, noverlap=768)
    sel = f <= 12000
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.pcolormesh(t, f[sel], 10 * np.log10(S[sel] + 1e-12), shading="auto", cmap="viridis")
    for a, b in r["bursts"]:
        ax.axvspan(a / fs, b / fs, color="red", alpha=0.15)
    ax.set_title(title or f"Blind front end: {len(r['bursts'])} bursts | {r['family']} | "
                 f"baud {r['baud']:.0f}")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Audio frequency (Hz)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)


def synth_gmsk_audio(n_frames=10, snr_db=12, fs=48000, baud=9600, seed=1):
    """FM-discriminator output for GMSK = Gaussian-filtered NRZ (BT=0.5) + noise."""
    rng = np.random.default_rng(seed)
    sps = fs / baud
    out = []
    for _ in range(n_frames):
        out.append(np.zeros(int(0.4 * fs)))
        bits = rng.integers(0, 2, 2000) * 2 - 1
        n = int(len(bits) * sps)
        x = bits[np.minimum((np.arange(n) / sps).astype(int), len(bits) - 1)].astype(float)
        sigma = sps * np.sqrt(np.log(2)) / (2 * np.pi * 0.5)
        k = np.arange(-int(4 * sigma), int(4 * sigma) + 1)
        g = np.exp(-k ** 2 / (2 * sigma ** 2)); g /= g.sum()
        out.append(np.convolve(x, g, mode="same") * 0.5)
    sig = np.concatenate(out + [np.zeros(int(0.4 * fs))])
    noise_p = np.mean(np.concatenate(out[1::2]) ** 2) / 10 ** (snr_db / 10)
    return (sig + rng.normal(0, np.sqrt(noise_p), len(sig))).astype(np.float32), fs