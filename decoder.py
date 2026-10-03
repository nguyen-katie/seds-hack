"""Autonomous 9600-baud G3RUH / AX.25 decoder for SatNOGS pass audio.

Pipeline: filter -> symbol timing -> slice -> G3RUH descramble -> NRZI decode
-> HDLC frame sync + bit unstuffing -> CRC-16 check -> AX.25 parse.
"""
import numpy as np
import soundfile as sf
from scipy import ndimage, signal

BAUD = 9600


# ---------- 1. Filter ----------

def load_audio(path):
    x, fs = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, fs


def clean(x, fs):
    """Remove slow DC drift (residual Doppler after FM demod).

    Scrambled 9600-baud data has real energy down to a few Hz, so a conventional
    high-pass or tight low-pass filter distorts the bits. Subtracting a 0.2 s moving
    average only removes drift slower than ~5 Hz. (The recording is already
    band-limited to ~11 kHz by the ground station.)
    """
    return x - ndimage.uniform_filter1d(x, int(0.2 * fs))


# ---------- 2-3. Symbol timing + slicing ----------

def sample_symbols(y, fs, phase):
    """Sample one value per symbol starting at `phase` (in samples)."""
    sps = fs / BAUD
    idx = np.arange(phase, len(y) - 1, sps)
    i0 = idx.astype(int)
    frac = idx - i0
    return y[i0] * (1 - frac) + y[i0 + 1] * frac, idx   # linear interpolation


# ---------- 4. G3RUH descramble + NRZI ----------

def descramble(b):
    """G3RUH self-synchronising descrambler, polynomial 1 + x^12 + x^17."""
    out = b.copy()
    out[17:] ^= b[5:-12] ^ b[:-17]
    out[:17] = 0
    return out


def nrzi_decode(b):
    """No transition -> 1, transition -> 0."""
    return np.concatenate([[0], (b[1:] == b[:-1]).astype(np.uint8)])


# ---------- 5. HDLC framing ----------

FLAG = "01111110"


def crc16_x25(data):
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
    return crc ^ 0xFFFF


def unstuff(bits):
    """Remove the 0 inserted after every run of five 1s. Returns None on an abort (7+ ones)."""
    out, ones = [], 0
    i = 0
    while i < len(bits):
        b = bits[i]
        if b == "1":
            ones += 1
            if ones > 6:
                return None
            out.append(b)
        else:
            if ones != 5:
                out.append(b)
            ones = 0
        i += 1
    return "".join(out)


def bits_to_bytes(bits):
    """Bits are sent LSB-first."""
    n = len(bits) // 8
    return bytes(int(bits[8 * k:8 * k + 8][::-1], 2) for k in range(n))


def check_frame(body):
    """Unstuff an HDLC body (bit string) and return the frame without FCS if the CRC passes."""
    raw = unstuff(body)
    if raw is None or len(raw) % 8 or len(raw) < 18 * 8:
        return None
    frame = bits_to_bytes(raw)
    fcs = frame[-2] | (frame[-1] << 8)
    if crc16_x25(frame[:-2]) != fcs:
        return None
    return frame[:-2] if valid_ax25_header(frame) else None


CALLSIGN_CHARS = set(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")


def valid_ax25_header(frame):
    """Reject CRC 'passes' that are really noise: both callsigns must be legal AX.25 characters.

    A 16-bit CRC passes random data 1 time in 65,536; bit-flip repair makes many tries,
    so this second check keeps the false-accept rate near zero.
    """
    if len(frame) < 16:
        return False
    return all((c >> 1) in CALLSIGN_CHARS and not (c & 1) for c in frame[0:6] + frame[7:13])


def decode_bits(raw):
    return nrzi_decode(descramble(raw))


def to_str(bits):
    return "".join("1" if b else "0" for b in bits)


def hdlc_frames(soft, max_flips=2, n_weak=8):
    """Yield (frame, start_symbol, n_corrected) for every CRC-valid frame between flags.

    `soft` holds the sampled symbol values; their magnitude is the slicer's confidence.
    A frame that fails its CRC is retried with its least-confident symbols flipped.
    """
    raw = (soft > 0).astype(np.uint8)
    s = to_str(decode_bits(raw))
    starts = []
    i = s.find(FLAG)
    while i != -1:
        starts.append(i)
        i = s.find(FLAG, i + 1)
    for a, b in zip(starts, starts[1:]):
        body = s[a + 8:b]
        if len(body) < 18 * 8 or len(body) > 400 * 8:   # AX.25 min header+FCS .. sane max
            continue
        frame = check_frame(body)
        if frame is not None:
            yield frame, a, 0
            continue
        if not max_flips:
            continue
        # Soft-decision repair: flip the weakest symbols (singly, then in pairs).
        lo, hi = a - 17, b + 8   # descrambler memory reaches 17 symbols back
        if lo < 0:
            continue
        weak = lo + 17 + np.argsort(np.abs(soft[lo + 17:hi]))[:n_weak]
        tries = [(w,) for w in weak]
        if max_flips >= 2:
            tries += [(w1, w2) for k, w1 in enumerate(weak) for w2 in weak[k + 1:]]
        for flips in tries:
            seg = raw[lo:hi].copy()
            for w in flips:
                seg[w - lo] ^= 1
            ss = to_str(decode_bits(seg))[17:]
            if ss[:8] != FLAG or ss[-8:] != FLAG:
                continue
            frame = check_frame(ss[8:-8])
            if frame is not None:
                yield frame, a, len(flips)
                break


# ---------- 6. AX.25 parse ----------

def ax25_addr(b):
    call = "".join(chr(c >> 1) for c in b[:6]).strip()
    ssid = (b[6] >> 1) & 0x0F
    return f"{call}-{ssid}" if ssid else call


def parse_ax25(frame):
    dst, src = ax25_addr(frame[0:7]), ax25_addr(frame[7:14])
    i = 14
    while not (frame[i - 1] & 1) and i + 7 <= len(frame):   # skip digipeater addresses
        i += 7
    payload = frame[i + 2:]   # skip control + PID
    return {"dst": dst, "src": src, "payload": payload}


# ---------- Full pipeline ----------

def decode(path, phases=10, max_flips=2):
    """Return a list of dicts, one per unique CRC-valid frame, sorted by time."""
    x, fs = load_audio(path)
    return decode_audio(x, fs, phases, max_flips)


def decode_audio(x, fs, phases=10, max_flips=2):
    """Same as decode(), for audio already in memory."""
    y = clean(x, fs)
    sps = fs / BAUD
    found = {}
    # Timing recovery by search: try sub-symbol phases, keep any CRC-valid frame.
    for phase in np.arange(0, sps, sps / phases):
        soft, idx = sample_symbols(y, fs, phase)
        for frame, pos, nfix in hdlc_frames(soft, max_flips):
            t = float(idx[pos] / fs)
            if frame not in found or nfix < found[frame]["corrected_bits"]:
                found[frame] = {"t": round(t, 3), "frame": frame, "corrected_bits": nfix}
    return sorted(found.values(), key=lambda r: r["t"])


def plot_decode(x, fs, frames, path, open_png=True):
    """Spectrogram of the recording with every decoded packet marked. Saves and opens a PNG."""
    import os
    import subprocess
    import sys
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    f, t, S = signal.spectrogram(x - x.mean(), fs, nperseg=1024, noverlap=512)
    S = 10 * np.log10(S + 1e-12)
    a1.pcolormesh(t, f / 1000, S, shading="auto", cmap="viridis", vmin=np.percentile(S, 5), vmax=np.percentile(S, 99.5))
    a1.set_ylim(0, 12)
    a1.set_ylabel("Frequency (kHz)")
    a1.set_title("Recording: vertical stripes are packet bursts", loc="left")

    colors = {0: "#2a6fdb", 1: "#9dbdf2", 2: "#e8743b"}
    names = {0: "decoded cleanly", 1: "rescued: 1 bit flipped", 2: "rescued: 2 bits flipped"}
    for bits in (0, 1, 2):
        ts = [r["t"] for r in frames if r["corrected_bits"] == bits]
        a2.plot(ts, np.zeros(len(ts)), "|", ms=40, mew=2, color=colors[bits], label=f"{names[bits]} ({len(ts)})")
    a2.set_yticks([])
    a2.set_xlim(0, len(x) / fs)
    a2.set_xlabel("Time into recording (s)")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=3, frameon=False)
    for side in ["top", "right", "left"]:
        a2.spines[side].set_visible(False)
    fixed = sum(r["corrected_bits"] > 0 for r in frames)
    a2.set_title(f"Decoded packets: {len(frames)} CRC-valid ({fixed} rescued by bit repair)", loc="left")

    name = os.path.basename(os.path.dirname(os.path.abspath(path))) or "audio"
    fig.suptitle(f"Decoder output: {name}", x=0.01, ha="left", fontsize=13)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    out = os.path.abspath(f"results/decode_{name}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved {out}")
    if open_png:
        if sys.platform == "win32":
            os.startfile(out)
        else:
            subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", out])


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    x, fs = load_audio(path)
    frames = decode_audio(x, fs)
    fixed = sum(r["corrected_bits"] > 0 for r in frames)
    print(f"{len(frames)} CRC-valid frames ({fixed} repaired by soft-decision)")
    for r in frames[:10]:
        p = parse_ax25(r["frame"])
        print(f"{r['t']:8.2f}s  {p['src']:>10} -> {p['dst']:<10} {len(r['frame']):4d} B  {p['payload'][:32].hex()}")
    if "--no-plot" not in sys.argv:
        plot_decode(x, fs, frames, path)
