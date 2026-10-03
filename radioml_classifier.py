"""RadioML 2016.10A modulation classifier, and a synthetic-to-real transfer test on SONATE-2.

    python radioml_classifier.py train      # train the CNN, save model + accuracy plots
    python radioml_classifier.py transfer   # classify real SONATE-2 signal windows with it

Training data: data/RML2016.10a_dict.pkl (11 modulations, SNR -20..18 dB, 2x128 I/Q per example).
"""
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import signal
from torch import nn

import decoder

RML_PATH = "data/RML2016.10a_dict.pkl"
MODEL_PATH = "results/radioml_cnn.pt"
RESULTS = "results"
SPS_RML = 8   # RadioML 2016.10A uses 8 samples per symbol
BLUE, ORANGE, GRAY = "#2a6fdb", "#e8743b", "#9aa0a6"

torch.manual_seed(0)
np.random.seed(0)


# ---------- Data ----------

def normalize(x):
    """Scale each example to unit RMS power, so the model can't cheat on signal level."""
    rms = np.sqrt((x ** 2).sum(axis=1, keepdims=True).mean(axis=2, keepdims=True))
    return (x / (rms + 1e-12)).astype(np.float32)


def load_radioml():
    with open(RML_PATH, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    mods = sorted({k[0] for k in data})
    snrs = sorted({k[1] for k in data})
    X = np.concatenate([data[(m, s)] for m in mods for s in snrs])
    y = np.repeat([mods.index(m) for m in mods for s in snrs], 1000)
    snr = np.repeat([s for m in mods for s in snrs], 1000)
    return normalize(X), y, snr, mods


# ---------- Model ----------

class CNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def predict(model, X, batch=2048):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(torch.softmax(model(torch.from_numpy(X[i:i + batch])), 1).numpy())
    return np.concatenate(out)


def train(epochs=10):
    X, y, snr, mods = load_radioml()
    idx = np.random.permutation(len(X))
    n_train = len(X) // 2   # standard 50/50 split for this dataset
    tr, te = idx[:n_train], idx[n_train:]

    model = CNN(len(mods))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(X[tr]), torch.from_numpy(y[tr])
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        total = 0.0
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            loss = lossf(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        sched.step()
        acc = (predict(model, X[te]).argmax(1) == y[te]).mean()
        print(f"epoch {ep + 1}/{epochs}  loss {total / len(perm):.3f}  test acc {acc:.3f}", flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    torch.save({"state": model.state_dict(), "mods": mods}, MODEL_PATH)
    pred = predict(model, X[te]).argmax(1)
    plot_accuracy(pred, y[te], snr[te], mods)


def plot_accuracy(pred, y, snr, mods):
    snrs = sorted(set(snr))
    acc = [(pred[snr == s] == y[snr == s]).mean() for s in snrs]
    pd.DataFrame({"snr_db": snrs, "accuracy": acc}).to_csv(f"{RESULTS}/radioml_accuracy.csv", index=False)
    print("overall test accuracy %.3f, at SNR >= 0 dB: %.3f" % ((pred == y).mean(), (pred[snr >= 0] == y[snr >= 0]).mean()))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.1]})
    a1.plot(snrs, acc, color=BLUE, lw=2, marker="o")
    a1.axhline(1 / len(mods), color=GRAY, ls="--", lw=1.5, label="random guess (1/11)")
    a1.set_xlabel("SNR (dB)")
    a1.set_ylabel("Test accuracy")
    a1.set_ylim(0, 1)
    a1.set_title("Modulation classification accuracy vs SNR", loc="left")
    a1.grid(alpha=0.2)
    a1.legend()

    hi = snr >= 0
    cm = np.zeros((len(mods), len(mods)))
    for t, p in zip(y[hi], pred[hi]):
        cm[t, p] += 1
    cm /= cm.sum(1, keepdims=True)
    a2.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    a2.set_xticks(range(len(mods)), mods, rotation=45, ha="right")
    a2.set_yticks(range(len(mods)), mods)
    for i in range(len(mods)):
        for j in range(len(mods)):
            if cm[i, j] >= 0.05:
                a2.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 0.5 else "black")
    a2.set_xlabel("Predicted")
    a2.set_ylabel("True")
    a2.set_title("Confusion matrix, SNR ≥ 0 dB", loc="left")
    plt.tight_layout()
    fig.savefig(f"{RESULTS}/radioml_accuracy.png", dpi=120)
    plt.close(fig)
    print(f"saved {RESULTS}/radioml_accuracy.png")


# ---------- Synthetic-to-real transfer ----------

def audio_to_iq(x, fs, baud=decoder.BAUD):
    """Rebuild complex baseband I/Q from a short stretch of FM-demodulated audio.

    The audio is the signal's instantaneous frequency. GMSK has constant amplitude and a
    phase change of +/- pi/2 per symbol (modulation index 0.5), so integrating the scaled
    frequency gives the phase, and exp(j*phase) gives I/Q. The frequency scale is estimated
    from this stretch of audio itself (FM noise is louder than the signal, so a global
    estimate would be wrong). Then resample to RadioML's 8 samples per symbol.
    """
    x = x - x.mean()
    sps = fs / baud
    level = np.percentile(np.abs(x), 90)   # ~ peak deviation within this stretch
    phase = np.cumsum(x / level) * (np.pi / 2) / sps
    up, down = int(SPS_RML * baud), int(fs)   # 48 kHz at 5 sps -> 76.8 kHz at 8 sps
    g = np.gcd(up, down)
    return signal.resample_poly(np.exp(1j * phase), up // g, down // g)


def windows(x, fs, centers_s, n=128):
    """Cut n-sample RadioML-style I/Q windows centred on the given times."""
    half = int(0.6 * n * fs / (SPS_RML * decoder.BAUD))   # audio samples covering the window + margin
    out = []
    for t in centers_s:
        i = int(t * fs)
        if i - half < 0 or i + half > len(x):
            continue
        iq = audio_to_iq(x[i - half:i + half], fs)
        m = len(iq) // 2
        w = iq[m - n // 2:m + n // 2]
        out.append(np.stack([w.real, w.imag]))
    return normalize(np.array(out)) if out else np.zeros((0, 2, n), np.float32)


def transfer():
    import glob
    import json
    ckpt = torch.load(MODEL_PATH)
    mods = ckpt["mods"]
    model = CNN(len(mods))
    model.load_state_dict(ckpt["state"])

    rows = []
    sig_counts, noise_counts = np.zeros(len(mods)), np.zeros(len(mods))
    for d in sorted(glob.glob("data/satnogs/sonate2_*/")):
        o = json.load(open(os.path.join(d, "obs.json")))
        tlog = pd.read_csv(f"{RESULTS}/telemetry_{o['id']}.csv")
        x, fs = decoder.load_audio(os.path.join(d, "audio.ogg"))
        x = decoder.clean(x, fs)
        # Signal windows: inside decoded packets (we know a real signal is there)
        centers = [t + k * 0.01 for t in tlog["audio_time_s"] for k in range(1, 6)]
        Xs = windows(x, fs, centers)
        # Noise windows: the first 20 s of the pass, before the satellite is audible
        Xn = windows(x, fs, np.arange(1, 20, 0.05))
        ps, pn = predict(model, Xs).argmax(1), predict(model, Xn).argmax(1)
        sig_counts += np.bincount(ps, minlength=len(mods))
        noise_counts += np.bincount(pn, minlength=len(mods))
        fsk = np.isin(ps, [mods.index("GFSK"), mods.index("CPFSK")]).mean()
        rows.append({"pass_id": o["id"], "signal_windows": len(Xs), "predicted_fsk_family_pct": round(100 * fsk, 1),
                     "top_prediction": mods[np.bincount(ps, minlength=len(mods)).argmax()],
                     "noise_windows": len(Xn),
                     "noise_top_prediction": mods[np.bincount(pn, minlength=len(mods)).argmax()]})
    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS}/radioml_transfer.csv", index=False)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    xpos = np.arange(len(mods))
    w = 0.4
    ax.bar(xpos - w / 2, 100 * sig_counts / sig_counts.sum(), w, color=BLUE, label="SONATE-2 packet windows")
    ax.bar(xpos + w / 2, 100 * noise_counts / noise_counts.sum(), w, color=GRAY, label="noise-only windows")
    ax.set_xticks(xpos, mods)
    ax.set_ylabel("% of windows")
    ax.set_title("What the RadioML-trained model calls real SONATE-2 signals (true: GMSK, an FSK-family modulation)", loc="left")
    ax.legend()
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    fig.savefig(f"{RESULTS}/radioml_transfer.png", dpi=120)
    plt.close(fig)
    print(f"saved {RESULTS}/radioml_transfer.png")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    {"train": train, "transfer": transfer}[cmd]()
