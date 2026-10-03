"""
IDENTIFY stage on the Track 1 benchmark: blind modulation classification on RadioML 2016.10A.
Expert signal features (higher-order cumulants, amplitude/phase/frequency statistics,
spectral lines of x^2 and x^4) + Random Forest. Trains in ~1-2 min on a laptop CPU.

Run from repo root:  python3 src/radioml_classify.py RML2016.10a_dict.pkl
Outputs: results/radioml_accuracy.png, results/radioml_confusion.png, results/radioml_accuracy.csv
"""
import os, sys, pickle, time
import numpy as np
import pandas as pd

PKL = sys.argv[1] if len(sys.argv) > 1 else "RML2016.10a_dict.pkl"
PER_KEY = 400            # examples per (modulation, SNR) used; 1000 available
os.makedirs("results", exist_ok=True)


def features(X):
    """X: (N, 2, 128) I/Q -> (N, F) feature matrix."""
    x = X[:, 0, :] + 1j * X[:, 1, :]
    x = x / (np.sqrt(np.mean(np.abs(x) ** 2, axis=1, keepdims=True)) + 1e-12)
    a = np.abs(x)
    ph = np.unwrap(np.angle(x), axis=1)
    f_inst = np.diff(ph, axis=1)
    ac = a / (a.mean(axis=1, keepdims=True) + 1e-12) - 1

    def m(p, q):  # moment E[x^(p-q) conj(x)^q]
        return np.mean(x ** (p - q) * np.conj(x) ** q, axis=1)
    M20, M21, M40, M41, M42 = m(2, 0), m(2, 1), m(4, 0), m(4, 1), m(4, 2)
    C40 = M40 - 3 * M20 ** 2
    C41 = M41 - 3 * M20 * M21
    C42 = M42 - np.abs(M20) ** 2 - 2 * M21 ** 2
    M60, M63 = m(6, 0), m(6, 3)
    C63 = M63 - 9 * C42 * M21 - 6 * M21 ** 3

    def line(y):
        S = np.abs(np.fft.fft(y, axis=1)) ** 2
        return S.max(axis=1) / (S.mean(axis=1) + 1e-12)

    def kurt(v):
        v = v - v.mean(axis=1, keepdims=True)
        return np.mean(v ** 4, axis=1) / (np.mean(v ** 2, axis=1) ** 2 + 1e-12)

    F = np.column_stack([
        np.abs(M20), np.abs(M40), np.abs(M60),
        np.abs(C40), np.abs(C41), np.abs(C42), np.abs(C63) ** (1 / 3),
        a.std(axis=1), kurt(a), np.mean(np.abs(ac), axis=1), ac.max(axis=1),
        f_inst.std(axis=1), np.mean(np.abs(f_inst), axis=1), kurt(f_inst),
        np.abs(np.diff(f_inst, axis=1)).mean(axis=1),
        line(x), line(x ** 2), line(x ** 4), line(a),
        np.abs(np.mean(x, axis=1)),
        np.std(X[:, 0, :], axis=1) / (np.std(X[:, 1, :], axis=1) + 1e-12),  # SSB/AM asymmetry
    ])
    return np.nan_to_num(F)


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

    t0 = time.time()
    with open(PKL, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    mods = sorted({k[0] for k in data})
    snrs = sorted({k[1] for k in data})
    print(f"Loaded {len(data)} keys: {len(mods)} modulations x {len(snrs)} SNRs ({time.time()-t0:.0f}s)")

    rng = np.random.default_rng(0)
    Xtr, ytr, Xte, yte, ste = [], [], [], [], []
    for (mod, snr), arr in data.items():
        idx = rng.permutation(len(arr))[:PER_KEY]
        half = len(idx) // 2
        Ftr, Fte = features(arr[idx[:half]]), features(arr[idx[half:]])
        Xtr.append(Ftr); ytr += [mod] * len(Ftr)
        Xte.append(Fte); yte += [mod] * len(Fte); ste += [snr] * len(Fte)
    Xtr, Xte = np.vstack(Xtr), np.vstack(Xte)
    ytr, yte, ste = np.array(ytr), np.array(yte), np.array(ste)
    print(f"Features: train {Xtr.shape}, test {Xte.shape} ({time.time()-t0:.0f}s)")

    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    print(f"Trained ({time.time()-t0:.0f}s)")

    rows = [dict(snr_db=s, accuracy=float(np.mean(pred[ste == s] == yte[ste == s]))) for s in snrs]
    r = pd.DataFrame(rows); r.to_csv("results/radioml_accuracy.csv", index=False)
    for _, row in r.iterrows():
        print(f"SNR {int(row.snr_db):4} dB: {row.accuracy:.1%}")
    hi = ste >= 10
    print(f"Mean accuracy at SNR >= 10 dB: {np.mean(pred[hi] == yte[hi]):.1%}")

    plt.figure(figsize=(7, 4.5))
    plt.plot(r.snr_db, 100 * r.accuracy, "o-")
    plt.axhline(100 / len(mods), ls="--", color="gray", label=f"chance ({100/len(mods):.0f}%)")
    plt.xlabel("SNR (dB)"); plt.ylabel("Classification accuracy (%)")
    plt.title("Blind modulation classification, RadioML 2016.10A (11 classes)")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig("results/radioml_accuracy.png", dpi=150); plt.close()

    cm = confusion_matrix(yte[hi], pred[hi], labels=mods, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(mods))); ax.set_xticklabels(mods, rotation=45, ha="right")
    ax.set_yticks(range(len(mods))); ax.set_yticklabels(mods)
    for i in range(len(mods)):
        for j in range(len(mods)):
            if cm[i, j] >= 0.05:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm[i, j] > 0.5 else "black", fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix, SNR >= 10 dB")
    fig.tight_layout(); fig.savefig("results/radioml_confusion.png", dpi=150); plt.close(fig)
    print(f"Saved results/radioml_*.png ({time.time()-t0:.0f}s total)")