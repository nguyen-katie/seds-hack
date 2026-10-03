"""One-page summary of the decoder's results. Saves results/review.png and opens it.

    python review_results.py          (run python run_pipeline.py first)
"""
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = "results"
TRUTH = "data/satnogs/sonate2_frames_clean.csv"
BLUE, LIGHT_BLUE, ORANGE, GRAY = "#2a6fdb", "#9dbdf2", "#e8743b", "#9aa0a6"


def tidy(ax):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def main(open_png=True):
    log = pd.read_csv(f"{RESULTS}/telemetry_log.csv")
    comp = pd.read_csv(f"{RESULTS}/comparison.csv")
    loss = pd.read_csv(f"{RESULTS}/counter_loss.csv")
    truth = pd.read_csv(TRUTH)
    labels = [f"{r.pass_id}\n{r.station.split('-')[0]}" for r in comp.itertuples()]
    xpos = np.arange(len(comp))

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.25)

    # 1. Ours vs SatNOGS on the same recording
    ax = fig.add_subplot(gs[0, 0])
    w = 0.38
    b1 = ax.bar(xpos - w / 2, comp["satnogs_same_recording"], w, color=GRAY, label="SatNOGS decoder")
    b2 = ax.bar(xpos + w / 2, comp["ours"], w, color=BLUE, label="Our decoder")
    ax.bar_label(b1, padding=3)
    ax.bar_label(b2, padding=3)
    ax.set_xticks(xpos, labels)
    ax.set_ylabel("Unique CRC-valid packets")
    ax.set_title("1. Packets decoded from the same recording", loc="left")
    ax.legend(loc="upper left")
    tidy(ax)

    # 2. How many packets needed bit repair
    ax = fig.add_subplot(gs[0, 1])
    rep = log.groupby(["pass_id", "corrected_bits"]).size().unstack(fill_value=0).reindex(comp["pass_id"])
    bottom = np.zeros(len(rep))
    for bits, color, name in [(0, BLUE, "decoded cleanly"), (1, LIGHT_BLUE, "rescued: 1 bit flipped"),
                              (2, ORANGE, "rescued: 2 bits flipped")]:
        vals = rep.get(bits, pd.Series(0, index=rep.index)).values
        ax.bar(xpos, vals, 0.55, bottom=bottom, color=color, label=name, edgecolor="white", linewidth=1)
        bottom += vals
    total_rescued = (log["corrected_bits"] > 0).sum()
    ax.set_xticks(xpos, labels)
    ax.set_ylabel("Packets")
    ax.set_title(f"2. Soft-decision rescue: {total_rescued} of {len(log)} packets "
                 f"({100 * total_rescued / len(log):.0f}%) saved by bit repair", loc="left")
    ax.set_ylim(0, bottom.max() * 1.4)
    ax.legend(loc="upper right")
    tidy(ax)

    # 3. Packet loss from the satellite's own counter
    ax = fig.add_subplot(gs[1, 0])
    got = ax.bar(xpos, loss["decoded"], 0.55, color=BLUE, label="decoded by us")
    ax.bar(xpos, loss["counter_says_sent"] - loss["decoded"], 0.55, bottom=loss["decoded"], color=GRAY,
           label="missed (gap in the satellite's counter)")
    for k, r in enumerate(loss.itertuples()):
        ax.text(k, r.counter_says_sent + 8, f"{r.estimated_loss_pct:.0f}% loss", ha="center", fontweight="bold")
    ax.set_xticks(xpos, labels)
    ax.set_ylabel("Packets the satellite sent (in our window)")
    ax.set_ylim(0, loss["counter_says_sent"].max() * 1.45)
    ax.set_title("3. Self-measured packet loss (no ground truth needed)", loc="left")
    ax.legend(loc="upper left")
    tidy(ax)

    # 4. Ghost frames table
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    same_rec = set(truth.loc[truth["observation_id"] == truth["pass_id"], "frame"])
    ghosts = log[~log["frame"].isin(same_rec)]

    def heard_by(frame):
        s = truth.loc[truth["frame"] == frame, "observer"].str.rsplit("-", n=1).str[0]
        names = sorted(set(s))
        if not names:
            return "nobody else"
        return ", ".join(names[:2]) + (f" +{len(names) - 2}" if len(names) > 2 else "")

    rows = [[str(r.pass_id), r.utc_approx[11:19], f"{r.length} B",
             str(r.corrected_bits) if r.corrected_bits else "-", heard_by(r.frame)] for r in ghosts.itertuples()]
    table = ax.table(cellText=rows, colLabels=["Pass", "UTC", "Size", "Bits fixed", "Also received by"],
                     loc="upper center", cellLoc="left", colWidths=[0.15, 0.13, 0.1, 0.14, 0.48])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_text_props(fontweight="bold")
    confirmed = sum(r[-1] != "nobody else" for r in rows)
    ax.set_title(f"4. Ghost frames: {len(ghosts)} valid packets SatNOGS missed\n"
                 f"({(ghosts['corrected_bits'] > 0).sum()} recovered by bit repair, "
                 f"{confirmed} confirmed by other stations)", loc="left")

    fig.suptitle(f"GHOST FRAMES: SONATE-2 decoder results ({len(log)} verified packets from {len(comp)} real passes)",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    out = f"{RESULTS}/review.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")

    plt.close(fig)
    if open_png:
        if sys.platform == "win32":
            os.startfile(os.path.abspath(out))
        else:
            subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", out])


if __name__ == "__main__":
    main(open_png="--no-open" not in sys.argv)
