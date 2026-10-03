"""Clean the SatNOGS DB frame export for SONATE-2.

Input:  data/satnogs/sonate2_frames.csv        (raw API export, one row per station per packet)
Output: data/satnogs/sonate2_frames_clean.csv  (valid SONATE-2 packets within our pass windows)
        data/satnogs/sonate2_packets_unique.csv (one row per distinct packet: our ground truth)

Usage: python clean_frames.py
"""
import glob
import json

import pandas as pd

DATA = "data/satnogs"
SAT_CALLSIGN = "DP0SNX"


def callsign(b):
    """Decode a 7-byte AX.25 address field (each character shifted left by 1 bit)."""
    call = "".join(chr(c >> 1) for c in b[:6]).strip()
    ssid = (b[6] >> 1) & 0x0F
    return f"{call}-{ssid}" if ssid else call


def load_passes():
    """Return [(observation_id, start, end)] for every downloaded pass."""
    passes = []
    for f in sorted(glob.glob(f"{DATA}/sonate2_*/obs.json")):
        o = json.load(open(f))
        passes.append((o["id"], pd.Timestamp(o["start"]), pd.Timestamp(o["end"])))
    return passes


def add_header_columns(df):
    """Parse the AX.25 header out of each hex frame."""
    raw = df["frame"].apply(bytes.fromhex)
    ok = raw.apply(len) >= 16   # a few entries are junk, too short to hold a header
    df["dst"] = raw[ok].apply(lambda b: callsign(b[0:7]))
    df["src"] = raw[ok].apply(lambda b: callsign(b[7:14]))
    df["length"] = raw.apply(len)
    df["payload_hex"] = raw.apply(lambda b: b[16:].hex())   # after 14 address bytes + control + PID
    return df


def clean(df, passes):
    before = len(df)

    # 1. Drop empty / constant columns
    drop_cols = [c for c in df.columns if df[c].isna().mean() > 0.95 or df[c].nunique(dropna=False) == 1]
    df = df.drop(columns=drop_cols)
    print("Dropped columns:", drop_cols)

    # 2. Keep only valid SONATE-2 packets: right sender, AX.25 UI control (0x03) and PID (0xF0)
    raw = df["frame"].apply(bytes.fromhex)
    valid = (df["src"] == SAT_CALLSIGN) & raw.apply(lambda b: len(b) >= 16 and b[14] == 0x03 and b[15] == 0xF0)
    print("Invalid packets removed:", (~valid).sum())
    df = df[valid].copy()

    # 3. Keep only our pass windows, and label each row with its pass
    df["pass_id"] = pd.NA
    for obs_id, start, end in passes:
        df.loc[df["timestamp"].between(start, end), "pass_id"] = obs_id
    print("Outside pass windows removed:", df["pass_id"].isna().sum())
    df = df.dropna(subset=["pass_id"]).astype({"pass_id": "int64"})

    # 4. Remove exact duplicate rows (same packet, same station, same time)
    dupes = df.duplicated(subset=["frame", "observer", "timestamp"])
    print("Exact duplicate rows removed:", dupes.sum())
    df = df[~dupes].sort_values("timestamp").reset_index(drop=True)

    print(f"Rows: {before} -> {len(df)}")
    return df


def unique_packets(df):
    """One row per distinct packet, with how many stations heard it."""
    return (df.groupby("frame")
            .agg(pass_id=("pass_id", "first"),
                 first_seen=("timestamp", "min"),
                 length=("length", "first"),
                 n_stations=("observer", "nunique"),
                 stations=("observer", lambda s: ", ".join(sorted(set(x.rsplit("-", 1)[0] for x in s)))),
                 payload_hex=("payload_hex", "first"))
            .reset_index()
            .sort_values("first_seen", ignore_index=True))


def main():
    df = pd.read_csv(f"{DATA}/sonate2_frames.csv", parse_dates=["timestamp"],
                     dtype={"observation_id": "Int64"})
    df = add_header_columns(df)
    df = clean(df, load_passes())
    packets = unique_packets(df)

    print("Unique packets per pass:")
    print(packets.groupby("pass_id").size().to_string())

    df.to_csv(f"{DATA}/sonate2_frames_clean.csv", index=False)
    packets.to_csv(f"{DATA}/sonate2_packets_unique.csv", index=False)
    print(f"Saved {DATA}/sonate2_frames_clean.csv and {DATA}/sonate2_packets_unique.csv")


if __name__ == "__main__":
    main()
