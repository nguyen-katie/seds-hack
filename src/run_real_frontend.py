"""Blind front end on real SONATE-2 SatNOGS recordings. Run from repo root:
   python3 src/run_real_frontend.py"""
import sys; sys.path.insert(0, "src")
import soundfile as sf
from frontend import analyze_v2, plot_frontend_v2

for obs in ["15052729", "15039241", "15104225"]:
    audio, fs = sf.read(f"data/satnogs/sonate2_{obs}/audio.ogg", dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    r = analyze_v2(audio, fs)
    print(f"obs {obs}: {len(audio)/fs:.0f}s | {len(r['bursts'])} bursts | {r['family']} | "
          f"baud {r['baud']:.0f} (line strength {r['prominence']:.0f}; "
          f"other hypothesis {r['alt'][0]:.0f} Hz @ {r['alt'][1]:.1f})")
    plot_frontend_v2(audio, fs, r, f"results/frontend_real_{obs}.png",
                     title=f"SONATE-2 obs {obs}: {len(r['bursts'])} bursts | {r['family']} | "
                           f"baud {r['baud']:.0f} (blind)")