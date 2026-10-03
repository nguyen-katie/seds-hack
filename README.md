# GHOST FRAMES: recovering the satellite packets ground stations throw away

**MATLAB in Space Hackathon · Track 1: Deep-Space Communication & Signal Intelligence**

When a satellite pass gets weak, ground stations lose packets: a couple of flipped bits break the checksum and the frame is discarded. We built a **zero-tuning receiver** that runs on real SatNOGS recordings of the **SONATE-2** CubeSat, finds the signal on its own, decodes it to verified telemetry, and **repairs damaged frames** using the receiver's own bit-confidence.

**Headline results**
- **780 checksum-verified telemetry packets** decoded from 3 real passes, with no settings changed between passes or ground stations.
- **10 ghost frames**: valid packets that SatNOGS's own decoder missed on the same recording. 8 were recovered by our bit-repair step; 7 were independently confirmed by other ground stations.
- **The receiver measures its own packet loss** from the satellite's packet counter, no ground truth needed (3% on the strongest pass).
- **Re-acquires within 0.4 s** after every injected signal dropout, and tolerates large frequency offsets.
- **Bit repair is worth about 1 dB**: near the noise limit it decodes twice as many packets as a plain decoder.
- **RadioML-trained modulation classifier** (84% accuracy at SNR ≥ 0 dB), tested on the real satellite signal.

---

## The problem

A spacecraft far from Earth can't wait for a human to tune a radio. Signals arrive weak and buried in noise. Their frequency drifts with Doppler shift as the spacecraft moves, and they drop out entirely when the link fades. A ground receiver has to handle all of that by itself.

We tested this on real data: recordings of the **SONATE-2** CubeSat captured by volunteer ground stations in the SatNOGS network. Each recording covers a full pass, from the satellite rising above the horizon to setting again. The signal starts and ends in noise and fades in and out along the way.

## Approach

SONATE-2 sends telemetry on **437.025 MHz as 9600-baud GMSK**, using the standard amateur-satellite packet format (**AX.25** framing with **G3RUH** scrambling). Our pipeline:

```
SatNOGS pass audio (48 kHz, already FM-demodulated by the ground station)
   │
   ▼
1. Drift removal   subtract a 0.2 s moving average (residual Doppler / DC drift)
2. Bit timing      sample once per bit (48 kHz / 9600 baud = 5 samples per bit),
                   searching 10 sub-bit offsets so no clock setting is needed
3. Descramble      G3RUH self-synchronising descrambler (1 + x^12 + x^17)
4. NRZI decode     no change = 1, change = 0, so signal polarity doesn't matter
5. Frame sync      find HDLC flags (01111110), remove stuffed bits
6. Verify          CRC-16 checksum on every frame; reject anything that fails
7. Repair          if the CRC fails, flip the least-confident bits (1 or 2) and re-check
8. Log             AX.25 header + payload -> telemetry log (CSV)
```

The decoder never "locks" to a single signal state: it searches for frame flags continuously, so when the signal fades out mid-pass and comes back, it **reacquires automatically** with no reset.

**Why these choices:**
- **No settings to tune.** Drift, timing and polarity are all handled from the recording itself. The same code runs unchanged on every pass and every ground station.
- **Gentle filtering.** Scrambled 9600-baud data has real content down to a few Hz, so a standard high-pass filter distorts the bits. Switching to a slow moving-average drift remover took one test minute from 15 to 24 packets (40 with repair).
- **The CRC proves each decode.** A frame that passes its checksum is correct with very high confidence, so every line in the telemetry log is verified data.
- **Soft-decision repair.** Each sampled bit keeps its confidence (how far it was from zero). Flipping only the least-confident bits recovers many packets that a single bit error would otherwise kill.
- **Independent ground truth.** We compare our decoded frames against SatNOGS's own decoder for the same recordings.

## Datasets

| Dataset | What we used it for |
|---|---|
| **SatNOGS Network** (SONATE-2, NORAD 59112) | Real pass recordings: the input to our receiver |
| **SatNOGS DB** | Frames decoded by SatNOGS for the same passes, used as ground truth |
| **RadioML 2016.10A** | Training a modulation classifier (11 modulations, -20 to +18 dB SNR), then testing whether it transfers to the real SONATE-2 signal. Data exploration in [explore.ipynb](explore.ipynb) |

Passes used:

| Observation | Date (UTC) | Ground station | Max elevation |
|---|---|---|---|
| [15104225](https://network.satnogs.org/observations/15104225/) | 2026-10-02 | PE0SAT-12 | 75° |
| [15052729](https://network.satnogs.org/observations/15052729/) | 2026-09-25 | DK0SB | 51° |
| [15039241](https://network.satnogs.org/observations/15039241/) | 2026-09-21 | DK0SB | 89° |

## How to run

```bash
pip install -r requirements.txt

# 1. Download the three pass recordings (no account needed)
python fetch_data.py
#    Optional: also download SatNOGS ground-truth frames (free db.satnogs.org account, token from profile settings)
SATNOGS_API_TOKEN=<your token> python fetch_data.py

# 2. Clean the ground-truth frames (only if you downloaded them)
python clean_frames.py

# 3. Decode every pass, write telemetry logs, make plots and compare with SatNOGS
python run_pipeline.py

# 4. Stress test: added noise, signal dropouts, frequency offsets (~5 min)
python stress_test.py

# 5. RadioML: download the dataset (641 MB), train the classifier (~4 min on CPU), run the transfer test
curl -L -o data/RML2016.10a_dict.pkl "https://huggingface.co/datasets/FlowVortex/RML/resolve/main/RML2016.10a_dict.pkl?download=true"
python radioml_classifier.py train
python radioml_classifier.py transfer
```

On Windows PowerShell, set the token with `$env:SATNOGS_API_TOKEN="<your token>"; python fetch_data.py`.

Outputs go to `results/`:

| File | Contents |
|---|---|
| `telemetry_log.csv` | Every decoded packet from every pass |
| `telemetry_<pass>.csv` | Decoded packets for one pass |
| `spectrum_<pass>.png` | Before/after spectrograms and average spectrum |
| `comparison.csv`, `comparison.png` | Our decoder vs. SatNOGS on the same recordings |
| `packet_counter.png`, `counter_loss.csv` | Packet loss measured from the satellite's own counter |
| `stress_test.png`, `stress_*.csv` | Noise, dropout and frequency-offset tests |
| `radioml_accuracy.png`, `radioml_transfer.png` | RadioML classifier accuracy and the real-signal transfer test |

To decode a single recording directly: `python decoder.py path/to/audio.ogg`

Raw data goes in `data/` (git-ignored).

## Results

### Before/after spectrum

Top: the raw recording from the ground station; the vertical stripes are packet bursts showing through the noise. Bottom: the clean signal rebuilt from our decoded, CRC-verified packets (re-encoded exactly as the satellite sends them). Right: the recovered signal's spectrum rolls off around half the bit rate, as 9600-baud GMSK should, while the raw audio is mostly flat noise out to the ground station's ~11 kHz audio filter.

![Before/after spectrum, pass 15104225](results/spectrum_15104225.png)

More passes: [15039241](results/spectrum_15039241.png) · [15052729](results/spectrum_15052729.png)

### Packets decoded vs. SatNOGS

Both decoders ran on the same ground-station recording:

![Comparison with SatNOGS](results/comparison.png)

| Pass | Station | SatNOGS decoder | **Our decoder** | Found by both | Repaired by soft decision |
|---|---|---|---|---|---|
| 15104225 | PE0SAT-12 | 110 | **119** | 110 | 8 |
| 15039241 | DK0SB | 382 | **265** | 265 | 118 |
| 15052729 | DK0SB | 602 | **396** | 395 | 192 |
| **Total** | | **1,094** | **780** | **770** | **318** |

- On the PE0SAT-12 pass we decoded **every packet SatNOGS did, plus 9 more**. 7 of those 9 were independently received by other ground stations, confirming they are real.
- On the two DK0SB passes we recover about **two thirds** of SatNOGS's packets. The spectrograms show where the rest go: we decode well in the strong middle of the pass but lose packets near the start and end, when the satellite is low and the signal weakest.
- **41% of our packets were saved by soft-decision repair**; without it they would have failed their checksum.
- **False accepts:** bit-flip repair makes many checksum attempts, so occasionally random noise passes the 16-bit CRC by chance. We caught 3 such frames (nonsense callsigns) and added a second check: both AX.25 callsigns must be legal characters. Every one of the 2,500+ real SatNOGS frames passes it.

### Measuring our own packet loss, without ground truth

A real deep-space receiver has no SatNOGS to check against. But SONATE-2 numbers its packets: payload byte 2 is an 8-bit counter that goes up by 1 per packet (byte 1 is the packet type, byte 3 a per-type counter; this layout is inferred from the data, not documented). We unwrap the counter (it rolls over 255 → 0 about every 95 s), and **every gap is a packet the satellite sent that we didn't decode**. So the receiver can report its own loss rate.

![Packet counter timeline](results/packet_counter.png)

| Pass | Decoded | Counter says sent (in our window) | **Estimated loss** | All stations combined, same window |
|---|---|---|---|---|
| 15104225 | 119 | 123 | **3%** | 120 |
| 15039241 | 265 | 356 | **26%** | 352 |
| 15052729 | 396 | 648 | **39%** | 594 |

**Checking the method:** wherever our packets overlap with other stations', the counter numbers line up exactly (265/265, 395/395, 117/117). The counter's estimate of packets sent is slightly *higher* than what all stations combined received, because it also counts packets no station heard (gray). Those are invisible to any comparison with ground truth.

### Stress test: noise, dropouts and frequency offsets

We took the strongest real recording (68 s, 119 packets) and injected impairments ([stress_test.py](stress_test.py)):

![Stress test](results/stress_test.png)

1. **Sensitivity (added noise).** Soft-decision rescue decodes more packets at every noise level, and the gap is largest near the cliff: at -16 dB, **58 packets vs. 29 (2×)**; at -18 dB, 98 vs. 80. At the 50% point the rescue curve sits **about 1 dB further into the noise**. Noise is added to the demodulated audio, so this is an audio-domain SNR, not RF SNR. The recording is already close to its decoding limit, which is why the curve falls off so sharply.
2. **Re-acquisition.** We replaced four stretches of the pass (2–5 s each) with signal-free receiver noise taken from the same recording. The decoder **picked up the first packet after every dropout, 0.06–0.35 s after the signal returned**, exactly the same packets as the clean run, with no reset or re-tuning.
3. **Frequency offset / drift.** Constant offsets up to 2× the signal's own swing, and slow drift (0.5 Hz), cost at most 1 packet. Fast drift (2 Hz, far beyond real residual Doppler) costs 29%: the 0.2 s drift remover can't follow it.

### RadioML: modulation classifier and synthetic-to-real transfer

We trained a small 1-D CNN (4 convolutional layers) on RadioML 2016.10A to recognise 11 modulation types from 128 I/Q samples, using the standard 50/50 train/test split.

![RadioML accuracy](results/radioml_accuracy.png)

- **57% overall, 84% at SNR ≥ 0 dB**, in line with published baselines for this dataset.
- At high SNR the FSK family (CPFSK, GFSK) is classified almost perfectly. QAM16/QAM64 and WBFM/AM-DSB are confused with each other, a known property of this dataset.

**Does it transfer to a real satellite?** SONATE-2 uses GMSK, an FSK-family modulation. Our recordings are FM-demodulated audio, so we rebuilt I/Q: the audio is the signal's instantaneous frequency, and integrating it gives the phase. We then resampled to RadioML's 8 samples per symbol and classified 128-sample windows from inside decoded packets, and from signal-free noise as a control.

![RadioML transfer test](results/radioml_transfer.png)

- Inside real packets, the model answers **FSK family 88–95% of the time** (mostly CPFSK).
- **Honest caveat:** noise-only windows are also called FSK family (99% GFSK). Rebuilding I/Q from audio makes everything constant-amplitude, which is FSK's defining feature, so the family answer is partly built in by the conversion.
- What it does show: the model **separates real signal from noise** (CPFSK vs. GFSK), so features learned on synthetic data respond to real satellite signals. A conclusive transfer test needs raw I/Q recordings.

### Telemetry log

All 780 decoded packets are in [results/telemetry_log.csv](results/telemetry_log.csv). A sample:

| Pass | UTC (approx.) | From | To | Bytes | Bits repaired | Payload (start) |
|---|---|---|---|---|---|---|
| 15039241 | 2026-09-21 16:10:45 | DP0SNX | CQ | 146 | 0 | `2700ba4618000864eb5b0073…` |
| 15039241 | 2026-09-21 16:10:48 | DP0SNX | CQ | 82 | 1 | `2702bc37180008c8c0320033…` |
| 15039241 | 2026-09-21 16:10:50 | DP0SNX | CQ | 63 | 0 | `2702bf39180008d3e0e70020…` |
| 15039241 | 2026-09-21 16:10:51 | DP0SNX | CQ | 146 | 1 | `2700c04918000864eb5e0073…` |
| 15039241 | 2026-09-21 16:11:03 | DP0SNX | CQ | 146 | 2 | `2700ca4f18000864eb640073…` |

`DP0SNX` is SONATE-2's callsign. Most packets are 146 or 271 bytes, which are likely the satellite's two main telemetry formats.

## Limitations and next steps

- **Input is SatNOGS audio, not raw IQ.** The ground station has already FM-demodulated the signal and corrected most of the Doppler shift. A next step is working from raw IQ recordings, which would mean tracking the carrier ourselves.
- **Payload fields are not fully parsed.** We verify and log each frame, but decoding individual sensor values (voltages, temperatures) needs SONATE-2's telemetry format specification.
- **Weak-signal edges of the pass.** We lose packets when the satellite is low on the horizon. SatNOGS's decoder works from the radio's full-rate data rather than the saved audio, and uses a tracking clock-recovery loop. Next steps: a proper timing loop (Gardner / Mueller-Müller) instead of the offset search, and trying 3-bit repairs.
- **Speed.** The offset search plus repair takes about 1 minute per 3 minutes of audio. A timing loop would also fix this.
- **One satellite, one modulation.** The pipeline is built for 9600-baud GMSK. A natural extension is identifying the modulation automatically (for example with a classifier trained on RadioML) so the receiver can handle any satellite.

## Team

- Katie Nguyen
- Wisena Joseph
- [jha-ruch](https://devpost.com/jha-ruch)

## Credits & License

- **SatNOGS:** Libre Space Foundation. Pass recordings and decoded frames from [SatNOGS Network](https://network.satnogs.org) and [SatNOGS DB](https://db.satnogs.org)
- **SONATE-2:** University of Würzburg
- **RadioML:** DeepSig Inc., licensed under CC BY-NC-SA 4.0 (non-commercial use, with attribution)
  - O'Shea, T. J., & West, N. "Radio Machine Learning Dataset Generation with GNU Radio." *Proceedings of the GNU Radio Conference*, 2016.
