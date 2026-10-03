# seds-hack
## GHOST FRAMES: Recovering satellite packets that ground stations discarded
### Overview

When a satellite passes near the horizon, its signal can become weak enough that noise causes individual bits to be corrupted. Even a small number of errors can cause a packet's checksum to fail, causing the frame to be rejected as invalid by the decoder.

GHOST FRAMES is an automated receiver designed to recover these otherwise discarded frames. Our system detects weak signal bursts, estimates key signal parameters, demodulates and decodes the transmission, and attempts to repair corrupted frames using the least-confident received bits.

We evaluate the system using real satellite recordings from SONATE-2 through SatNOGS, while using RadioML 2016.10A to benchmark modulation classification across different signal-to-noise ratios.


### Data
RadioML 2016.10A

RadioML 2016.10A is a dataset of simulated radio signals containing 11 different modulation types across a range of signal-to-noise ratios (SNRs).

We use RadioML to:

.....

SatNOGS/SONATE-2

We use real radio observations of SONATE-2 from SatNOGS. These recordings contain the actual received satellite signal along with decoded data provided by the SatNOGS network.

We use this data to:

- Detect signal bursts in real recordings.
- Estimate signal parameters such as baud rate.
- Demodulate and decode real satellite transmissions.
- Compare our decoded frames against SatNOGS's decoded results.
- Test whether our frame-rescue method can recover frames that the standard decoder did not successfully decode.


### How It Works
Our receiver follows a fully automated pipeline:

Detect → Estimate → Demodulate → Decode → Rescue → Verify

Detect: Find signal bursts using the local noise floor rather than a manually selected threshold.

Estimate: Automatically estimate signal characteristics such as baud rate and tone frequencies.

Demodulate: Convert the received waveform into soft bits that retain information about how confident the receiver is in each bit.

Decode: Recover AX.25 frames and verify them using their CRC.

Rescue: When a frame fails its CRC, test combinations of the least-confident bits to determine whether a small number of errors can be corrected.

Verify: Compare recovered frames against SatNOGS decoded data to determine whether the receiver recovered information that the standard decoder missed.

### Results

### How to Run it

### Limitations and Next Steps




## Team
- Katie Nguyen
- Wisena Joseph
- [jha-ruch](https://devpost.com/jha-ruch)

## Credits & License

- **RadioML:** DeepSig Inc., licensed under CC BY-NC-SA 4.0 (non-commercial use, with attribution)
  - O'Shea, T. J., & West, N. "Radio Machine Learning Dataset Generation with GNU Radio." *Proceedings of the GNU Radio Conference*, 2016.
- **SatNOGS:** Libre Space Foundation
