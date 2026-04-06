# Audio Frequency Range Detector

A DSP tool that analyzes audio files (church bells, instruments, any sound) and tells you their dominant frequency range using FFT and energy percentile methods.

## What this tool does

- Cleans up audio: resamples to 22kHz, removes silence, normalizes volume, applies a high-pass filter (cuts rumble below 50 Hz)
- Computes the frequency spectrum with FFT
- Finds the significant frequency range using cumulative energy percentiles (5% to 95%)
- Shows harmonic peaks and saves a plot for every file

## How to use it

1. Clone the repo:
    git clone https://github.com/arihantk21/audio-frequency-range-detector.git
    cd audio-frequency-range-detector

2. Install dependencies:
   pip install -r requirements.txt

3. Put your `.wav` files into `raw_audio/`. The repo already includes a few sample bell sounds from Pixabay (free to use).

4. Preprocess the audio:
   python preprocess.py

5. Run the analysis:
   python analyze.py


You'll see a summary table in the terminal and frequency spectrum plots in the `plots/` folder.

## Example results (church bells)

I ran this on 28 church bell recordings. Here's what came out:

- The lowest meaningful frequency (after filtering) is about 50 Hz.
- The highest frequency with notable energy goes up to ~3.7 kHz.
- But the core energy – where the sound really lives – sits between **250 Hz and 1450 Hz** (median 5%-95% energy range).

So the answer: church bells are audible from 50 Hz to 3.7 kHz, but the frequencies you actually hear concentrate around 250–1450 Hz.

## Why the approach works

A simple min/max frequency method gives weird results (e.g., 10 Hz) because of DC offset and low-frequency noise. This tool filters out rumble (high-pass at 50 Hz) and defines the "significant" range by energy percentiles, which is more robust.

## Project files

- `preprocess.py` – cleans audio
- `analyze.py` – runs FFT and extracts range
- `requirements.txt` – Python packages needed
- `raw_audio/` – place your `.wav` files here (includes Pixabay samples)
- `plots/` – generated after analysis

## License

MIT – free to use and modify.

## Author

Arihant Kamdar – built to learn DSP concepts like FFT, filtering, and spectral analysis.
