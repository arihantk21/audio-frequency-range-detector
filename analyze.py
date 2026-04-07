import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import pandas as pd

# === CONFIGURATION ===
PROCESSED_FOLDER = "preprocessed"
PLOTS_FOLDER = "plots"
TARGET_SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
PEAK_PROMINENCE = 0.1
LOW_FREQ_CUTOFF = 50          # high-pass filter cutoff (Hz)
ENERGY_PERCENTILE_LOW = 5     # lower bound of significant energy
ENERGY_PERCENTILE_HIGH = 95    # upper bound of significant energy

def high_pass_filter(y, sr, cutoff=50):
    """Apply a high-pass filter to remove low-frequency noise/rumble."""
    nyquist = sr / 2
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, y)

def analyze_audio(file_path):
    """Load preprocessed audio and extract frequency range and peaks."""
    y, sr = librosa.load(file_path, sr=TARGET_SR)
    # Remove DC offset and apply high-pass filter
    y = y - np.mean(y)
    y = high_pass_filter(y, sr, cutoff=LOW_FREQ_CUTOFF)
    
    # STFT and average spectrum
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    avg_spectrum = np.mean(S_db, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    
    # Find peaks in the spectrum
    peaks, _ = find_peaks(avg_spectrum, prominence=PEAK_PROMINENCE)
    peak_freqs = freqs[peaks]
    peak_mags = avg_spectrum[peaks]
    
    # Energy percentile method (5%-95%)
    linear_spectrum = 10 ** (avg_spectrum / 10)
    cumulative_energy = np.cumsum(linear_spectrum)
    total_energy = cumulative_energy[-1]
    low_idx = np.searchsorted(cumulative_energy, ENERGY_PERCENTILE_LOW / 100 * total_energy)
    high_idx = np.searchsorted(cumulative_energy, ENERGY_PERCENTILE_HIGH / 100 * total_energy)
    low_idx = max(0, low_idx)
    high_idx = min(len(freqs)-1, high_idx)
    low_freq_percentile = freqs[low_idx]
    high_freq_percentile = freqs[high_idx]
    
    # -10 dB bandwidth (alternative measure)
    max_mag = np.max(avg_spectrum)
    threshold = max_mag - 10
    sig_idx = np.where(avg_spectrum >= threshold)[0]
    if len(sig_idx) > 0:
        low_10db = freqs[sig_idx[0]]
        high_10db = freqs[sig_idx[-1]]
    else:
        low_10db = high_10db = 0
    
    return {
        "filename": os.path.basename(file_path),
        "low_freq_hz_percentile": low_freq_percentile,
        "high_freq_hz_percentile": high_freq_percentile,
        "low_freq_hz_10db": low_10db,
        "high_freq_hz_10db": high_10db,
        "peak_freqs_hz": peak_freqs.tolist(),
        "peak_mags_db": peak_mags.tolist(),
        "freqs": freqs,
        "spectrum": avg_spectrum,
        "y": y,           # store filtered audio for waveform/spectrogram
        "sr": sr
    }

def plot_waveform(y, sr, filename, save_path):
    """Plot time domain waveform."""
    plt.figure(figsize=(10, 4))
    time = np.linspace(0, len(y)/sr, len(y))
    plt.plot(time, y, linewidth=0.8, color='b')
    plt.title(f"Waveform: {filename}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_spectrogram(y, sr, filename, save_path, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Plot spectrogram (frequency vs time)."""
    plt.figure(figsize=(10, 5))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    librosa.display.specshow(D, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.colorbar(label='dB')
    plt.title(f"Spectrogram: {filename}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_spectrum(result, save_path):
    """Plot frequency spectrum with highlighted ranges and peaks."""
    plt.figure(figsize=(10, 5))
    plt.semilogx(result["freqs"], result["spectrum"], linewidth=1.5)
    plt.title(f"Frequency Spectrum: {result['filename']}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Highlight percentile-based range (green)
    plt.axvspan(result["low_freq_hz_percentile"], result["high_freq_hz_percentile"],
                alpha=0.25, color='green', label='5%-95% energy range')
    
    # Highlight -10 dB bandwidth (red, optional)
    if result["low_freq_hz_10db"] > 0:
        plt.axvspan(result["low_freq_hz_10db"], result["high_freq_hz_10db"],
                    alpha=0.15, color='red', label='-10 dB bandwidth')
    
    # Mark harmonic peaks
    plt.plot(result["peak_freqs_hz"], result["peak_mags_db"], 'rx', markersize=8, label='Harmonic peaks')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    # Create output folder
    os.makedirs(PLOTS_FOLDER, exist_ok=True)
    
    # Get all preprocessed .wav files
    files = [f for f in os.listdir(PROCESSED_FOLDER) if f.endswith('.wav')]
    if not files:
        print(f"No .wav files found in '{PROCESSED_FOLDER}'. Run preprocess.py first.")
        return
    
    print(f"Analyzing {len(files)} audio files...\n")
    results = []
    
    for fname in files:
        file_path = os.path.join(PROCESSED_FOLDER, fname)
        res = analyze_audio(file_path)
        results.append(res)
        
        # Base name for plots
        base_name = os.path.splitext(fname)[0]
        
        # Generate three plots
        plot_spectrum(res, os.path.join(PLOTS_FOLDER, f"{base_name}_spectrum.png"))
        plot_waveform(res["y"], res["sr"], fname, os.path.join(PLOTS_FOLDER, f"{base_name}_waveform.png"))
        plot_spectrogram(res["y"], res["sr"], fname, os.path.join(PLOTS_FOLDER, f"{base_name}_spectrogram.png"))
        
        print(f"  {fname}: dominant range = {res['low_freq_hz_percentile']:.1f} - {res['high_freq_hz_percentile']:.1f} Hz")
    
    # === Summary table ===
    df = pd.DataFrame([{
        "File": r["filename"],
        "Low (5% energy) Hz": round(r["low_freq_hz_percentile"], 1),
        "High (95% energy) Hz": round(r["high_freq_hz_percentile"], 1),
        "-10dB Low Hz": round(r["low_freq_hz_10db"], 1),
        "-10dB High Hz": round(r["high_freq_hz_10db"], 1),
        "Peaks (Hz)": str([round(p,1) for p in r["peak_freqs_hz"]])
    } for r in results])
    
    print("\n" + "="*80)
    print("SUMMARY TABLE (Percentile-based Frequency Ranges)")
    print("="*80)
    print(df.to_string(index=False))
    
    # === Conclusion ===
    lows = [r["low_freq_hz_percentile"] for r in results if r["low_freq_hz_percentile"] > 0]
    highs = [r["high_freq_hz_percentile"] for r in results if r["high_freq_hz_percentile"] > 0]
    
    if lows and highs:
        median_low = np.median(lows)
        median_high = np.median(highs)
        mean_low = np.mean(lows)
        mean_high = np.mean(highs)
        
        print("\n" + "="*80)
        print("CONCLUSION")
        print("="*80)
        print(f"Number of bell samples: {len(results)}")
        print(f"Total observable frequency range (across all samples):")
        print(f"  Minimum low frequency: {min(lows):.1f} Hz")
        print(f"  Maximum high frequency: {max(highs):.1f} Hz")
        print(f"\nTypical significant frequency range (median across samples):")
        print(f"  {median_low:.0f} Hz to {median_high:.0f} Hz")
        print(f"\nTypical significant frequency range (mean across samples):")
        print(f"  {mean_low:.0f} Hz to {mean_high:.0f} Hz")
        print(f"\nBased on the 5%-95% energy percentile method, church bell sounds")
        print(f"concentrate their audible energy approximately between {median_low:.0f} Hz and {median_high:.0f} Hz.")
    else:
        print("No valid frequency ranges found.")

if __name__ == "__main__":
    main()