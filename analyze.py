import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import pandas as pd

# === CONFIGURATION ===
PROCESSED_FOLDER = "preprocessed"
PLOTS_FOLDER = "plots"
TARGET_SR = 22050
N_FFT = 4096
PEAK_PROMINENCE = 0.1
LOW_FREQ_CUTOFF = 50      # remove frequencies below 50 Hz (rumble/DC)
ENERGY_PERCENTILE_LOW = 10   # lower bound of significant energy
ENERGY_PERCENTILE_HIGH = 90 # upper bound of significant energy

def high_pass_filter(y, sr, cutoff=50):
    """Apply a high-pass filter to remove low-frequency noise."""
    nyquist = sr / 2
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, y)

def analyze_audio(file_path):
    """Load, filter, and extract meaningful frequency range."""
    y, sr = librosa.load(file_path, sr=TARGET_SR)
    
    # Remove DC offset
    y = y - np.mean(y)
    
    # Apply high-pass filter to eliminate rumble (optional but recommended)
    y = high_pass_filter(y, sr, cutoff=LOW_FREQ_CUTOFF)
    
    # Compute STFT and average spectrum
    D = librosa.stft(y, n_fft=N_FFT)
    S = np.abs(D)
    avg_spectrum = np.mean(S, axis=1)   # linear averaging (correct)
    S_db = librosa.amplitude_to_db(avg_spectrum, ref=np.max)  # only for plotting
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    
    # Find spectral peaks
    prominence = 0.1 * np.max(avg_spectrum)
    peaks, _ = find_peaks(avg_spectrum, prominence=prominence)
    peak_freqs = freqs[peaks]
    peak_mags = avg_spectrum[peaks]
    
    # Convert spectrum from dB to linear power for energy calculation
    linear_spectrum = avg_spectrum ** 2   # true power (energy)
    cumulative_energy = np.cumsum(linear_spectrum)
    total_energy = cumulative_energy[-1]
    
    # Find frequencies where cumulative energy crosses percentiles
    low_idx = np.searchsorted(cumulative_energy, ENERGY_PERCENTILE_LOW / 100 * total_energy)
    high_idx = np.searchsorted(cumulative_energy, ENERGY_PERCENTILE_HIGH / 100 * total_energy)
    
    # Ensure indices are within bounds
    low_idx = max(0, low_idx)
    high_idx = min(len(freqs)-1, high_idx)
    
    low_freq = freqs[low_idx]
    high_freq = freqs[high_idx]
    
    # Also compute the -10 dB bandwidth (original method) for comparison
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
        "low_freq_hz_percentile": low_freq,
        "high_freq_hz_percentile": high_freq,
        "low_freq_hz_10db": low_10db,
        "high_freq_hz_10db": high_10db,
        "peak_freqs_hz": peak_freqs.tolist(),
        "peak_mags_db": peak_mags.tolist(),
        "freqs": freqs,
        "spectrum": avg_spectrum
    }

def plot_spectrum(result, save=True):
    """Generate improved plot with both energy percentiles and -10dB range."""
    plt.figure(figsize=(12, 6))
    plt.semilogx(result["freqs"], result["spectrum"], linewidth=1.5)
    plt.title(f"Frequency Spectrum: {result['filename']}", fontsize=14)
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Magnitude (dB)", fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Highlight percentile-based range (green)
    plt.axvspan(result["low_freq_hz_percentile"], result["high_freq_hz_percentile"],
                alpha=0.25, color='green', label=f'5%-95% energy range')
    
    # Highlight -10 dB range (red, semi-transparent)
    if result["low_freq_hz_10db"] > 0:
        plt.axvspan(result["low_freq_hz_10db"], result["high_freq_hz_10db"],
                    alpha=0.15, color='red', label='-10 dB bandwidth')
    
    # Mark peaks
    plt.plot(result["peak_freqs_hz"], result["peak_mags_db"], 'rx', markersize=8, label='Harmonic peaks')
    plt.legend(loc='upper right')
    
    if save:
        os.makedirs(PLOTS_FOLDER, exist_ok=True)
        plot_path = os.path.join(PLOTS_FOLDER, result["filename"].replace('.wav', '_spectrum.png'))
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot: {plot_path}")
    plt.close()

def main():
    files = [f for f in os.listdir(PROCESSED_FOLDER) if f.endswith('.wav')]
    if not files:
        print(f"No .wav files found in '{PROCESSED_FOLDER}'. Run preprocess.py first.")
        return
    
    print(f"Analyzing {len(files)} audio files...\n")
    results = []
    for fname in files:
        path = os.path.join(PROCESSED_FOLDER, fname)
        res = analyze_audio(path)
        results.append(res)
        plot_spectrum(res)
        print(f"  {fname}: significant energy range = {res['low_freq_hz_percentile']:.1f} - {res['high_freq_hz_percentile']:.1f} Hz")
    
    # Create summary DataFrame
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
    
    # Use median for robust "typical" range
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
        print(f"\n✅ Based on the 5%-95% energy percentile method, church bell sounds")
        print(f"   concentrate their audible energy between approximately {median_low:.0f} Hz and {median_high:.0f} Hz.")
    else:
        print("No valid frequency ranges found.")

if __name__ == "__main__":
    main()