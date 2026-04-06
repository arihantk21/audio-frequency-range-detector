import os
import librosa
import numpy as np
import soundfile as sf

RAW_FOLDER = "raw_audio"
PROCESSED_FOLDER = "preprocessed"
TARGET_SR = 22050
TRIM_TOP_DB = 60

def preprocess_one_file(input_path, output_path):
    print(f"Processing: {os.path.basename(input_path)}")
    y, sr = librosa.load(input_path, sr=None, mono=True)
    y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    y_trimmed, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    if len(y_trimmed) == 0:
        print(f"  Warning: {input_path} is all silence after trimming. Skipped.")
        return
    max_val = np.max(np.abs(y_trimmed))
    if max_val > 0:
        y_trimmed = y_trimmed / max_val
    sf.write(output_path, y_trimmed, TARGET_SR)
    print(f"  Saved: {output_path}")

def main():
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    files = [f for f in os.listdir(RAW_FOLDER) if f.lower().endswith('.wav')]
    if not files:
        print(f"No .wav files found in '{RAW_FOLDER}'. Please add some.")
        return
    print(f"Found {len(files)} audio files.\n")
    for fname in files:
        input_path = os.path.join(RAW_FOLDER, fname)
        output_path = os.path.join(PROCESSED_FOLDER, f"proc_{fname}")
        preprocess_one_file(input_path, output_path)
    print("\n[SUCCESS] Preprocessing complete! Cleaned files in 'preprocessed' folder.")

if __name__ == "__main__":
    main()