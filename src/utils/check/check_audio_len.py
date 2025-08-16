import os
import argparse
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_audio_lengths(audio_dir, sr=16000):
    lengths = []
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a'}

    for root, _, files in os.walk(audio_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                filepath = os.path.join(root, file)
                try:
                    y, _ = librosa.load(filepath, sr=sr)
                    duration = len(y) / sr
                    lengths.append(duration)
                except Exception as e:
                    print(f"Failed to process {filepath}: {e}")
    return lengths

def plot_distribution(lengths, output_path=None):
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel("Audio Length (seconds)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Audio File Lengths")
    output_path = "audio_length_distribution.png"
    plt.savefig(output_path)
    print(f"Saved histogram to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check audio length distribution in a folder.")
    parser.add_argument("folder", help="Path to folder containing audio files")
    parser.add_argument("--sr", type=int, default=16000, help="Sampling rate (default: 16000)")
    parser.add_argument("--save", help="Path to save histogram (optional)")
    args = parser.parse_args()

    print(f"Scanning audio files in: {args.folder}")
    lengths = get_audio_lengths(args.folder, sr=args.sr)

    total = len(lengths)
    under_15 = sum(l < 15 for l in lengths)
    under_20 = sum(l < 20 for l in lengths)

    print(f"Found {total} audio files.")
    print(f"Number of files shorter than 15s: {under_15}")
    print(f"Number of files shorter than 20s: {under_20}")

    plot_distribution(lengths, args.save)
