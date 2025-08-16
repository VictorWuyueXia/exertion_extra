import os
import soundfile as sf
import torchaudio
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

audio_dir = "/Users/cielwang/Desktop/Research with Jingping/data_after_cardio/cleaned_data/cleaned_audio"  # <-- Update this
output_plot = "audio_length_distribution.png"

durations_sec = []

for fname in tqdm(sorted(os.listdir(audio_dir))):
    if not fname.endswith(".wav"):
        continue
    filepath = os.path.join(audio_dir, fname)
    y, sr = sf.read(filepath)

    if sr != 16000:
        y = torchaudio.functional.resample(torch.tensor(y), orig_freq=sr, new_freq=16000).numpy()
        sr = 16000

    duration = len(y) / sr
    durations_sec.append(duration)

# Plot
plt.figure(figsize=(8, 5))
plt.hist(durations_sec, bins=50, color='skyblue', edgecolor='black')
plt.xlabel("Audio Duration (seconds)")
plt.ylabel("Number of Files")
plt.title("Distribution of Audio Durations")
plt.tight_layout()
plt.savefig(output_plot)
plt.close()

