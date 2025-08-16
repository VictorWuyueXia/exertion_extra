#!/usr/bin/env python3
"""
Sliding window segmentation for new split data.
Applies 15s window with 1s stride to all splits (train, val, test).
Adapts the logic from segment_audio_label.py for the new split structure.
"""

import os
import csv
import torch
import numpy as np
import pandas as pd
import torchaudio
from tqdm import tqdm
import soundfile as sf
from pathlib import Path


# --- Config ---
sr_target = 16000  # audio
label_sr = 20      # label (unused now when using CSV exertion labels)
window_sec = 15.0
stride_sec = 1.0

base_dir = "./Data"


def segment_indices(total_len, window_len, stride_len):
    """Generate segment indices for sliding window."""
    return [(start, min(start + window_len, total_len))
            for start in range(0, total_len, stride_len)
            if start + window_len <= total_len]


def _read_exertion_map(labels_csv_path):
    """Read labels.csv and return filename -> exertion(int) dict."""
    mapping = {}
    with open(labels_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # normalize headers
        field_map = {name.strip(): name for name in reader.fieldnames or []}
        fname_key = field_map.get("filename")
        ex_key = field_map.get("exertion")
        if not fname_key or not ex_key:
            raise ValueError(f"labels.csv missing required headers at {labels_csv_path}: {reader.fieldnames}")
        for row in reader:
            fname = (row.get(fname_key) or "").strip()
            ex_raw = (row.get(ex_key) or "").strip()
            if not fname:
                continue
            try:
                ex_val = int(ex_raw)
            except ValueError:
                continue
            mapping[fname] = ex_val
    return mapping


def process_split_segments(split_name, audio_dir, labels_csv_path, out_audio_dir, out_label_dir):
    """
    Process sliding window segmentation for a specific split.
    
    Args:
        split_name: 'train', 'val', or 'test'
        audio_dir: Directory with audio files
        label_dir: Directory with label files
        out_audio_dir: Output directory for segmented audio
        out_label_dir: Output directory for segmented labels
    """
    print(f"\nProcessing {split_name} set...")
    
    # Create output directories
    os.makedirs(out_audio_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)
    
    # Get all audio files and label map (filename -> exertion)
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    exertion_map = _read_exertion_map(labels_csv_path)
    
    segment_counts = []
    
    for fname in tqdm(sorted(audio_files), desc=f"Processing {split_name}"):
        session_id = os.path.splitext(fname)[0]
        audio_path = os.path.join(audio_dir, fname)
        # Lookup exertion label from CSV
        if fname not in exertion_map:
            print(f"Missing exertion in labels.csv for {fname}")
            continue
        exertion_val = int(exertion_map[fname])
        
        # Load audio
        y, sr = sf.read(audio_path)
        if sr != sr_target:
            y = torchaudio.functional.resample(
                torch.tensor(y), orig_freq=sr, new_freq=sr_target
            ).numpy()
        audio_len = len(y)
        
        # Compute segment indices
        audio_window = int(window_sec * sr_target)
        audio_stride = int(stride_sec * sr_target)
        
        audio_segments = segment_indices(audio_len, audio_window, audio_stride)
        
        num_segments = len(audio_segments)
        segment_counts.append({"session_id": session_id, "num_segments": num_segments})
        
        # Create segments
        for i, (a_start, a_end) in enumerate(audio_segments, start=1):
            seg_name = f"{session_id}_stride_{i}"
            
            # Save audio segment
            audio_segment = y[a_start:a_end]
            sf.write(os.path.join(out_audio_dir, f"{seg_name}.wav"), audio_segment, sr_target)
            
            # Save label segment (scalar exertion only)
            np.save(os.path.join(out_label_dir, f"{seg_name}.npy"), np.array(exertion_val, dtype=np.int64))
    
    # Save segment counts
    counts_df = pd.DataFrame(segment_counts)
    counts_df.to_csv(os.path.join(out_audio_dir, f"{split_name}_segment_counts.csv"), index=False)
    
    print(f"{split_name} set: {len(audio_files)} files, {sum(counts_df['num_segments'])} total segments")
    return counts_df


def main():
    """Main function to process all splits."""
    print("Starting sliding window segmentation for new split data...")
    
    # Import pandas here to avoid dependency issues
    import pandas as pd
    
    splits = ['train', 'val', 'test']
    total_segments = 0
    
    for split_name in splits:
        # Define paths
        audio_dir = os.path.join(base_dir, f"new_split_{split_name}", "audio")
        labels_csv_path = os.path.join(base_dir, f"new_split_{split_name}", "labels.csv")
        out_audio_dir = os.path.join(base_dir, f"new_split_{split_name}", "segmented_audio")
        out_label_dir = os.path.join(base_dir, f"new_split_{split_name}", "segmented_labels")
        
        # Check if input directories exist
        if not os.path.exists(audio_dir):
            print(f"Warning: Audio directory not found: {audio_dir}")
            continue
        if not os.path.exists(labels_csv_path):
            print(f"Warning: labels.csv not found: {labels_csv_path}")
            continue
        
        # Process this split
        counts_df = process_split_segments(split_name, audio_dir, labels_csv_path, out_audio_dir, out_label_dir)
        total_segments += sum(counts_df['num_segments'])
    
    print("\n" + "="*60)
    print("SLIDING WINDOW SEGMENTATION COMPLETE")
    print("="*60)
    print(f"Total segments created: {total_segments}")
    print(f"Window size: {window_sec}s")
    print(f"Stride: {stride_sec}s")
    print(f"Audio sample rate: {sr_target}Hz")
    print(f"Label frame rate: {label_sr}Hz")
    print("\nOutput directories:")
    for split_name in splits:
        print(f"  {base_dir}/new_split_{split_name}/segmented_audio/")
        print(f"  {base_dir}/new_split_{split_name}/segmented_labels/")


if __name__ == "__main__":
    main()
