#!/usr/bin/env python3
"""
Feature extraction for new split data.
Extracts:
1. MFB: (750, 40) - Kaldi-style mel filter bank
2. MFCC: (1501, 40) - Mel-frequency cepstral coefficients  
3. Wav2Vec2: (749, 768) - Layers 4, 6, 7, 8, 12

All features extracted from 15s audio segments resampled to 16kHz.
"""

import os
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.compliance.kaldi as kaldi
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from pathlib import Path


# Global variables for Wav2Vec2 model (loaded once)
processor = None
model = None
device = None

def initialize_wav2vec2_model():
    """Initialize Wav2Vec2 model on GPU - called once"""
    global processor, model, device
    if model is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base", cache_dir=".cache")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", cache_dir=".cache", output_hidden_states=True)
        model = model.to(device)
        model.eval()
        print("Wav2Vec2 model loaded successfully")


def load_and_resample_audio(wav_path, sr_target=16000):
    """Load audio and resample to target sample rate"""
    waveform, sr = torchaudio.load(wav_path)
    
    # Convert to mono if stereo
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if sr != sr_target:
        waveform = torchaudio.functional.resample(waveform, sr, sr_target)
    
    return waveform.squeeze(0).numpy(), sr_target


def extract_mfb(audio, sr=16000, target_frames=750):
    """提取MFB特征，目标形状为 (750, 40)"""
    # 转为tensor以适配Kaldi
    if isinstance(audio, np.ndarray):
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
    else:
        audio_tensor = audio

    # Kaldi fbank参数调整为15s音频提取750帧
    # 15s / 750 ≈ 0.02s ≈ 20ms
    frame_shift_ms = (15.0 / target_frames) * 1000  # 毫秒

    mfb = kaldi.fbank(
        waveform=audio_tensor.unsqueeze(0),
        num_mel_bins=40,
        sample_frequency=sr,
        frame_length=25.0,
        frame_shift=frame_shift_ms,
        dither=0.0,
        energy_floor=0.0,
        use_energy=False
    )

    # 对齐到target_frames帧
    mfb = align_frames(mfb, target_frames)

    return mfb.numpy()


def extract_mfcc(audio, sr=16000, target_frames=750):
    """使用torchaudio提取MFCC特征，目标形状为 (750, 40)"""
    # 如有需要，转为tensor
    if isinstance(audio, np.ndarray):
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
    else:
        audio_tensor = audio

    # 计算hop_length以获得目标帧数
    # 15s / 750 ≈ 0.02s，每帧约20ms，@16kHz约320采样点
    hop_length = int(sr * 15.0 / target_frames)

    mfcc_transform = T.MFCC(
        sample_rate=sr,
        n_mfcc=40,
        melkwargs={
            "n_fft": 512,
            "win_length": 400,  # 25ms @ 16kHz
            "hop_length": hop_length,
            "n_mels": 128,
            "f_max": 8000,
            "window_fn": torch.hann_window
        }
    )

    mfcc = mfcc_transform(audio_tensor.unsqueeze(0)).squeeze(0).T  # 形状: (T, 40)

    # 对齐到target_frames帧
    mfcc = align_frames(mfcc, target_frames)

    return mfcc.numpy()


def extract_wav2vec2(audio, sr=16000, target_frames=750, selected_layers=(4, 6, 7, 8, 12)):
    """Extract Wav2Vec2 embeddings - target shape (750, 768) for each layer"""
    initialize_wav2vec2_model()
    
    # Convert to tensor if needed
    if isinstance(audio, np.ndarray):
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
    else:
        audio_tensor = audio
    
    # Process with Wav2Vec2
    inputs = processor(
        audio_tensor, 
        sampling_rate=sr, 
        return_tensors="pt", 
        padding=True
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
        # Extract selected layers
        layer_features = {}
        for layer_idx in selected_layers:
            # Get features for this layer
            features = outputs.hidden_states[layer_idx][0].cpu().numpy()  # shape: (T, 768)

            # Align to target frames (align_frames returns torch.Tensor)
            features = align_frames(features, target_frames)

            # Save as numpy
            if torch.is_tensor(features):
                features = features.cpu().numpy()
            layer_features[f'layer{layer_idx}'] = features
    
    return layer_features


def align_frames(feature, target_frames):
    """Align feature to target number of frames. Accepts numpy array or torch tensor, returns torch tensor."""
    # Ensure torch tensor for uniform ops
    if isinstance(feature, np.ndarray):
        feature = torch.tensor(feature, dtype=torch.float32)

    current_frames = feature.shape[0]

    if current_frames == target_frames:
        return feature
    elif current_frames < target_frames:
        pad_length = target_frames - current_frames
        if feature.dim() == 1:
            last_frame = feature[-1:].repeat(pad_length)
            return torch.cat([feature, last_frame], dim=0)
        else:
            last_frame = feature[-1:].repeat(pad_length, 1)
            return torch.cat([feature, last_frame], dim=0)
    else:
        return feature[:target_frames]


def process_single_file(args):
    """Process a single audio file and extract all features"""
    fname, audio_dir, features_dir = args
    
    session_id = os.path.splitext(fname)[0]
    wav_path = os.path.join(audio_dir, fname)
    session_dir = os.path.join(features_dir, session_id)
    wav2vec_dir = os.path.join(session_dir, "wav2vec2")
    
    # Create directories
    os.makedirs(session_dir, exist_ok=True)
    os.makedirs(wav2vec_dir, exist_ok=True)
    
    try:
        # Load and resample audio
        audio, sr = load_and_resample_audio(wav_path, sr_target=16000)
        
        # Verify audio duration (should be ~15s)
        duration = len(audio) / sr
        if abs(duration - 15.0) > 0.5:  # Allow 0.5s tolerance
            print(f"Warning: {session_id} duration is {duration:.2f}s (expected ~15s)")
        
        # Extract MFB (750, 40)
        mfb = extract_mfb(audio, sr, target_frames=750)
        np.save(os.path.join(session_dir, "mfb.npy"), mfb)
        
        # Extract MFCC (750, 40)
        mfcc = extract_mfcc(audio, sr, target_frames=750)
        np.save(os.path.join(session_dir, "mfcc.npy"), mfcc)
        
        # Extract Wav2Vec2 embeddings (750, 768) for each layer
        wav2vec_features = extract_wav2vec2(audio, sr, target_frames=750, selected_layers=(4, 6, 7, 8, 12))
        
        for layer_name, features in wav2vec_features.items():
            np.save(os.path.join(wav2vec_dir, f"wav2vec2_{layer_name}.npy"), features)
        
        return f"Success: {session_id}"
        
    except Exception as e:
        return f"Failed on {session_id}: {e}"


def extract_features_for_split(split_name, audio_dir, features_dir):
    """Extract features for a specific split"""
    print(f"\n{'='*60}")
    print(f"EXTRACTING FEATURES FOR {split_name.upper()} SET")
    print(f"{'='*60}")
    
    # Get all audio files
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    print(f"Found {len(audio_files)} audio files")
    
    # Process files
    results = []
    for fname in tqdm(audio_files, desc=f"Processing {split_name}"):
        result = process_single_file((fname, audio_dir, features_dir))
        results.append(result)
    
    # Print results summary
    success_count = sum(1 for r in results if r.startswith("Success"))
    failed_count = len(results) - success_count
    
    print(f"\n{split_name.upper()} SET SUMMARY:")
    print(f"  Total files: {len(audio_files)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {failed_count}")
    
    # Show some failures if any
    failures = [r for r in results if r.startswith("Failed")]
    if failures:
        print(f"  Failures:")
        for failure in failures[:5]:  # Show first 5
            print(f"    {failure}")
        if len(failures) > 5:
            print(f"    ... and {len(failures) - 5} more")
    
    return success_count, failed_count


def main():
    """Main function to extract features for all splits"""
    print("FEATURE EXTRACTION FOR NEW SPLIT DATA")
    print("="*60)
    
    base_dir = "."
    splits = ['train', 'val', 'test']
    
    total_success = 0
    total_failed = 0
    
    for split_name in splits:
        # Define paths
        audio_dir = os.path.join(base_dir, f"Data/new_split_{split_name}", "segmented_audio")
        features_dir = os.path.join(base_dir, f"Data/new_split_{split_name}", "features")
        
        # Check if audio directory exists
        if not os.path.exists(audio_dir):
            print(f"\nWARNING: Audio directory not found: {audio_dir}")
            continue
        
        # Extract features for this split
        success, failed = extract_features_for_split(split_name, audio_dir, features_dir)
        total_success += success
        total_failed += failed
    
    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"Total successful: {total_success}")
    print(f"Total failed: {total_failed}")
    print(f"Success rate: {(total_success / (total_success + total_failed) * 100):.1f}%")
    
    print(f"\nFeature dimensions:")
    print(f"  MFB: (750, 40)")
    print(f"  MFCC: (750, 40)")
    print(f"  Wav2Vec2: (750, 768) for layers 4, 6, 7, 8, 12")
    
    print(f"\nOutput directories:")
    for split_name in splits:
        print(f"  {base_dir}/new_split_{split_name}/features/")


if __name__ == "__main__":
    main()
