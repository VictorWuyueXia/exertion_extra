import os
import numpy as np
import pandas as pd
import soundfile as sf

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from set_seed import seed_worker


# --- 1. Metadata extraction ---
def get_session_metadata(label_dir, audio_dir):
    session_metadata = []

    for fname in sorted(os.listdir(label_dir)):
        
        session_id = os.path.splitext(fname)[0]
        participant_id = session_id.split("_")[1]
        speed = session_id.split("_")[2]
        task = session_id.split("_")[5]
        stride = session_id.split("_")[-1]

        wav_path = os.path.join(audio_dir, f"{session_id}.wav")
        if os.path.exists(wav_path):
            duration = torchaudio.info(wav_path).num_frames / torchaudio.info(wav_path).sample_rate
        else:
            duration = 0.0

        session_metadata.append({
            "session": session_id,
            "participant": participant_id,
            "speed": speed,
            "task": task,
            "stride": stride,
            "duration": duration 
        })
    return pd.DataFrame(session_metadata)

# --- 2. Dataset class ---
class SessionSequenceDataset(Dataset):
    def __init__(self, metadata_df, feature_dir, use_mfcc=True, use_mfb=False, use_embed=False, label_dir=None, selected_wav2vec2_layers=(12,)):
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.feature_dir = feature_dir
        self.use_mfcc = use_mfcc
        self.use_mfb = use_mfb
        self.use_embed = use_embed
        self.label_dir = label_dir
        self.selected_wav2vec2_layers = selected_wav2vec2_layers    

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        session_id = row["session"]
        session_feature_dir = os.path.join(self.feature_dir, session_id)
        
        # Validate that feature directory exists
        if not os.path.exists(session_feature_dir):
            raise FileNotFoundError(f"Feature directory does not exist: {session_feature_dir}")
        
        # Load features
        mfcc = mfb = None
        embeds = []
        
        if self.use_mfcc:
            mfcc_path = os.path.join(session_feature_dir, "mfcc.npy")
            if not os.path.exists(mfcc_path):
                raise FileNotFoundError(f"MFCC file does not exist: {mfcc_path}")
            mfcc = np.load(mfcc_path)  # (1501, 40)
        else:
            mfcc = None
        
        if self.use_mfb:
            mfb_path = os.path.join(session_feature_dir, "mfb.npy")
            if not os.path.exists(mfb_path):
                raise FileNotFoundError(f"MFB file does not exist: {mfb_path}")
            mfb = np.load(mfb_path)  # (300, 40)
        else:
            mfb = None
        
        if self.use_embed:
            # Sanitize layer ids to avoid malformed entries (e.g., stray brackets)
            def _sanitize_layer_id(x):
                s = ''.join([c for c in str(x) if c.isdigit()])
                return int(s) if s else None

            sel_layers = [lid for lid in (_sanitize_layer_id(x) for x in self.selected_wav2vec2_layers) if lid is not None]

            wav2vec_dir = os.path.join(session_feature_dir, "wav2vec2")
            if not os.path.isdir(wav2vec_dir):
                raise FileNotFoundError(f"Wav2Vec2 directory does not exist: {wav2vec_dir}")

            if len(sel_layers) > 1:
                embed_list = []
                for layer_id in sel_layers:
                    layer_path = os.path.join(wav2vec_dir, f"wav2vec2_layer{layer_id}.npy")
                    if not os.path.exists(layer_path):
                        raise FileNotFoundError(f"Wav2Vec2 layer file does not exist: {layer_path}")
                    embed_list.append(np.load(layer_path))  # each shape: (T, D)
                embeds = np.concatenate(embed_list, axis=1)  # shape: (T, sum(D))
            else:
                # Fallback: if no valid layer parsed, try default 12, otherwise raise
                layer_id = sel_layers[0] if sel_layers else 12
                layer_path = os.path.join(wav2vec_dir, f"wav2vec2_layer{layer_id}.npy")
                if not os.path.exists(layer_path):
                    # Try to pick any available layer file
                    candidates = [f for f in os.listdir(wav2vec_dir) if f.startswith("wav2vec2_layer") and f.endswith(".npy")]
                    if not candidates:
                        raise FileNotFoundError(f"No Wav2Vec2 layer files found in: {wav2vec_dir}")
                    layer_path = os.path.join(wav2vec_dir, sorted(candidates)[0])
                embeds = np.load(layer_path)
        else:
            embeds = None

    
        # Load scalar label (exertion), expected to be a single integer per segment
        label_path = os.path.join(self.label_dir, f"{session_id}.npy")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file does not exist: {label_path}")
        y_np = np.load(label_path)
        # Ensure scalar int; convert 1-5 → 0-4 for CrossEntropy
        try:
            y_int = int(y_np)
        except Exception as e:
            raise ValueError(f"Label at {label_path} is not a scalar integer: value={y_np}") from e
        y_int = y_int - 1
        if not (0 <= y_int <= 4):
            raise ValueError(f"Label out of expected range [0..4] after shift at {label_path}: {y_int}")
        y = torch.tensor(y_int, dtype=torch.long)
        
        return (
            mfcc, mfb, embeds, y,
            session_id,
            row["participant"], row["speed"], row["task"], row["duration"]
        )


def collate_multi_feature_batch(batch):
    def to_tensor_and_pad(seqs):
        # tensors = [torch.tensor(x, dtype=torch.float32) for x in seqs]
        tensors = [x.detach().clone().float() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32) for x in seqs]
        lengths = [len(x) for x in tensors]
        padded = pad_sequence(tensors, batch_first=True)
        return padded, torch.tensor(lengths, dtype=torch.long)

    result = {}

    # Extract all elements
    batch_size = len(batch)
    mfcc_seqs = [b[0] for b in batch]
    mfb_seqs = [b[1] for b in batch]
    embed_seqs = [b[2] for b in batch]
    labels = [b[3] for b in batch]
    meta = list(zip(*[b[4:] for b in batch]))

    # mfcc
    if all(a is not None for a in mfcc_seqs):
        result["mfcc"], result["mfcc_lengths"] = to_tensor_and_pad(mfcc_seqs)
    else:
        result["mfcc"], result["mfcc_lengths"] = None, None

    # MFB
    if all(m is not None for m in mfb_seqs):
        result["mfb"], result["mfb_lengths"] = to_tensor_and_pad(mfb_seqs)
    else:
        result["mfb"], result["mfb_lengths"] = None, None

    # Embed
    if any(e is not None for e in embed_seqs):
        embed_tensor = torch.tensor(np.array(embed_seqs), dtype=torch.float32)  # shape: (B, 749, 768)
        result["embed"] = embed_tensor
        result["embed_lengths"] = torch.full((len(embed_seqs),), fill_value=embed_tensor.shape[1], dtype=torch.long)
    else:
        result["embed"], result["embed_lengths"] = None, None

        
    # Labels (scalar per segment): stack into shape (B,)
    result["labels"] = torch.tensor([
        int(x.item()) if isinstance(x, torch.Tensor) else int(x)
        for x in labels
    ], dtype=torch.long)
    result["label_lengths"] = None

    # === Unified Truncation ===
    candidate_lengths = []
    # Do not use label lengths for unified truncation when labels are scalar
    if result["mfcc"] is not None:
        candidate_lengths.append(result["mfcc"].shape[1])
    if result["mfb"] is not None:
        candidate_lengths.append(result["mfb"].shape[1])
    if result["embed"] is not None:
        candidate_lengths.append(result["embed"].shape[1])
    
    if not candidate_lengths:
        raise ValueError("No valid sequences found in batch")
        
    min_len = min(candidate_lengths)

    # Set primary sequence length for masking
    if result["embed_lengths"] is not None:
        result["lengths"] = torch.clamp(result["embed_lengths"], max=min_len)
    elif result["mfcc_lengths"] is not None:
        result["lengths"] = torch.clamp(result["mfcc_lengths"], max=min_len)
    else:
        # Fallback to mfb lengths (primary feature) when using scalar labels
        result["lengths"] = torch.clamp(result["mfb_lengths"], max=min_len) if result["mfb_lengths"] is not None else torch.tensor([min_len] * batch_size, dtype=torch.long)

    # Build concatenated feature when multiple inputs are present (time-aligned by data prep to same T)
    concat_list = []
    if result["embed"] is not None:
        concat_list.append(result["embed"])  # (B, T, 768 [*layers])
    if result["mfcc"] is not None:
        concat_list.append(result["mfcc"])   # (B, T, 40)
    if result["mfb"] is not None:
        concat_list.append(result["mfb"])    # (B, T, 40)

    if len(concat_list) >= 2:
        result["concat"] = torch.cat(concat_list, dim=-1)  # (B, T, F_concat)
        result["concat_lengths"] = result["lengths"]
    else:
        result["concat"], result["concat_lengths"] = None, None

    # Add metadata
    result.update({
        "session_ids": meta[0],
        "participants": meta[1],
        "speeds": meta[2],
        "tasks": meta[3],
        "durations": meta[4]
    })

    return result


# --- 4. Greedy duration-balanced fold split ---
def greedy_grouped_split(df, n_folds=5, segment_duration=15.0):
    """
    Groups all segments by original session (e.g., d01_P01_6_0_clip_1), and assigns them to folds.
    Ensures all segments of a session go to the same fold.
    Balances folds by total approximate duration (num_segments × 15s).
    """
    # Extract base session ID (excluding stride)
    df["base_session"] = df["session"].apply(lambda x: "_".join(x.split("_")[:-2]))  # removes _stride_#
    
    # Group by base session
    grouped = df.groupby("base_session")
    
    # Build a list of (base_session_id, [segment_ids])
    session_groups = [(base, group["session"].tolist()) for base, group in grouped]

    # Sort by group size (longer sessions have more segments)
    session_groups.sort(key=lambda x: len(x[1]), reverse=True)

    # Initialize folds
    folds = [{"sessions": [], "total_duration": 0.0} for _ in range(n_folds)]

    # Assign each session group to the fold with the lowest total duration
    for base_session, segment_ids in session_groups:
        min_fold = min(folds, key=lambda f: f["total_duration"])
        min_fold["sessions"].extend(segment_ids)
        min_fold["total_duration"] += len(segment_ids) * segment_duration

    return folds


def get_fixed_fold_split(folds, fold_idx):
    """
    Given 5 folds, return fixed train/val/test splits: 3/1/1
    """
    test_fold = fold_idx
    val_fold = (fold_idx + 1) % len(folds)
    train_folds = [i for i in range(len(folds)) if i not in [test_fold, val_fold]]

    test_sessions = folds[test_fold]["sessions"]
    val_sessions = folds[val_fold]["sessions"]
    train_sessions = sum([folds[i]["sessions"] for i in train_folds], [])

    return train_sessions, val_sessions, test_sessions


def get_dataloader_from_sessions(session_ids, meta_df, feature_dir, label_dir,
                                 use_mfcc=True, use_mfb=False, use_embed=False,selected_wav2vec2_layers = (12,),
                                 batch_size=4, shuffle=False, num_workers=6, pin_memory=True):
    subset_df = meta_df[meta_df["session"].isin(session_ids)]
    dataset = SessionSequenceDataset(
        subset_df,
        feature_dir=feature_dir,
        use_mfcc=use_mfcc,
        use_mfb=use_mfb,
        use_embed=use_embed,
        label_dir=label_dir,
        selected_wav2vec2_layers=selected_wav2vec2_layers
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_multi_feature_batch,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        pin_memory=True
    )
    
    return loader
