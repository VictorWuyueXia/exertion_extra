import os
import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mode
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sklearn.metrics import (
    classification_report, 
    accuracy_score,
    roc_auc_score, 
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def two_stage_predictions_to_classes(break_logits, type_logits, threshold=0.5):
    """
    Convert two-stage predictions back to original 4-class format
    
    Args:
        break_logits: (B, T) - binary break detection logits
        type_logits: (B, T, 3) - break type classification logits
        threshold: threshold for break detection
    
    Returns:
        final_predictions: (B, T) with values 0, 1, 2, 3
    """
    batch_size, seq_len = break_logits.shape
    
    # Stage 1: detect breaks
    break_probs = torch.sigmoid(break_logits)  # (B, T)
    is_break = (break_probs > threshold)  # (B, T)
    
    # Stage 2: classify break types
    type_predictions = torch.argmax(type_logits, dim=-1)  # (B, T) with values 0,1,2
    
    # Combine predictions
    final_predictions = torch.zeros_like(break_logits)  # Start with all 0s (no break)
    
    # Where breaks are detected, use type predictions (+1 to shift back to original labels)
    final_predictions[is_break] = type_predictions[is_break].float() + 1
    
    return final_predictions


def compute_regression_metrics(y_true_list, y_pred_list, session_ids, save_dir=None, prefix=""):
    """
    Computes regression metrics and saves per-session y_true/y_pred.

    Args:
        y_true_list (List[np.ndarray]): List of arrays of shape (T,) — one per session.
        y_pred_list (List[np.ndarray]): Same shape as y_true_list.
        session_ids (List[str]): List of session IDs.
        save_dir (str): Directory to save outputs.
        prefix (str): Prefix for output filenames.
        
    Returns:
        dict: MAE, MSE, R²
    """
    assert len(y_true_list) == len(y_pred_list) == len(session_ids)

    # Flatten for global metrics
    all_y_true = np.concatenate(y_true_list)
    all_y_pred = np.concatenate(y_pred_list)

    mae = mean_absolute_error(all_y_true, all_y_pred)
    mse = mean_squared_error(all_y_true, all_y_pred)
    r2 = r2_score(all_y_true, all_y_pred)

    metrics = {
        "mae": mae,
        "mse": mse,
        "r2_score": r2
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # Save per-session predictions
        session_data = []
        for sid, y_t, y_p in zip(session_ids, y_true_list, y_pred_list):
            session_data.append({
                "session_id": sid,
                "y_true": " ".join([f"{v:.4f}" for v in y_t]),
                "y_pred": " ".join([f"{v:.4f}" for v in y_p])
            })

        df = pd.DataFrame(session_data)
        df.to_csv(os.path.join(save_dir, f"{prefix}regression_predictions.csv"), index=False)

        # Save metrics
        pd.DataFrame([metrics]).to_csv(os.path.join(save_dir, f"{prefix}regression_metrics_summary.csv"), index=False)

    return metrics



def apply_temporal_smoothing(preds, window=5):
    padded = np.pad(preds, (window // 2, window // 2), mode='edge')
    smoothed = []

    for i in range(len(preds)):
        window_slice = padded[i:i + window]
        mode_result = mode(window_slice, keepdims=True)
        smoothed.append(mode_result.mode[0] if hasattr(mode_result, 'mode') else mode_result[0])
    
    return smoothed

def merge_nearby_segments(preds, frame_rate=20, max_gap_sec=0.3):
    gap_frames = int(frame_rate * max_gap_sec)
    preds = np.array(preds)
    merged = preds.copy()

    for cls in [1, 2, 3]:
        indices = np.where(preds == cls)[0]
        if len(indices) == 0:
            continue
        segments = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
        for i in range(1, len(segments)):
            prev_end = segments[i-1][-1]
            curr_start = segments[i][0]
            if curr_start - prev_end <= gap_frames:
                merged[prev_end+1:curr_start] = cls
    return merged.tolist()


def compute_tIoU(gt, pred, cls):
    gt_mask = np.array(gt) == cls
    pred_mask = np.array(pred) == cls
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return intersection / union if union > 0 else 0.0


def compute_segment_precision_recall(gt, pred, cls):
    def find_segments(x):
        segments = []
        in_seg = False
        for i, val in enumerate(x):
            if val == cls and not in_seg:
                start = i
                in_seg = True
            elif val != cls and in_seg:
                segments.append((start, i-1))
                in_seg = False
        if in_seg:
            segments.append((start, len(x)-1))
        return segments

    gt_segments = find_segments(gt)
    pred_segments = find_segments(pred)

    def overlaps(a, b):
        return a[1] >= b[0] and b[1] >= a[0]

    tp = 0
    for p in pred_segments:
        if any(overlaps(p, g) for g in gt_segments):
            tp += 1

    precision = tp / len(pred_segments) if pred_segments else 0.0
    recall = tp / len(gt_segments) if gt_segments else 0.0
    return precision, recall


def compute_false_break_rate(y_true, y_pred):
    """
    Calculate how often breaks ('b', 's', 'bs') are predicted when GT is 'o'.
    """
    y_true = torch.cat(y_true).cpu().numpy()
    y_pred = torch.cat(y_pred).cpu().numpy()
    mask = y_true != -100
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    o_idx = 0
    break_indices = [1, 2, 3]

    total_o = (y_true == o_idx).sum()
    false_breaks = ((y_true == o_idx) & np.isin(y_pred, break_indices)).sum()

    rate = false_breaks / total_o if total_o > 0 else 0.0
    return rate


def save_segment_level_predictions(session_ids, preds, trues, output_path):
    """
    Save per-segment predictions and ground truths to CSV for offline error analysis.
    """
    records = []
    for sid, y_pred, y_true in zip(session_ids, preds, trues):
        records.append({
            "session_id": sid,
            "y_pred": y_pred.cpu().tolist(),
            "y_true": y_true.cpu().tolist()
        })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def save_fold_metrics(metrics_dict, output_path):
    """
    Save fold metrics (accuracy, AUC, F1s, etc.) to JSON or CSV.
    Handles NumPy arrays by converting to lists.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert non-serializable entries
    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.int64, np.float32, np.float64)):
            return float(o)
        return o

    cleaned = {k: convert(v) for k, v in metrics_dict.items()}

    if output_path.endswith(".json"):
        with open(output_path, "w") as f:
            json.dump(cleaned, f, indent=2)
    elif output_path.endswith(".csv"):
        pd.DataFrame([cleaned]).to_csv(output_path, index=False)
    else:
        raise ValueError("Unsupported file format. Use .json or .csv")


def save_confusion_matrix(cm, labels, output_path, title="Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    

def group_preds_by_session(y_preds, y_trues, session_ids):
    """
    Groups predictions and labels by session ID.
    Returns a dict of session_id -> (list of preds, list of trues)
    """
    grouped = defaultdict(lambda: {"preds": [], "trues": []})
    
    for preds, trues, session_id in zip(y_preds, y_trues, session_ids):
        grouped[session_id]["preds"].extend(preds.cpu().tolist())
        grouped[session_id]["trues"].extend(trues.cpu().tolist())
    
    return grouped


def compute_per_file_accuracy(session_dict):
    """
    Computes accuracy for each session (after removing -100 padding).
    Returns:
        dict: session_id -> accuracy
    """
    per_file_acc = {}
    for sid, data in session_dict.items():
        preds = np.array(data["preds"])
        trues = np.array(data["trues"])
        mask = trues != -100
        if np.any(mask):
            acc = (preds[mask] == trues[mask]).mean()
        else:
            acc = float("nan")
        per_file_acc[sid] = acc
    return per_file_acc


def compute_per_original_session_accuracy(session_dict):
    """
    Computes accuracy per original session by grouping segment-level predictions.

    Assumes session_id looks like: d01_P01_6_0_clip_1_stride_3
    Groups by: d01_P01_6_0_clip_1
    """
    grouped = defaultdict(lambda: {"preds": [], "trues": []})

    for sid, data in session_dict.items():
        base_sid = "_".join(sid.split("_")[:-2])  # remove _stride_#
        preds = np.array(data["preds"])
        trues = np.array(data["trues"])
        mask = trues != -100
        grouped[base_sid]["preds"].extend(preds[mask])
        grouped[base_sid]["trues"].extend(trues[mask])

    per_original_acc = {}
    for sid, data in grouped.items():
        preds = np.array(data["preds"])
        trues = np.array(data["trues"])
        if len(trues) > 0:
            acc = (preds == trues).mean()
        else:
            acc = float("nan")
        per_original_acc[sid] = acc

    return per_original_acc


def compute_metrics(y_true, y_pred, label_names=["0", "1", "2", "3", "4"], output_path=None):
    """Compute only accuracy and confusion matrix for current classification task.

    Returns a dict: {"accuracy": float, "labels": List[int], "confusion_matrix": List[List[int]]}
    """
    y_true = torch.cat(y_true).cpu().numpy()
    y_pred = torch.cat(y_pred).cpu().numpy()

    # Mask out padding (-100) if present
    mask = y_true != -100
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # Accuracy
    acc = accuracy_score(y_true, y_pred) if y_true.size > 0 else float("nan")

    # High/Low Accuracy: map 0,1 -> 0 (low), 2,3,4 -> 1 (high)
    if y_true.size > 0:
        y_true_hl = (y_true >= 2).astype(int)
        y_pred_hl = (y_pred >= 2).astype(int)
        acc_hl = accuracy_score(y_true_hl, y_pred_hl)
    else:
        acc_hl = float("nan")

    # Confusion matrix with fixed labels (0..4)
    labels_full = list(range(len(label_names)))  # -> [0,1,2,3,4]
    cm = confusion_matrix(y_true, y_pred, labels=labels_full)

    result = {
        "accuracy": float(acc),
        "binary_accuracy": float(acc_hl),
        "labels": labels_full,
        "confusion_matrix": cm.tolist()
    }
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"High/Low Accuracy: {acc_hl:.4f}")
    return result
    
    
def save_per_file_accuracy(per_file_acc_dict, output_path="results/per_file_accuracy.csv"):
    """
    Saves per-file accuracy to CSV (or JSON if .json extension is used).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if output_path.endswith(".json"):
        with open(output_path, "w") as f:
            json.dump(per_file_acc_dict, f, indent=2)
    else:
        df = pd.DataFrame([
            {"session_id": sid, "accuracy": acc}
            for sid, acc in per_file_acc_dict.items()
        ])
        df.to_csv(output_path, index=False)


def aggregate_fold_metrics(metrics_list, cm_dir = "results/confusion_matrices"):
    """
    Aggregate only accuracy and confusion matrices for current classification task.
    """
    accs = [m.get('accuracy') for m in metrics_list if 'accuracy' in m]

    all_y_true = []
    all_y_pred = []

    # Save individual confusion matrices with dynamic labels
    for i, m in enumerate(metrics_list):
        y_true = np.asarray(m["y_true"])  # 1D
        y_pred = np.asarray(m["y_pred"])  # 1D
        labels_present = sorted(set(y_true).union(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels_present)
        os.makedirs(cm_dir, exist_ok=True)
        save_confusion_matrix(cm,
                              labels=[str(l) for l in labels_present],
                              output_path=os.path.join(cm_dir, f"confusion_matrix_fold_{i+1}.png"),
                              title=f"Confusion Matrix - Fold {i+1}")
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

    # Aggregate predictions
    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)

    # labels_present_agg = sorted(set(all_y_true).union(all_y_pred))
    # cm_agg = confusion_matrix(all_y_true, all_y_pred, labels=labels_present_agg)
    # save_confusion_matrix(cm_agg,
    #                       labels=[str(l) for l in labels_present_agg],
    #                       output_path=os.path.join(cm_dir, "confusion_matrix_aggregated.png"),
    #                       title="Confusion Matrix - Aggregated")
    labels_full = list(range(5))  # 或函数参数传入
    cm_agg = confusion_matrix(all_y_true, all_y_pred, labels=labels_full)
    save_confusion_matrix(cm_agg, labels=[str(l) for l in labels_full], output_path=os.path.join(cm_dir, "confusion_matrix_aggregated.png"), title="Confusion Matrix - Aggregated")

    avg_accuracy = float(np.mean(accs)) if accs else float('nan')

    print("\n========== Aggregated Metrics (CrossValidation) ==========")
    print(f"Average Accuracy     : {avg_accuracy:.4f}")
    print("====================================================")

    return {
        "avg_accuracy": avg_accuracy,
        "confusion_matrix_aggregated": cm_agg.tolist(),
        "labels": [int(l) for l in labels_full]
    }
    
# Notes for metrics:
# Aggregated Metrics overall
# avg_accuracy: accuracy on average across all folds
# avg_auc: auc on average cross folds
# macro_f1: F1 score computed per class, then averaged
# weighted_f1: F1 per class, weighted by class support
# macro_precision: # TP / (TP + FP)
# macro_recall: TP / (TP + FN)

# === Aggregated Metrics per class ===
# o: {'precision': how many predicted samples were correct?, 
#     'recall': how many actual samples were detected?, 
#     f1-score': yhe harmonic mean of precision and recall, 
#     'support': total number of samples in this class}
# the same for b, s, bs

# accuracy: Overall accuracy across all samples

# macro avg: Equal-weighted average across all classes
# {'precision': (P_o + P_b + P_s + P_bs) / 4, 
#  'recall': (R_o + R_b + R_s + R_bs) / 4, 
#  'f1-score': (F1_o + F1_b + F1_s + F1_bs) / 4, 
#  'support': total number of samples across all classes}

# weighted avg: Support-weighted average (heavily influenced by class distribution)
# {'precision': (P_o * supp_o + P_b * supp_b + ...) / total_support, 
#  'recall': , 
#  'f1-score': , 
#  'support': }