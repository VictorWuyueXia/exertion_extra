import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_regression_from_csv(csv_path, save_dir="offline_plots", num_samples=10):
    os.makedirs(save_dir, exist_ok=True)

    # Load the CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from: {csv_path}")

    # Randomly pick sample indices
    seed = 66
    random.seed(seed)
    sample_indices = random.sample(range(len(df)), min(num_samples, len(df)))

    for i in sample_indices:
        row = df.iloc[i]
        session_id = row['session_id']

        # Parse space-separated lists into numpy arrays
        gt_sample = np.array([float(v) for v in row['y_true'].split()])
        pred_sample = np.array([float(v) for v in row['y_pred'].split()])
        T = len(gt_sample)

        # Replace -100 with NaN for ground truth
        gt_to_plot = gt_sample.copy()
        gt_to_plot[gt_to_plot == -100] = np.nan

        # Plotting
        plt.figure(figsize=(10, 4))
        plt.plot(range(T), gt_to_plot, drawstyle='steps-mid', label="Ground Truth", alpha=0.7)
        plt.plot(range(T), pred_sample, drawstyle='steps-mid', linestyle="--", label="Prediction", alpha=0.7)
        plt.title(f"{session_id} - Ground Truth vs Prediction")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.ylim(-0.5, 3.5)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{session_id}_detailed.png"))
        plt.close()


plot_regression_from_csv("/work/users/y/u/yuyuwang/cardio_pause/experiments/fused_Focal_switch_vgg16_input_channels2_lr0.001_embed_layers12_2025-08-06/metrics/fold_1/test_regression_predictions.csv", 
                         save_dir="/work/users/y/u/yuyuwang/cardio_pause/experiments/fused_Focal_switch_vgg16_input_channels2_lr0.001_embed_layers12_2025-08-06/metrics/fold_1/pred_label_plots", 
                         num_samples=50)