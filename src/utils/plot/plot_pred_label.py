import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import os

def plot_all_preds(
    pred_csv_file,
    output_dir,
    sr_label=20
):
    os.makedirs(output_dir, exist_ok=True)

    # Load prediction CSV
    df_pred = pd.read_csv(pred_csv_file)

    for idx, row in df_pred.iterrows():
        session_id = row['session_id']
        try:
            y_pred = np.array(ast.literal_eval(row['y_pred']))
            y_true = np.array(ast.literal_eval(row['y_true']))

            time = np.arange(len(y_pred)) / sr_label

            # Plot
            plt.figure(figsize=(12, 3))
            plt.plot(time, y_pred, drawstyle='steps-mid', color='red', label='Predicted')
            plt.plot(time, y_true[:len(y_pred)], drawstyle='steps-mid', color='blue', linestyle='--', alpha=0.6, label='True (CSV)')

            plt.ylabel("Class")
            plt.xlabel("Time (s)")
            plt.title(f"Session: {session_id}")
            plt.yticks([0, 1, 2, 3], ["o", "b", "s", "bs"])
            plt.grid(True)
            plt.legend()

            # Save plot
            output_path = os.path.join(output_dir, f"{session_id}.png")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"[Saved] {output_path}")

        except Exception as e:
            print(f"[Error] {session_id}: {e}")


# Example usage
plot_all_preds(
    pred_csv_file="/work/users/y/u/yuyuwang/cardio_pause/experiments/PWCE_vgg16_input_channels1_lr0.001_embed_layers8_2025-08-06/metrics/raw_test_segment_preds_fold_1.csv",
    output_dir="/work/users/y/u/yuyuwang/cardio_pause/experiments/PWCE_vgg16_input_channels1_lr0.001_embed_layers8_2025-08-06/metrics/pred_label_plots"
)