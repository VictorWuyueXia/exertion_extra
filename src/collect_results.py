#!/usr/bin/env python3
import os
import sys
import glob
import yaml
import json
import ast
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from metrics import save_confusion_matrix


def read_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def pick_latest_dir(dirs: list[str]) -> Optional[str]:
    if not dirs:
        return None
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        return None
    return max(dirs, key=lambda p: os.path.getmtime(p))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_list_column(cell: str) -> list:
    try:
        return ast.literal_eval(cell)
    except Exception:
        # Fallback: try to load as JSON
        try:
            return json.loads(cell)
        except Exception:
            raise ValueError(f"Unable to parse list from cell: {cell[:100]}...")


def collect_logs_and_confmat(config_path: str) -> Tuple[str, str]:
    # Resolve paths and naming
    cfg = read_yaml(config_path)
    model = cfg["experiment"]["model"]
    input_features = cfg["experiment"]["input_features"]
    feature_str = "_".join(input_features)
    task_str = str(cfg["experiment"]["task"])

    # 判断是否包含embed特征，如果有则加上layer信息
    layer_str = ""
    if any("embed" in feat for feat in input_features):
        # 兼容不同配置格式
        selected_layers = cfg["experiment"].get("selected_wav2vec2_layers", None)
        if selected_layers is None:
            # 有些配置可能在别的地方
            selected_layers = cfg.get("selected_wav2vec2_layers", None)
        if selected_layers is not None:
            if isinstance(selected_layers, (list, tuple)):
                layer_str = "_layer" + "-".join(str(l) for l in selected_layers)
            else:
                layer_str = f"_layer{selected_layers}"
        else:
            layer_str = "_layer?"

    # 项目根目录
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # 结果输出目录，包含layer信息（如有）
    results_dir = os.path.join(project_root, "results", f"{model}_{task_str}_{feature_str}{layer_str}")
    ensure_dir(results_dir)

    # Experiments directory under Data (base_dir)
    base_dir = cfg["paths"]["base_dir"]
    experiments_root = os.path.join(base_dir, "experiments")
    prefix = f"new_split_cls_{model}_{feature_str}_"
    candidates = glob.glob(os.path.join(experiments_root, f"{prefix}*"))
    run_dir = pick_latest_dir(candidates)
    if not run_dir:
        raise FileNotFoundError(f"No experiment directory found under {experiments_root} with prefix {prefix}")

    config_dir = os.path.join(run_dir, "config")
    metrics_dir = os.path.join(run_dir, "metrics")

    # Gather logs
    log_paths = []
    for name in ["training_log.txt", "terminal_log.txt"]:
        p = os.path.join(config_dir, name)
        if os.path.isfile(p):
            log_paths.append(p)
    merged_log_path = os.path.join(results_dir, "train_log.txt")
    with open(merged_log_path, "w", encoding="utf-8") as out:
        out.write(f"# Experiment: {os.path.basename(run_dir)}\n\n")
        for p in log_paths:
            out.write(f"===== {os.path.basename(p)} =====\n")
            try:
                with open(p, "r", encoding="utf-8") as f:
                    out.write(f.read())
            except Exception as e:
                out.write(f"<Error reading {p}: {e}>\n")
            out.write("\n\n")

    # Build confusion matrix from saved test metrics
    test_metrics_files = sorted(glob.glob(os.path.join(metrics_dir, "test_metrics_fold_*.csv")))
    if not test_metrics_files:
        raise FileNotFoundError(f"No test metrics CSVs found in {metrics_dir}")
    latest_metrics = test_metrics_files[-1]
    df = pd.read_csv(latest_metrics)
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise ValueError(f"Metrics file {latest_metrics} missing y_true/y_pred columns")
    y_true = parse_list_column(df.loc[0, "y_true"]) if isinstance(df.loc[0, "y_true"], str) else list(df.loc[0, "y_true"])
    y_pred = parse_list_column(df.loc[0, "y_pred"]) if isinstance(df.loc[0, "y_pred"], str) else list(df.loc[0, "y_pred"])

    # labels_present = sorted(set(y_true) | set(y_pred))
    # cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    labels_full = list(range(5))  # 或从 config 读取
    cm = confusion_matrix(y_true, y_pred, labels=labels_full)
    
    # Save confusion matrix image
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    save_confusion_matrix(cm, labels=[str(l) for l in labels_full], output_path=cm_path, title=f"Confusion Matrix - {model} {feature_str}{layer_str}")

    print(f"Logs saved to: {merged_log_path}")
    print(f"Confusion matrix saved to: {cm_path}")
    return merged_log_path, cm_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/collect_results.py <path/to/config.yaml>")
        sys.exit(1)
    collect_logs_and_confmat(sys.argv[1])


