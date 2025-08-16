import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional
from metrics import compute_metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from models import get_model
from losses import PenaltyCrossEntropyLoss


class CoralHead(nn.Module):
    """CORAL head: shared weight vector with per-threshold biases.

    Given an input representation of shape (B, D), produces (B, K-1) logits as:
    logits = x @ w + b, where w is shared across thresholds and b has length K-1.
    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc_shared = nn.Linear(input_dim, 1, bias=False)  # shared weights
        # Unconstrained parameters reparameterized to ordered thresholds via cumulative softplus
        self.delta = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x):
        # x: (B, D) → base: (B, 1)
        base = self.fc_shared(x)
        # thresholds: non-decreasing of shape (K-1,)
        thresholds = torch.cumsum(F.softplus(self.delta), dim=0)
        # logits for P(y > r): base - theta_r ensures monotonic probabilities across r
        return base - thresholds  # broadcast to (B, K-1)


def make_coral_targets(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    # y in [0..K-1], return cumulative targets of shape (B, K-1): t[:,k] = 1 if y > k else 0
    K = num_classes
    thresholds = torch.arange(K - 1, device=y.device).unsqueeze(0)  # (1, K-1)
    return (y.unsqueeze(1) > thresholds).float()
from metrics import group_preds_by_session, compute_per_file_accuracy, save_per_file_accuracy, save_fold_metrics, compute_per_original_session_accuracy, save_confusion_matrix

def train_one_epoch(model, dataloader, criterion, optimizer, device, model_type, feature_type, use_coral=False, coral_head: Optional[CoralHead] = None, num_classes: int = 5):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        x = batch[feature_type].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)  # shape: (B,) scalar labels
        lengths = batch["lengths"]

        if model_type == "mlp":
            # Pool temporal dimension by mean, produce (B, F)
            x_pooled = x.mean(dim=1)
            logits = model(x_pooled)  # (B, C_repr)
            if use_coral:
                coral_logits = coral_head(logits)  # (B, K-1)
                target_cum = make_coral_targets(y, num_classes)
                loss = criterion(coral_logits, target_cum)
            else:
                loss = criterion(logits, y)

        elif model_type == "lstm":
            logits = model(x, lengths)  # (B, T, C)
            # Average over time for scalar supervision
            logits_pooled = logits.mean(dim=1)  # (B, C)
            if use_coral:
                coral_logits = coral_head(logits_pooled)
                target_cum = make_coral_targets(y, num_classes)
                loss = criterion(coral_logits, target_cum)
            else:
                loss = criterion(logits_pooled, y)
            
        elif model_type == "tcnnlstm":
            logits, _, _, _ = model(x, lengths)  # (B, T, C)
            logits_pooled = logits.mean(dim=1)   # (B, C)
            if use_coral:
                coral_logits = coral_head(logits_pooled)
                target_cum = make_coral_targets(y, num_classes)
                loss = criterion(coral_logits, target_cum)
            else:
                loss = criterion(logits_pooled, y)

        elif model_type == "alexnet":
            x_img = x.permute(0, 2, 1).unsqueeze(1)  # (B, 1, F, T)
            logits = model(x_img)                    # (B, T, C)
            logits_pooled = logits.mean(dim=1)       # (B, C)
            if use_coral:
                coral_logits = coral_head(logits_pooled)
                target_cum = make_coral_targets(y, num_classes)
                loss = criterion(coral_logits, target_cum)
            else:
                loss = criterion(logits_pooled, y)

        elif model_type == "grunet":
            logits = model(x, return_repr=False)  # (B, T, C)
            logits_pooled = logits.mean(dim=1)    # (B, C)
            if use_coral:
                coral_logits = coral_head(logits_pooled)
                target_cum = make_coral_targets(y, num_classes)
                loss = criterion(coral_logits, target_cum)
            else:
                loss = criterion(logits_pooled, y)
            
        elif model_type == "vgg16":
            x_img = x.permute(0, 2, 1).unsqueeze(1)
            logits = model(x_img)                    # (B, T, C)
            logits_pooled = logits.mean(dim=1)       # (B, C)
            if use_coral:
                coral_logits = coral_head(logits_pooled)
                target_cum = make_coral_targets(y, num_classes)
                loss = criterion(coral_logits, target_cum)
            else:
                loss = criterion(logits_pooled, y)
            
        else:
           raise ValueError("Unsupported model type")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, model_type, feature_type, use_coral=False, coral_head: Optional[CoralHead] = None, num_classes: int = 5):
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []
    all_sessions = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch[feature_type].to(device, non_blocking=True)
            y = batch["labels"].to(device)  # shape: (B,)
            lengths = batch["lengths"]

            if model_type == "mlp":
                x_pooled = x.mean(dim=1)  # (B, F)
                logits = model(x_pooled)  # (B, C)
                if use_coral:
                    coral_logits = coral_head(logits)
                    target_cum = make_coral_targets(y, num_classes)
                    loss = criterion(coral_logits, target_cum)
                    preds = (torch.sigmoid(coral_logits) > 0.5).sum(dim=1)
                else:
                    loss = criterion(logits, y)
                    preds = torch.argmax(logits, dim=1)
                for i in range(preds.size(0)):
                    all_preds.append(preds[i].unsqueeze(0).detach().cpu())
                    all_labels.append(y[i].unsqueeze(0).detach().cpu())
                    all_sessions.append(batch["session_ids"][i])

            elif model_type == "lstm":
                logits = model(x, lengths)   # (B, T, C)
                logits_pooled = logits.mean(dim=1)  # (B, C)
                if use_coral:
                    coral_logits = coral_head(logits_pooled)
                    target_cum = make_coral_targets(y, num_classes)
                    loss = criterion(coral_logits, target_cum)
                    preds = (torch.sigmoid(coral_logits) > 0.5).sum(dim=1)
                else:
                    loss = criterion(logits_pooled, y)
                    preds = torch.argmax(logits_pooled, dim=1)
                for i in range(preds.size(0)):
                    all_preds.append(preds[i].unsqueeze(0).detach().cpu())
                    all_labels.append(y[i].unsqueeze(0).detach().cpu())
                    all_sessions.append(batch["session_ids"][i])

            elif model_type == "tcnnlstm":
                logits, _, _, _ = model(x, lengths)  # (B, T, C)
                logits_pooled = logits.mean(dim=1)    # (B, C)
                if use_coral:
                    coral_logits = coral_head(logits_pooled)
                    target_cum = make_coral_targets(y, num_classes)
                    loss = criterion(coral_logits, target_cum)
                    preds = (torch.sigmoid(coral_logits) > 0.5).sum(dim=1)
                else:
                    loss = criterion(logits_pooled, y)
                    preds = torch.argmax(logits_pooled, dim=1)
                for i in range(preds.size(0)):
                    all_preds.append(preds[i].unsqueeze(0).detach().cpu())
                    all_labels.append(y[i].unsqueeze(0).detach().cpu())
                    all_sessions.append(batch["session_ids"][i])

            elif model_type == "alexnet":
                x_img = x.permute(0, 2, 1).unsqueeze(1)  # (B, 1, F, T)
                logits = model(x_img)                    # (B, T, C)
                logits_pooled = logits.mean(dim=1)       # (B, C)
                if use_coral:
                    coral_logits = coral_head(logits_pooled)
                    target_cum = make_coral_targets(y, num_classes)
                    loss = criterion(coral_logits, target_cum)
                    preds = (torch.sigmoid(coral_logits) > 0.5).sum(dim=1)
                else:
                    loss = criterion(logits_pooled, y)
                    preds = torch.argmax(logits_pooled, dim=1)
                for i in range(preds.size(0)):
                    all_preds.append(preds[i].unsqueeze(0).detach().cpu())
                    all_labels.append(y[i].unsqueeze(0).detach().cpu())
                    all_sessions.append(batch["session_ids"][i])
                    
            elif model_type == "grunet":
                logits = model(x, return_repr=False)   # (B, T, C)
                logits_pooled = logits.mean(dim=1)     # (B, C)
                if use_coral:
                    coral_logits = coral_head(logits_pooled)
                    target_cum = make_coral_targets(y, num_classes)
                    loss = criterion(coral_logits, target_cum)
                    preds = (torch.sigmoid(coral_logits) > 0.5).sum(dim=1)
                else:
                    loss = criterion(logits_pooled, y)
                    preds = torch.argmax(logits_pooled, dim=1)
                for i in range(preds.size(0)):
                    all_preds.append(preds[i].unsqueeze(0).detach().cpu())
                    all_labels.append(y[i].unsqueeze(0).detach().cpu())
                    all_sessions.append(batch["session_ids"][i])
                    
            elif model_type == "vgg16":
                x_img = x.permute(0, 2, 1).unsqueeze(1)  # (B, 1, F, T)
                logits = model(x_img)                   # (B, T, C)
                logits_pooled = logits.mean(dim=1)      # (B, C)
                if use_coral:
                    coral_logits = coral_head(logits_pooled)
                    target_cum = make_coral_targets(y, num_classes)
                    loss = criterion(coral_logits, target_cum)
                    preds = (torch.sigmoid(coral_logits) > 0.5).sum(dim=1)
                else:
                    loss = criterion(logits_pooled, y)
                    preds = torch.argmax(logits_pooled, dim=1)
                for i in range(preds.size(0)):
                    all_preds.append(preds[i].unsqueeze(0).detach().cpu())
                    all_labels.append(y[i].unsqueeze(0).detach().cpu())
                    all_sessions.append(batch["session_ids"][i])

            else:
                raise ValueError("Unsupported model type")

            total_loss += loss.item()

    # For scalar supervision, collect predictions per sample
    return total_loss / len(dataloader), all_preds, all_labels, all_sessions


def train_model(model_type, model_params, feature_type, input_dim, dataloaders, device, num_epochs=20, early_stopping=True, lr_scheduler=True, log_file=None, criterion=None, use_coral=False, num_classes=5):
    model_hparams = {k: v for k, v in model_params.items() if k != "lr"}  # Clean out learning rate
    model_output_dim = (num_classes - 1) if use_coral else num_classes
    model = get_model(model_type, model_hparams, input_dim=input_dim, output_dim=model_output_dim).to(device)

    coral_head = CoralHead(model_output_dim, num_classes).to(device) if use_coral else None

    params = list(model.parameters()) + (list(coral_head.parameters()) if coral_head is not None else [])
    optimizer = torch.optim.Adam(params, lr=model_params.get("lr", 1e-4))
    criterion = criterion

    scheduler = None
    if lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss = train_one_epoch(
            model,
            dataloaders["train"],
            criterion,
            optimizer,
            device,
            model_type=model_type,
            feature_type=feature_type,
            use_coral=use_coral,
            coral_head=coral_head,
            num_classes=num_classes
        )
        val_loss, val_preds, val_labels, _ = evaluate(
        model,
        dataloaders["val"],
        criterion,
        device,
        model_type=model_type,
        feature_type=feature_type,
        use_coral=use_coral,
        coral_head=coral_head,
        num_classes=num_classes
        )
        

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if log_file:
            log_file.write(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}\n")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if lr_scheduler:
            prev_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != prev_lr:
                msg = f"[LR Scheduler] Epoch {epoch+1}: LR changed from {prev_lr:.6f} → {new_lr:.6f}"
                print(msg)
                if log_file:
                    log_file.write(msg + "\n")


        min_epochs = 10
        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience and epoch + 1 >= min_epochs:
                    print("Early stopping triggered.")
                    break

    return model, coral_head, train_losses, val_losses


def plot_training_curves(train_losses, val_losses, loss_plt_dir="results/loss_curves", fold_idx=0):
    os.makedirs(loss_plt_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Val Loss", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training & Validation Loss (Fold {fold_idx + 1})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    filename = f"train_val_loss_fold_{fold_idx + 1}.png"
    plt.savefig(os.path.join(loss_plt_dir, filename))
    plt.close()

    
def run_experiment(train_loader, val_loader, test_loader, input_dim, epochs=10, device='cpu', loss_plt_dir=None, per_file_dir=None, metrics_dir=None, cm_dir=None, fold_idx=0, model_type="mlp", feature_type=None, **model_param_dict):
    
    print("Device:", device)
    model_params = model_param_dict.get(f"{model_type}_params", None)
    
    # Loss selection
    use_coral = model_param_dict.get("use_coral", False)
    num_classes = model_param_dict.get("num_classes", 5)
    if use_coral:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        penalty_matrix = [
        [0.0, 3.0, 3.0, 3.0],
        [2.0, 0.0, 5.0, 2.0],
        [2.0, 5.0, 0.0, 2.0],
        [2.0, 2.0, 2.0, 0.0]
        ]
        weight_tensor = torch.tensor([0.24, 6.24, 4.65, 0.91], dtype=torch.float32).to(device)
        criterion = PenaltyCrossEntropyLoss(penalty_matrix, weight=weight_tensor)

    model, coral_head, train_losses, val_losses = train_model(
    model_type=model_type,
    feature_type = feature_type,
    model_params=model_params,
    dataloaders={"train": train_loader, "val": val_loader},
    input_dim=input_dim,
    device=device,
    num_epochs=epochs,
    early_stopping=True,
    lr_scheduler=True,
    criterion=criterion,
    use_coral=use_coral,
    num_classes=num_classes
    )
    print("Model on:", next(model.parameters()).device)

    # Evaluate on validation set to compute final val metrics
    val_loss, val_preds, val_labels, val_sessions = evaluate(model, val_loader, criterion, device, model_type, feature_type, use_coral=use_coral, coral_head=coral_head, num_classes=num_classes)

    # Flatten preds for smoothing + merging
    val_preds_flat = [p.cpu().numpy().tolist() for p in val_preds]
    val_labels_flat = [t.cpu().numpy().tolist() for t in val_labels]

    # Skip temporal smoothing and segment-level metrics in scalar-label classification

    
    for pred_type, val_preds_cur in [("raw", val_preds)]:
        val_grouped = group_preds_by_session(val_preds_cur, val_labels, val_sessions)
        val_per_file_acc = compute_per_file_accuracy(val_grouped)
        save_per_file_accuracy(val_per_file_acc, output_path=os.path.join(per_file_dir, f"{pred_type}_val_fold_{fold_idx + 1}.csv"))
        
        val_per_original_acc = compute_per_original_session_accuracy(val_grouped)
        save_per_file_accuracy(val_per_original_acc, output_path=os.path.join(per_file_dir, f"{pred_type}_val_fold_{fold_idx + 1}_original_session.csv"))

        y_val_true = torch.cat(val_labels).cpu().numpy()
        y_val_pred = torch.cat(val_preds_cur).cpu().numpy()
        acc = accuracy_score(y_val_true, y_val_pred)
        acc_hl = accuracy_score((y_val_true >= 2).astype(int), (y_val_pred >= 2).astype(int))

        labels_present = sorted(set(y_val_true).union(y_val_pred))
        cm_val = confusion_matrix(y_val_true, y_val_pred, labels=labels_present)
        val_cm_path = os.path.join(cm_dir, f"{pred_type}_val_confusion_matrix_fold_{fold_idx + 1}.png")
        save_confusion_matrix(cm_val, labels=[str(l) for l in labels_present], output_path=val_cm_path, title=f"{pred_type.upper()} val_confusion_matrix_fold_{fold_idx + 1}")

        print(f"[{pred_type.upper()}] Fold {fold_idx+1} | Val Acc: {acc:.3f} | High/Low Acc: {acc_hl:.3f}")


    test_loss, test_preds, test_labels, test_sessions = evaluate(model, test_loader, criterion, device, model_type, feature_type, use_coral=use_coral, coral_head=coral_head, num_classes=num_classes)
    
    test_preds_flat = [p.cpu().numpy().tolist() for p in test_preds]
    test_labels_flat = [t.cpu().numpy().tolist() for t in test_labels]
    
    # Skip segment-level metrics for scalar labels

    
    for pred_type, test_preds_cur in [("raw", test_preds)]:
        test_grouped = group_preds_by_session(test_preds_cur, test_labels, test_sessions)
        
        test_per_file_acc = compute_per_file_accuracy(test_grouped)
        save_per_file_accuracy(test_per_file_acc, output_path=os.path.join(per_file_dir, f"{pred_type}_test_fold_{fold_idx + 1}.csv"))
        
        test_per_original_acc = compute_per_original_session_accuracy(test_grouped)
        save_per_file_accuracy(test_per_original_acc, output_path=os.path.join(per_file_dir, f"{pred_type}_test_fold_{fold_idx + 1}_original_session.csv"))
        
        y_test_true = torch.cat(test_labels).cpu().numpy()
        y_test_pred = torch.cat(test_preds_cur).cpu().numpy()
        accuracy = accuracy_score(y_test_true, y_test_pred)
        acc_hl = accuracy_score((y_test_true >= 2).astype(int), (y_test_pred >= 2).astype(int))
        print(f"[{pred_type.upper()}] Fold {fold_idx+1} | Test Acc: {accuracy:.3f} | High/Low Acc: {acc_hl:.3f}")

    print("\n[Validation Performance]")
    val_report_dict = compute_metrics(val_labels, val_preds, label_names=["0","1","2","3","4"])



    val_report_path = os.path.join(metrics_dir, f"classification_report_val_fold_{fold_idx + 1}.json")
    with open(val_report_path, "w") as f:
        json.dump(val_report_dict, f, indent=2)


    print("\n[Test Performance]")
    report_dict = compute_metrics(test_labels, test_preds, label_names=["0","1","2","3","4"])

    report_path = os.path.join(metrics_dir, f"classification_report_test_fold_{fold_idx + 1}.json")
    
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    plot_training_curves(train_losses, val_losses, loss_plt_dir, fold_idx)
    
    # Save test metrics for current fold (no AUC in current setup)
    save_fold_metrics({
        "accuracy": accuracy,
        "binary_accuracy": acc_hl,
        "fold_idx": fold_idx,
        "model_params": model_params,
        "model_type": model_type,
        "feature_type": feature_type,
        "y_true": y_test_true,
        "y_pred": y_test_pred,
        "test_per_file_acc": test_per_file_acc
        
    }, output_path= os.path.join(metrics_dir, f"test_metrics_fold_{fold_idx + 1}.csv"))


    return {
        "accuracy": accuracy,
        "binary_accuracy": acc_hl,
        "y_true": y_test_true,
        "y_pred": y_test_pred,
        "test_per_file_acc": test_per_file_acc
    }


# def evaluate(model, dataloader, criterion, device, model_type, feature_type="acoustic"):
#     model.eval()
    
#     total_loss = 0
#     all_preds = []
#     all_labels = []
#     all_sessions = []
    
#     with torch.no_grad():
#         for batch in dataloader:
#             x = batch[feature_type].to(device)
#             y = batch["labels"].to(device)
#             lengths = batch["lengths"]

#             if model_type == "mlp":
#                 x_flat = []
#                 y_flat = []
#                 for i in range(x.size(0)):
#                     valid_len = min(x[i].size(0), y[i].size(0), lengths[i])
#                     x_flat.append(x[i, :valid_len])
#                     y_flat.append(y[i, :valid_len])
                    
#                 x_flat = torch.cat(x_flat, dim=0)
#                 y_flat = torch.cat(y_flat, dim=0).long()

#                 # Ensure equal length
#                 assert x_flat.shape[0] == y_flat.shape[0], f"Mismatch: x={x_flat.shape}, y={y_flat.shape}"

#                 logits = model(x_flat)
#                 loss = criterion(logits, y_flat)

#                 mask = y_flat != -100
#                 logits_masked = logits[mask]
#                 y_masked = y_flat[mask]

#             elif model_type == "lstm":
#                 logits = model(x, lengths)
#                 B, T, C = logits.shape
#                 logits_flat = logits.view(-1, C)
#                 y_flat = y.view(-1).long()
#                 loss = criterion(logits_flat, y_flat)

#                 mask = y_flat != -100
#                 logits_masked = logits_flat[mask]
#                 y_masked = y_flat[mask]


#             else:
#                 raise ValueError("Unsupported model type")

#             preds = torch.argmax(logits_masked, dim=1)
            
#             # Split predictions and labels per session
#             start = 0
#             for i, seq_len in enumerate(lengths):
#                 valid_len = (y[i, :seq_len] != -100).sum().item()
#                 end = start + valid_len

#                 all_preds.append(preds[start:end])
#                 all_labels.append(y_masked[start:end])
#                 all_sessions.append(batch["session_ids"][i])

#                 start = end
                
#             total_loss += loss.item()

#     return total_loss / len(dataloader), all_preds, all_labels, all_sessions
