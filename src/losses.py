import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFocalRegressionLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None, delta=0.5):
        """
        Simplified Focal Loss for regression - easier to tune
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta
        
        if class_weights is None:
            # Based on typical imbalance: 'o' most common, 's' least common
            self.class_weights = {0.0: 1.0, 1.0: 4.0, 2.0: 2.0, 3.0: 2.5}
        else:
            self.class_weights = class_weights
    
    def forward(self, y_pred, y_true):
        y_pred = y_pred.float()
        y_true = y_true.float()
        
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)
        
        # Class weights
        true_classes = y_true_flat.round().clamp(0, 3)
        weights = torch.tensor([
            self.class_weights[cls.item()] for cls in true_classes
        ], device=y_true.device)
        
        # Distance-based "difficulty" - further predictions are harder
        distances = torch.abs(y_pred_flat - y_true_flat)
        difficulties = torch.clamp(distances / self.delta, min=0.0, max=1.0)
        
        # Focal weighting: focus more on hard examples
        focal_weights = torch.pow(difficulties, self.gamma)
        
        # Base loss (smooth L1)
        base_loss = F.smooth_l1_loss(y_pred_flat, y_true_flat, 
                                   reduction='none', beta=self.delta)
        
        # Apply focal and class weighting
        focal_loss = self.alpha * focal_weights * base_loss * weights
        
        return focal_loss.mean()
    

class PenaltyMSELoss(nn.Module):
    def __init__(self, penalty_weights=None):
        """
        Args:
            penalty_weights (dict): e.g., {0.0: 1.0, 1.0: 5.0, 2.0: 5.0, 3.0: 5.0}
        """
        super().__init__()
        if penalty_weights is None:
            # Default: no penalty
            self.penalty_weights = {0.0: 1.0, 1.0: 5.0, 2.0: 5.0, 3.0: 5.0}
        else:
            self.penalty_weights = penalty_weights

    def forward(self, y_pred, y_true):
        y_pred = y_pred.float()
        y_true = y_true.float()

        # Flatten
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # Get weights based on rounded ground truth
        class_ids = y_true.round().clamp(0, 3)  # ensure in range
        weights = torch.tensor(
            [self.penalty_weights[cls.item()] for cls in class_ids],
            device=y_true.device
        )

        loss = F.mse_loss(y_pred, y_true, reduction='none')  # (N,)
        weighted_loss = (loss * weights).mean()
        return weighted_loss


class PenaltyHuberLoss(nn.Module):
    def __init__(self, delta=0.5, penalty_weights=None):
        """
        Args:
            delta (float): Huber threshold
            penalty_weights (dict): e.g., {0.0: 1.0, 1.0: 5.0, 2.0: 5.0, 3.0: 5.0}
        """
        super().__init__()
        self.delta = delta
        if penalty_weights is None:
            self.penalty_weights = {0.0: 1.0, 1.0: 5.0, 2.0: 5.0, 3.0: 5.0}
        else:
            self.penalty_weights = penalty_weights

    def forward(self, y_pred, y_true):
        y_pred = y_pred.float()
        y_true = y_true.float()

        # Flatten
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        class_ids = y_true.round().clamp(0, 3)
        weights = torch.tensor(
            [self.penalty_weights[cls.item()] for cls in class_ids],
            device=y_true.device
        )

        diff = y_pred - y_true
        abs_diff = torch.abs(diff)
        huber_loss = torch.where(
            abs_diff < self.delta,
            0.5 * diff ** 2,
            self.delta * (abs_diff - 0.5 * self.delta)
        )
        weighted_loss = (huber_loss * weights).mean()
        return weighted_loss
    

class PenaltyCrossEntropyLoss(nn.Module):
    def __init__(self, penalty_matrix, weight=None, ignore_index=-100):
        super().__init__()
        self.penalty_matrix = torch.tensor(penalty_matrix, dtype=torch.float32)
        self.class_weight = torch.tensor(weight, dtype=torch.float32) if weight is not None else None
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        device = logits.device
        logits = logits.to(device)
        targets = targets.to(device)
        penalty_matrix = self.penalty_matrix.to(device)
        
        log_probs = F.log_softmax(logits, dim=1)
        nll = F.nll_loss(log_probs, targets, reduction='none', ignore_index=self.ignore_index)
        
        if self.class_weight is not None:
            class_weight = self.class_weight.to(device)
            nll = nll * class_weight[targets]

        # Compute weighted penalty
        probs = F.softmax(logits, dim=1)
        penalties = penalty_matrix[targets]  # shape: (N, C)
        weighted_penalty = (penalties * probs).sum(dim=1)

        total_loss = nll + weighted_penalty

        # Mask ignored targets
        mask = targets != self.ignore_index
        return total_loss[mask].mean()


class MAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return F.l1_loss(y_pred.float(), y_true.float())


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return F.mse_loss(y_pred.float(), y_true.float())


class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        return F.huber_loss(y_pred.float(), y_true.float(), delta=self.delta)


