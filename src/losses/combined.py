import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryLoss2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "weight_x",
            torch.tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]]),
        )
        self.register_buffer(
            "weight_y",
            torch.tensor([[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]]),
        )

    def forward(self, pred, target, weight_map=None):
        pred_sigmoid = torch.sigmoid(pred)
        grad_pred_x = F.conv2d(pred_sigmoid, self.weight_x, padding=1)
        grad_pred_y = F.conv2d(pred_sigmoid, self.weight_y, padding=1)
        grad_target_x = F.conv2d(target.float(), self.weight_x, padding=1)
        grad_target_y = F.conv2d(target.float(), self.weight_y, padding=1)

        mag_pred = torch.sqrt(grad_pred_x**2 + grad_pred_y**2 + 1e-6)
        mag_target = torch.sqrt(grad_target_x**2 + grad_target_y**2 + 1e-6)

        loss_map = F.l1_loss(mag_pred, mag_target, reduction="none")
        if weight_map is not None:
            loss_map = loss_map * weight_map

        return loss_map.mean()


class TverskyLoss(nn.Module):
    """Asymmetric Tversky loss with optional uncertainty weighting."""

    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets, weight_map=None):
        probs = torch.sigmoid(logits)

        if weight_map is not None:
            tp = (probs * targets * weight_map).sum(dim=(1, 2, 3))
            fp = (probs * (1.0 - targets) * weight_map).sum(dim=(1, 2, 3))
            fn = ((1.0 - probs) * targets * weight_map).sum(dim=(1, 2, 3))
        else:
            tp = (probs * targets).sum(dim=(1, 2, 3))
            fp = (probs * (1.0 - targets)).sum(dim=(1, 2, 3))
            fn = ((1.0 - probs) * targets).sum(dim=(1, 2, 3))

        tversky_score = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1.0 - tversky_score.mean()


class HepaLiteCombinedLoss25D(nn.Module):
    def __init__(self):
        super().__init__()
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)
        self.boundary = BoundaryLoss2D()

    def _empty_slice_penalty(self, logits, targets):
        empty_mask = targets.sum(dim=(1, 2, 3)) < 0.5
        if not empty_mask.any():
            return logits.new_tensor(0.0)
        return torch.sigmoid(logits[empty_mask]).mean()

    def forward(self, logits, targets, weight_map=None, is_pseudo=False):
        loss_tversky = self.tversky(
            logits, targets, weight_map=weight_map if is_pseudo else None
        )

        if is_pseudo:
            loss_bce = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            if weight_map is not None:
                loss_bce = (loss_bce * weight_map).mean()
            else:
                loss_bce = loss_bce.mean()
            loss_empty = self._empty_slice_penalty(logits, targets)
            return 0.8 * loss_tversky + 0.2 * loss_bce + 0.2 * loss_empty
        else:
            loss_bce = F.binary_cross_entropy_with_logits(logits, targets)
            loss_boundary = self.boundary(logits, targets)

            loss_empty = self._empty_slice_penalty(logits, targets)
            return (
                0.6 * loss_tversky
                + 0.3 * loss_bce
                + 0.1 * loss_boundary
                + 0.5 * loss_empty
            )


class HepaLiteStudentLoss25D(nn.Module):
    """Topology-oriented loss with deep supervision for the student model."""

    def __init__(self):
        super().__init__()
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)
        self.ds_weights = [0.4, 0.3, 0.2, 0.1]

    def _empty_slice_penalty(self, logits, targets):
        empty_mask = targets.sum(dim=(1, 2, 3)) < 0.5
        if not empty_mask.any():
            return logits.new_tensor(0.0)
        return torch.sigmoid(logits[empty_mask]).mean()

    def _calculate_single_level_loss(
        self, logits, targets, weight_map=None, is_pseudo=False
    ):
        loss_tversky = self.tversky(
            logits, targets, weight_map=weight_map if is_pseudo else None
        )

        if is_pseudo:
            loss_bce = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            if weight_map is not None:
                loss_bce = (loss_bce * weight_map).mean()
            else:
                loss_bce = loss_bce.mean()
            loss_empty = self._empty_slice_penalty(logits, targets)
            return 0.8 * loss_tversky + 0.2 * loss_bce + 0.2 * loss_empty
        else:
            loss_bce = F.binary_cross_entropy_with_logits(logits, targets)
            loss_empty = self._empty_slice_penalty(logits, targets)
            return 0.7 * loss_tversky + 0.3 * loss_bce + 0.5 * loss_empty

    def forward(self, logits, targets, weight_map=None, is_pseudo=False):
        if isinstance(logits, list):
            total_loss = 0.0
            num_levels = min(len(logits), len(self.ds_weights))
            for i in range(num_levels):
                level_loss = self._calculate_single_level_loss(
                    logits[i], targets, weight_map=weight_map, is_pseudo=is_pseudo
                )
                total_loss += self.ds_weights[i] * level_loss
            return total_loss
        else:
            return self._calculate_single_level_loss(
                logits, targets, weight_map=weight_map, is_pseudo=is_pseudo
            )
