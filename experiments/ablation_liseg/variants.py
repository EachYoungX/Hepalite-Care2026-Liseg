import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.models import AxialFactorizedBlock2D, HepaLiteStudent25D


class ControlledStudent(nn.Module):
    """Student variant with controlled context, CoordConv, and SE switches."""

    def __init__(self, context_slices=5, use_coordconv=True, use_se=True):
        super().__init__()
        if context_slices not in (1, 3, 5):
            raise ValueError("context_slices must be one of 1, 3, or 5")
        self.context_slices = context_slices
        self.network = HepaLiteStudent25D(
            in_channels=context_slices,
            out_classes=1,
            init_features=16,
            deep_supervision=True,
        )
        if not use_coordconv:
            self.network.add_coords = nn.Identity()
            self.network.stem = AxialFactorizedBlock2D(context_slices, 16)
        if not use_se:
            for module in self.network.modules():
                if isinstance(module, AxialFactorizedBlock2D):
                    module.se = nn.Identity()

    def _select_context(self, x):
        if x.shape[1] == self.context_slices:
            return x
        center = x.shape[1] // 2
        radius = self.context_slices // 2
        return x[:, center - radius : center + radius + 1]

    def forward(self, x):
        return self.network(self._select_context(x))


class ContextDatasetView(torch.utils.data.Dataset):
    """Expose only the requested center context while retaining dataset sampling."""

    def __init__(self, dataset, context_slices):
        self.dataset = dataset
        self.context_slices = context_slices

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, index):
        image, mask, weight, stage, is_pseudo = self.dataset[index]
        center = image.shape[0] // 2
        radius = self.context_slices // 2
        image = image[center - radius : center + radius + 1]
        return image, mask, weight, stage, is_pseudo

    def set_epoch(self, epoch, pseudo_ratio=20):
        self.dataset.set_epoch(epoch, pseudo_ratio=pseudo_ratio)



class SymmetricDiceBCELoss25D(nn.Module):
    """Symmetric Dice+BCE control with the same deep-supervision weights."""

    def __init__(self):
        super().__init__()
        self.ds_weights = [0.4, 0.3, 0.2, 0.1]

    @staticmethod
    def _single(logits, targets):
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum(dim=(1, 2, 3))
        denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = 1.0 - ((2.0 * intersection + 1e-5) / (denominator + 1e-5)).mean()
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        empty = targets.sum(dim=(1, 2, 3)) < 0.5
        empty_penalty = probabilities[empty].mean() if empty.any() else logits.new_tensor(0.0)
        return 0.7 * dice + 0.3 * bce + 0.5 * empty_penalty

    def forward(self, logits, targets, weight_map=None, is_pseudo=False):
        if isinstance(logits, list):
            return sum(
                self.ds_weights[index] * self._single(level_logits, targets)
                for index, level_logits in enumerate(logits[: len(self.ds_weights)])
            )
        return self._single(logits, targets)
