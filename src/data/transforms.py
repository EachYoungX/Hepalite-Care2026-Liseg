import numpy as np
import torch
import random
import cv2


class Robust25DTransform:
    """Apply synchronized augmentations to five-channel 2.5D inputs."""

    def __init__(self, prob=0.5, bg_value=0.0):
        self.prob = prob
        self.bg_value = bg_value

    def __call__(self, image, mask, weight):
        """Transform image, mask, and confidence-weight arrays consistently."""
        C, H, W = image.shape

        # Restrict geometric augmentation to small rotations to preserve spatial priors.
        if random.random() < self.prob:
            angle = random.uniform(-15.0, 15.0)
            M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)

            for c in range(C):
                image[c] = cv2.warpAffine(
                    image[c],
                    M,
                    (W, H),
                    flags=cv2.INTER_LINEAR,
                    borderValue=self.bg_value,
                )

            mask[0] = cv2.warpAffine(
                mask[0], M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0
            )

            weight[0] = cv2.warpAffine(
                weight[0], M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0
            )

        # Simulate low-intensity lesions while retaining the original target mask.
        if random.random() < self.prob:
            y_indices, x_indices = np.where(mask[0] > 0)
            if len(x_indices) > 500:
                center_idx = random.randint(0, len(x_indices) - 1)
                cx, cy = x_indices[center_idx], y_indices[center_idx]

                r = random.randint(12, 32)

                for c in range(C):
                    cv2.circle(image[c], (cx, cy), r, float(self.bg_value), -1)

        if random.random() < self.prob:
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-0.08, 0.08)

            img_mask = image > self.bg_value
            image[img_mask] = image[img_mask] * alpha + beta
            image = np.clip(image, 0, 1)

        if random.random() < self.prob:
            gamma = random.uniform(0.75, 1.4)
            img_mask = image > self.bg_value
            image[img_mask] = np.power(image[img_mask], gamma)
            image = np.clip(image, 0, 1)

        if random.random() < 0.15:
            noise = np.random.normal(0, 0.015, image.shape).astype(np.float32)
            img_mask = image > self.bg_value
            image[img_mask] += noise[img_mask]
            image = np.clip(image, 0, 1)

        # Drop only contextual slices; retain the central target slice.
        if random.random() < 0.25:
            num_drops = random.choice([1, 2])
            context_channels = [0, 1, 3, 4]
            dropped_channels = random.sample(context_channels, num_drops)
            for c in dropped_channels:
                image[c] = self.bg_value

        return image, mask, weight
