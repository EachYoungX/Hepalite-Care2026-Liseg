import os
import csv
import glob
import numpy as np
import torch
import matplotlib

matplotlib.rcParams["font.sans-serif"] = [
    "WenQuanYi Micro Hei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from scipy.ndimage import label, binary_fill_holes, binary_opening
from scipy.ndimage import gaussian_filter1d


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        exp_root,
        fold_num,
        val_threshold=0.5,  # Segmentation threshold used during validation.
        pseudo_ratio=20,  # Maximum number of pseudo-labeled cases.
        pure_real_baseline=False,  # Train using manually labeled cases only.
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.val_threshold = val_threshold
        self.pseudo_ratio = pseudo_ratio
        self.pure_real_baseline = pure_real_baseline
        self.scaler = GradScaler("cuda")
        self.current_epoch = 0

        self.fold_dir = os.path.join(exp_root, f"fold_{fold_num}")
        self.ckpt_dir = os.path.join(self.fold_dir, "checkpoints")
        self.vis_dir = os.path.join(self.fold_dir, "visualizations")
        for d in [self.ckpt_dir, self.vis_dir]:
            os.makedirs(d, exist_ok=True)

        self.log_path = os.path.join(self.fold_dir, "metrics_log.csv")
        self.sampling_manifest_path = os.path.join(
            self.fold_dir, "sampling_manifest.csv"
        )
        self.poor_cases_log = os.path.join(self.fold_dir, "poor_cases.txt")

        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_dice"])
        with open(self.sampling_manifest_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "sampling_seed", "pseudo_count", "case_ids"])

    def _run_epoch(self, dataloader, is_train, calculate_metrics=True):
        self.model.train() if is_train else self.model.eval()
        total_loss = 0.0
        all_dices = []
        case_avg_dices = {}

        if is_train:
            pbar = tqdm(
                dataloader,
                desc=f"E{self.current_epoch+1} Train",
                leave=False,
            )
            for batch_idx, (images, masks, weights, labels, is_pseudo_batch) in enumerate(pbar):
                images, masks, weights = (
                    images.to(self.device),
                    masks.to(self.device),
                    weights.to(self.device),
                )

                with torch.set_grad_enabled(True):
                    with autocast("cuda"):
                        logits = self.model(images)
                        pseudo_mask = is_pseudo_batch.to(self.device)
                        real_mask = ~pseudo_mask
                        loss = torch.tensor(0.0, device=self.device)

                        num_pseudo = pseudo_mask.sum().item()
                        num_real = real_mask.sum().item()
                        total_batch_size = images.size(0)

                        if isinstance(logits, list):
                            if num_pseudo > 0:
                                pseudo_logits_list = [out[pseudo_mask] for out in logits]
                                p_loss = self.criterion["seg"](
                                    pseudo_logits_list,
                                    masks[pseudo_mask],
                                    weight_map=weights[pseudo_mask],
                                    is_pseudo=True,
                                )
                                loss += p_loss * (num_pseudo / total_batch_size)

                            if num_real > 0:
                                real_logits_list = [out[real_mask] for out in logits]
                                r_loss = self.criterion["seg"](
                                    real_logits_list, masks[real_mask], is_pseudo=False
                                )
                                loss += r_loss * (num_real / total_batch_size)
                        else:
                            if num_pseudo > 0:
                                p_loss = self.criterion["seg"](
                                    logits[pseudo_mask],
                                    masks[pseudo_mask],
                                    weight_map=weights[pseudo_mask],
                                    is_pseudo=True,
                                )
                                loss += p_loss * (num_pseudo / total_batch_size)
                            if num_real > 0:
                                r_loss = self.criterion["seg"](
                                    logits[real_mask], masks[real_mask], is_pseudo=False
                                )
                                loss += r_loss * (num_real / total_batch_size)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                total_loss += loss.item()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            return (total_loss / len(dataloader)), all_dices, case_avg_dices

        else:
            dataset = dataloader.dataset
            val_samples = dataset.all_real_samples

            pbar = tqdm(
                val_samples,
                desc=f"E{self.current_epoch+1} Val (3D-CCA)",
                leave=False,
            )

            for s in pbar:
                case_name = s["case_name"]

                img_3d = np.load(s["data"])
                if img_3d.ndim == 4:
                    img_3d = img_3d[0]
                mask_3d = np.load(s["mask"])
                if mask_3d.ndim == 4:
                    mask_3d = mask_3d[0]

                D, H, W = img_3d.shape
                pred_3d = np.zeros_like(img_3d, dtype=np.float32)

                for z in range(D):
                    z_prev2 = max(0, z - 2)
                    z_prev1 = max(0, z - 1)
                    z_next1 = min(D - 1, z + 1)
                    z_next2 = min(D - 1, z + 2)
                    slice_5d = img_3d[[z_prev2, z_prev1, z, z_next1, z_next2]].copy()

                    with torch.no_grad():
                        with autocast("cuda"):
                            input_t = torch.from_numpy(slice_5d).unsqueeze(0).to(self.device).float()
                            logits = self.model(input_t)
                            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
                            pred_3d[z] = probs

                pred_bin_3d = np.zeros_like(pred_3d, dtype=np.uint8)
                for z in range(D):
                    binary_slice = (pred_3d[z] > self.val_threshold).astype(np.uint8)
                    if np.sum(binary_slice) > 0:
                        opened = binary_opening(binary_slice, structure=np.ones((3, 3)))
                        pred_bin_3d[z] = opened

                # Retain the dominant 3D component to suppress isolated false positives.
                labeled_3d, num_features = label(pred_bin_3d)
                if num_features > 0:
                    volumes = np.bincount(labeled_3d.ravel())
                    volumes[0] = 0
                    max_label = volumes.argmax()
                    pred_bin_3d[labeled_3d != max_label] = 0

                pred_bin_3d = binary_fill_holes(pred_bin_3d).astype(np.uint8)

                initial_areas = [np.sum(pred_bin_3d[z]) for z in range(D)]
                smooth_areas = gaussian_filter1d(initial_areas, sigma=2.0, mode="nearest")

                liver_start, liver_end = 0, D - 1
                for z in range(D - 2):
                    if initial_areas[z] > 100 and initial_areas[z+1] > 100 and initial_areas[z+2] > 100:
                        liver_start = z
                        break
                for z in range(D - 1, 1, -1):
                    if initial_areas[z] > 100 and initial_areas[z-1] > 100 and initial_areas[z-2] > 100:
                        liver_end = z
                        break

                for z in range(D):
                    if z < liver_start or z > liver_end or smooth_areas[z] < 100:
                        pred_bin_3d[z] = 0

                intersect = np.sum(pred_bin_3d * mask_3d)
                union = np.sum(pred_bin_3d) + np.sum(mask_3d)
                case_dice = (2.0 * intersect + 1e-5) / (union + 1e-5)

                all_dices.append(case_dice)
                case_avg_dices[case_name] = case_dice

            return 0.0, all_dices, case_avg_dices

    def train_one_fold(self, train_loader, val_loader, num_epochs=50, scheduler=None):
        best_val_dice = 0.0
        best_poor_cases = []

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            pseudo_ratio = 0 if self.pure_real_baseline else self.pseudo_ratio
            train_loader.dataset.set_epoch(epoch, pseudo_ratio=pseudo_ratio)
            selected_cases = getattr(
                train_loader.dataset, "selected_pseudo_case_names", []
            )
            with open(self.sampling_manifest_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        epoch + 1,
                        getattr(train_loader.dataset, "sampling_seed", ""),
                        len(selected_cases),
                        ";".join(selected_cases),
                    ]
                )
            train_loss, _, _ = self._run_epoch(train_loader, is_train=True)

            _, val_dices, case_avg_dices = self._run_epoch(
                val_loader, is_train=False, calculate_metrics=True
            )
            mean_dice = np.mean(val_dices)

            if scheduler:
                scheduler.step(mean_dice)

            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch + 1, train_loss, mean_dice])

            print(
                f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Local Val Dice (3D-CCA): {mean_dice:.4f}"
            )

            poor_list = [
                f"{c_id} (Dice: {d_val:.4f})"
                for c_id, d_val in case_avg_dices.items()
                if d_val < 0.70
            ]
            if poor_list:
                print(
                    f"   [WARNING] {len(poor_list)} cases underperformed this epoch: {', '.join(poor_list)}"
                )

            if mean_dice > best_val_dice:
                best_val_dice = mean_dice
                best_poor_cases = [
                    f"{c_id}:{d_val:.4f}"
                    for c_id, d_val in case_avg_dices.items()
                    if d_val < 0.70
                ]

                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.ckpt_dir, "best_model.pth"),
                )

                v_batch = next(iter(val_loader))
                self.model.eval()
                with torch.no_grad():
                    with autocast("cuda"):
                        v_logits = self.model(v_batch[0].to(self.device))
                self.save_visualization(v_batch[0], v_batch[1], v_logits, epoch)

        with open(self.poor_cases_log, "w") as f:
            for item in best_poor_cases:
                f.write(f"{item}\n")
        print(f"Poor-case list written to: {self.poor_cases_log}")

        return best_val_dice

    def save_visualization(self, images, masks, logits, epoch):
        prob = torch.sigmoid(logits[0, 0]).detach().cpu().numpy()
        img = images[0, 2].cpu().numpy()
        gt = masks[0, 0].cpu().numpy()

        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(img, cmap="gray")
        plt.title("Input GED4")
        plt.subplot(132)
        plt.imshow(gt, cmap="gray")
        plt.title("GT")
        plt.subplot(133)
        plt.imshow(prob, cmap="jet", vmin=0, vmax=1)
        plt.title(f"Pred Prob E{epoch+1}")
        plt.savefig(os.path.join(self.vis_dir, f"epoch_{epoch+1}.png"))
        plt.close()
