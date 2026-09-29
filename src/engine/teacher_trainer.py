import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.amp import autocast, GradScaler


class TeacherTrainer:
    def __init__(self, model, optimizer, criterion, device, fold_dir):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scaler = GradScaler("cuda")
        self.current_epoch = 0

        self.fold_dir = fold_dir
        self.ckpt_dir = os.path.join(self.fold_dir, "checkpoints")
        self.vis_dir = os.path.join(self.fold_dir, "visualizations")
        for d in [self.ckpt_dir, self.vis_dir]:
            os.makedirs(d, exist_ok=True)

        self.log_path = os.path.join(self.fold_dir, "teacher_metrics.csv")
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_dice"])

    def _run_epoch(self, dataloader, is_train):
        self.model.train() if is_train else self.model.eval()
        total_loss = 0.0
        all_dices = []

        desc = f"Epoch {self.current_epoch+1} {'[Train]' if is_train else '[Val]'}"
        if len(dataloader) == 0:
            return (0.0, []) if not is_train else (0.0, [])

        pbar = tqdm(dataloader, desc=desc, leave=False)

        for images, masks, _, labels, _ in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)

            with torch.set_grad_enabled(is_train):
                with autocast("cuda"):
                    seg_logits, staging_logits = self.model(images)

                    if is_train:
                        l_seg = self.criterion["seg"](
                            seg_logits, masks, is_pseudo=False
                        )
                        l_stage = self.criterion["staging"](staging_logits, labels)
                        loss = l_seg + 0.1 * l_stage
                    else:
                        probs = torch.sigmoid(seg_logits)
                        pred_bin = (probs > 0.5).float()
                        intersect = (pred_bin * masks).sum(dim=(1, 2, 3))
                        union = pred_bin.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
                        dice = (2.0 * intersect + 1e-5) / (union + 1e-5)
                        all_dices.extend(dice.cpu().numpy())

                if is_train:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    total_loss += loss.item()
                    pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        return avg_loss, all_dices

    def train(self, train_loader, val_loader, num_epochs, scheduler=None):
        best_val_dice = 0.0
        for epoch in range(num_epochs):
            self.current_epoch = epoch

            train_loss, _ = self._run_epoch(train_loader, is_train=True)
            _, val_dices = self._run_epoch(val_loader, is_train=False)

            if len(val_dices) == 0:
                mean_dice = 0.0
            else:
                mean_dice = float(np.mean(val_dices))
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(mean_dice)
                else:
                    scheduler.step()

            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch + 1, train_loss, mean_dice])

            print(
                f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f} | Val Dice: {mean_dice:.4f}"
            )

            if mean_dice > best_val_dice:
                best_val_dice = mean_dice
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.ckpt_dir, "teacher_best.pth"),
                )
                print(f"  [+] New best score. Dice: {best_val_dice:.4f}")

        return best_val_dice
