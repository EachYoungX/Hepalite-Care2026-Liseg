import os
import glob
import numpy as np
import torch
import random
import logging
from torch.utils.data import Dataset


class HybridDataset25D(Dataset):
    def __init__(
        self,
        train_dir,
        vendors,
        val_dir=None,
        transform=None,
        case_ids=None,
        is_train=True,
        val_pseudo_weight_scale=0.35,
        sampling_seed=42,
    ):
        self.train_dir = train_dir
        self.vendors = vendors
        self.val_dir = val_dir
        self.transform = transform
        self.is_train = is_train
        self.val_pseudo_weight_scale = val_pseudo_weight_scale
        self.case_ids = set(case_ids or [])
        self.sampling_seed = int(sampling_seed)
        self.all_real_samples = []
        self.all_pseudo_samples = []
        self.active_samples = []
        self.selected_pseudo_case_names = []
        self.slices_per_subject = 4 if is_train else 2
        self.val_indices = []

        # GED4 is the final channel in the seven-channel input convention.
        self.ged4_idx = 6

        self._load_samples(case_ids)
        logging.info(
            f"[{('train' if is_train else 'validation')} set | directory: {os.path.basename(train_dir)}] loaded. "
            f"Manual cases: {len(self.all_real_samples)}, pseudo-labeled cases: {len(self.all_pseudo_samples)}"
        )
        self.set_epoch(0)

    def _load_samples(self, case_ids):
        for vendor in self.vendors:
            v_train_path = os.path.join(self.train_dir, vendor)
            if not os.path.exists(v_train_path):
                continue

            data_files = sorted(glob.glob(os.path.join(v_train_path, "*_data.npy")))
            for d_path in data_files:
                case_name = os.path.basename(d_path).replace("_data.npy", "")

                m_path = d_path.replace("_data.npy", "_mask.npy")
                pm_path = d_path.replace("_data.npy", "_pseudo_mask.npy")
                w_path = d_path.replace("_data.npy", "_weight.npy")

                is_real = os.path.exists(m_path)
                has_pseudo = os.path.exists(pm_path)
                has_weight = os.path.exists(w_path)

                if self.is_train:
                    # A manually labeled case must never fall through into the
                    # pseudo pool. This protects held-out folds from stale
                    # pseudo files that happen to coexist with a real mask.
                    if is_real:
                        if has_pseudo or has_weight:
                            logging.warning(
                                "Ignoring stale pseudo-label artifacts for real case %s "
                                "(pseudo=%s, weight=%s)",
                                case_name,
                                has_pseudo,
                                has_weight,
                            )
                        if case_name not in self.case_ids:
                            continue
                        target_mask = m_path
                        is_pseudo = False
                    elif has_pseudo:
                        target_mask = pm_path
                        is_pseudo = True
                    else:
                        continue
                else:
                    if is_real and case_name in self.case_ids:
                        target_mask = m_path
                        is_pseudo = False
                    else:
                        continue

                self._add_to_pool(
                    case_name,
                    d_path,
                    target_mask,
                    w_path,
                    is_pseudo,
                    pseudo_weight_scale=1.0,
                )

        if self.is_train and self.val_dir is not None:
            for vendor in self.vendors:
                v_val_path = os.path.join(self.val_dir, vendor)
                if not os.path.exists(v_val_path):
                    continue

                data_files = sorted(glob.glob(os.path.join(v_val_path, "*_data.npy")))
                for d_path in data_files:
                    case_name = os.path.basename(d_path).replace("_data.npy", "")
                    pm_path = d_path.replace("_data.npy", "_pseudo_mask.npy")
                    w_path = d_path.replace("_data.npy", "_weight.npy")

                    if os.path.exists(pm_path):
                        self._add_to_pool(
                            case_name,
                            d_path,
                            pm_path,
                            w_path,
                            is_pseudo=True,
                            pseudo_weight_scale=self.val_pseudo_weight_scale,
                        )

    def _add_to_pool(
        self,
        case_name,
        d_path,
        target_mask_path,
        w_path,
        is_pseudo,
        pseudo_weight_scale=1.0,
    ):
        try:
            mask_mmap = np.load(target_mask_path, mmap_mode="r")
            if mask_mmap.ndim == 4:
                mask_mmap = mask_mmap[0]

            slice_areas = np.sum(mask_mmap, axis=(1, 2))
            fg_slices = np.where(slice_areas > 0)[0]
            bg_slices = np.where(slice_areas == 0)[0]
            if len(fg_slices) < 1:
                if not is_pseudo:
                    fg_slices = [mask_mmap.shape[0] // 2]
                else:
                    return

            edge_slices = []
            if len(fg_slices) > 0:
                fg_areas = slice_areas[fg_slices]
                small_area_limit = max(300, float(fg_areas.max()) * 0.25)
                small_fg = fg_slices[fg_areas <= small_area_limit]
                span_edges = np.unique(np.concatenate([fg_slices[:3], fg_slices[-3:]]))
                edge_slices = np.unique(np.concatenate([small_fg, span_edges])).astype(
                    int
                )

            black_slices = []
            try:
                img_mmap = np.load(d_path, mmap_mode="r")
                if img_mmap.ndim == 4:
                    img_ged4 = (
                        img_mmap[self.ged4_idx]
                        if img_mmap.shape[0] > self.ged4_idx
                        else img_mmap[0]
                    )
                else:
                    img_ged4 = img_mmap
                nonzero_counts = np.sum(np.abs(img_ged4) > 1e-6, axis=(1, 2))
                black_slices = np.where((nonzero_counts <= 16) & (slice_areas == 0))[0]
            except Exception:
                black_slices = []

            sample_info = {
                "case_name": case_name,
                "data": d_path,
                "mask": target_mask_path,
                "weight": w_path if (is_pseudo and os.path.exists(w_path)) else None,
                "fg_slices": fg_slices,
                "bg_slices": bg_slices,
                "edge_slices": edge_slices,
                "black_slices": black_slices,
                "stage_label": 0,
                "is_pseudo": is_pseudo,
                "pseudo_weight_scale": float(pseudo_weight_scale),
                "total_z": mask_mmap.shape[0],
            }

            if is_pseudo:
                fg_ratio = len(fg_slices) / mask_mmap.shape[0]
                if 0.05 < fg_ratio < 0.8:
                    self.all_pseudo_samples.append(sample_info)
            else:
                self.all_real_samples.append(sample_info)
        except Exception as exc:
            logging.warning(
                "Skipping invalid dataset case %s (%s): %s",
                case_name,
                target_mask_path,
                exc,
            )

    def __len__(self):
        if not self.is_train:
            return len(self.val_indices)
        return len(self.active_samples) * self.slices_per_subject

    def set_epoch(self, epoch, pseudo_ratio=20):
        sampling_rng = random.Random(self.sampling_seed + int(epoch))
        self.selected_pseudo_case_names = []

        if not self.is_train:
            self.active_samples = self.all_real_samples
            self.val_indices = [
                (sample_idx, z)
                for sample_idx, sample in enumerate(self.active_samples)
                for z in range(sample["total_z"])
            ]
            return

        # Increase pseudo-label usage gradually while preserving a real-label majority.
        if pseudo_ratio == 0:
            current_p_ratio = 0
        else:
            if epoch < 5:
                current_p_ratio = 0
            elif epoch < 10:
                current_p_ratio = 5
            elif epoch < 15:
                current_p_ratio = 10
            elif epoch < 20:
                current_p_ratio = 15
            else:
                current_p_ratio = min(pseudo_ratio, 20)

        selected_real = self.all_real_samples * 2
        num_pseudo = min(current_p_ratio, len(self.all_pseudo_samples))

        if num_pseudo > 0:
            selected_pseudo = sampling_rng.sample(self.all_pseudo_samples, num_pseudo)
            self.selected_pseudo_case_names = [
                sample["case_name"] for sample in selected_pseudo
            ]
            self.active_samples = selected_real + selected_pseudo
        else:
            self.active_samples = selected_real

        sampling_rng.shuffle(self.active_samples)

    def __getitem__(self, idx):
        if self.is_train:
            sub_idx = idx // self.slices_per_subject
            s = self.active_samples[sub_idx]
        else:
            sub_idx, fixed_z = self.val_indices[idx]
            s = self.active_samples[sub_idx]

        img_3d_raw = np.load(s["data"], mmap_mode="r")
        mask_3d_mmap = np.load(s["mask"], mmap_mode="r")

        if img_3d_raw.ndim == 4:
            img_ged4 = (
                img_3d_raw[self.ged4_idx]
                if img_3d_raw.shape[0] > self.ged4_idx
                else img_3d_raw[0]
            )
        else:
            img_ged4 = img_3d_raw

        if mask_3d_mmap.ndim == 4:
            mask_3d_mmap = mask_3d_mmap[0]

        if self.is_train:
            # Mix foreground, boundary, background, and near-empty slices to reduce
            # false positives outside the target organ.
            r = random.random()
            if r < 0.40:
                z = int(random.choice(s["fg_slices"]))
            elif r < 0.65 and len(s["edge_slices"]) > 0:
                z = int(random.choice(s["edge_slices"]))
            elif r < 0.90 and len(s["bg_slices"]) > 0:
                z = int(random.choice(s["bg_slices"]))
            elif len(s["black_slices"]) > 0:
                z = int(random.choice(s["black_slices"]))
            elif len(s["bg_slices"]) > 0:
                z = int(random.choice(s["bg_slices"]))
            else:
                z = random.randint(0, s["total_z"] - 1)
        else:
            z = fixed_z

        z_prev2 = max(0, z - 2)
        z_prev1 = max(0, z - 1)
        z_next1 = min(s["total_z"] - 1, z + 1)
        z_next2 = min(s["total_z"] - 1, z + 2)

        img_5d = img_ged4[[z_prev2, z_prev1, z, z_next1, z_next2], :, :].copy()
        mask_2d = mask_3d_mmap[z, :, :].copy()

        if s["is_pseudo"]:
            if s["weight"] is not None:
                weight_3d = np.load(s["weight"], mmap_mode="r")
                if weight_3d.ndim == 4:
                    weight_3d = weight_3d[0]
                weight_2d = weight_3d[z, :, :].copy()
                weight_2d = np.clip(weight_2d, 0.15, 1.0)
            else:
                weight_2d = np.ones_like(mask_2d, dtype=np.float32) * 0.5
            weight_2d = np.clip(
                weight_2d * s.get("pseudo_weight_scale", 1.0), 0.05, 1.0
            )
        else:
            weight_2d = np.ones_like(mask_2d, dtype=np.float32)

        img_5d = np.nan_to_num(img_5d)

        if self.transform is not None:
            mask_3d_temp = mask_2d[np.newaxis, ...]
            weight_3d_temp = weight_2d[np.newaxis, ...]
            img_5d, mask_3d_temp, weight_3d_temp = self.transform(
                img_5d, mask_3d_temp, weight_3d_temp
            )

            return (
                torch.from_numpy(img_5d).float(),
                torch.from_numpy(mask_3d_temp).float(),
                torch.from_numpy(weight_3d_temp).float(),
                torch.tensor(s["stage_label"]).long(),
                torch.tensor(s["is_pseudo"]).bool(),
            )
        else:
            return (
                torch.from_numpy(img_5d).float(),
                torch.from_numpy(mask_2d).unsqueeze(0).float(),
                torch.from_numpy(weight_2d).unsqueeze(0).float(),
                torch.tensor(s["stage_label"]).long(),
                torch.tensor(s["is_pseudo"]).bool(),
            )
