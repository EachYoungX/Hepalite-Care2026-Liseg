import numpy as np
train_sample = np.load("dataset/processed/training_set/Vendor_B1/0241-B1-S2_data.npy")[0]
val_sample = np.load("dataset/processed/validation_set/Vendor_A/0930-A_data.npy")[0]
print(f"Train GED4 Mean: {train_sample.mean():.4f} | Max: {train_sample.max():.4f}")
print(f"Val Sample Mean: {val_sample.mean():.4f} | Max: {val_sample.max():.4f}")