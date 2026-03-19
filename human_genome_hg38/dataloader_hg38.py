import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split


class HG38Dataset(Dataset):
    def __init__(self, token_path):
        print(f"Loading {token_path}...")
        self.data = np.load(token_path)  # shape: (N, SEQ_LEN)
        print(f"Loaded {len(self.data):,} sequences")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)


def get_hg38_dataloaders(token_path, batch_size=64, val_ratio=0.01):
    dataset = HG38Dataset(token_path)

    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    return train_loader, val_loader