import os

import pandas as pd
import torch
from torch.utils.data import Dataset


class PTClassificationDataset(Dataset):
    """
    Читає bag-level .pt фічі (features_uni2h_*), збережені як dict:
        {'features': [N_patches, D], 'coords': [N_patches, 2], 'slide_id': str, ...}

    Очікує csv зі сплітів build_splits.py: колонки slide_id, patient_id, label, label_str.
    train/test/val використовують РІЗНІ директорії з фічами (train_full vs test_full),
    тому feature_dir передається окремо і НЕ виводиться зі split-назви автоматично.
    """

    def __init__(self, split_csv: str, feature_dir: str):
        self.df = pd.read_csv(split_csv)
        self.feature_dir = feature_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = row["slide_id"]
        path = os.path.join(self.feature_dir, f"{slide_id}.pt")

        # weights_only=False: свої ж фічі, довіряємо джерелу.
        # torch>=2.6 інакше впаде на numpy scalar в source_attrs.
        data = torch.load(path, map_location="cpu", weights_only=False)

        feats = data["features"]  # [N_patches, 1536], float32
        coords = data["coords"].float()  # [N_patches, 2] — позиційні embeddings
        label = int(row["label"])
        return feats, label, coords
