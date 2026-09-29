from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ImageRecord:
    image_path: str
    label: int
    subgroup: str
    sample_id: str


class TabularImageDataset(Dataset):
    def __init__(self, records: list[ImageRecord], transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, record.label, record.subgroup, record.sample_id
