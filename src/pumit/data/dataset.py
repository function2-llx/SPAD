"""PUMIT dataset class for DA-aware batch sampling."""

from __future__ import annotations

from collections.abc import Callable

from torch.utils.data import Dataset

from pumit.transforms import TransInfo


class PUMITDataset(Dataset):
    """Dataset that accepts (index, trans_info) tuples from DABatchSampler."""

    def __init__(self, data: list[dict], transform: Callable):
        self.data = data
        self.transform = transform

    def __getitem__(self, item: tuple[int, TransInfo]):
        index, trans_info = item
        data = dict(self.data[index])
        data['_trans'] = trans_info
        return self.transform(data), trans_info

    def __len__(self):
        return len(self.data)
