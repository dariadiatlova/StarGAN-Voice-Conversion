import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import pad_2D


def get_dataloader(data_dir_path: str, batch_size: int, drop_last: bool = False, shuffle: bool = False,
                   sort: bool = True, num_workers: int = 128):
    dataset = Dataset(data_dir=data_dir_path, batch_size=batch_size, sort=sort, drop_last=drop_last)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn,
                        shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, pin_memory=True)
    return loader


class Dataset(Dataset):
    def __init__(self, data_dir: str, batch_size: int, sort: bool = True, drop_last: bool = False):
        self.data_dir = f"{data_dir}/mels"
        self.ruslan_files = [i for i in os.listdir(self.data_dir) if i.split("_")[0] == "0"]
        self.sort = sort
        self.drop_last = drop_last
        self.batch_size = batch_size

    def __len__(self):
        return len(self.ruslan_files)

    def __getitem__(self, idx):
        ruslan_fn = self.ruslan_files[idx]
        sample_id = ruslan_fn.split('_')[-1]
        hleb_fn = f"1_{sample_id}"
        ruslan_mel = torch.load(f"{self.data_dir}/{ruslan_fn}").squeeze(0).numpy()
        hleb_mel = torch.load(f"{self.data_dir}/{hleb_fn}").squeeze(0).numpy()
        sample = {"sample_id": int(sample_id[:-3]), "ruslan_mel": ruslan_mel, "hleb_mel": hleb_mel,
                  "audio_size": int(ruslan_mel.shape[1])}
        return sample

    def reprocess(self, data, idxs):
        """
        Batch consists of a List[["sample_id"], [2d ruslan mel spec], [2d hleb mel spec], [int, second mel dim size]]
        """
        ids = torch.Tensor([data[idx]["sample_id"] for idx in idxs]).long()
        ruslan_mels = torch.from_numpy(pad_2D([data[idx]["ruslan_mel"] for idx in idxs])).float()
        hleb_mels = torch.from_numpy(pad_2D([data[idx]["hleb_mel"] for idx in idxs])).float()
        audio_sizes = torch.Tensor([data[idx]["audio_size"] for idx in idxs]).long()
        return ids, ruslan_mels, hleb_mels, audio_sizes

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["audio_size"] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size):]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]
        output = [self.reprocess(data, idx) for idx in idx_arr]

        return output
