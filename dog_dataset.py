import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from dog_utils import pad_2D


def get_dataloader(data_dir_path: str, batch_size: int, drop_last: bool = False, shuffle: bool = False,
                   sort: bool = True, num_workers: int = 16):
    dataset = DogDataset(data_dir=data_dir_path, batch_size=batch_size, sort=sort, drop_last=drop_last)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn,
                        shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, pin_memory=True)
    return loader


class DogDataset(Dataset):
    def __init__(self, data_dir: str, batch_size: int, sort: bool = True, drop_last: bool = False):
        self.data_dir = f"{data_dir}/mels"
        self.human_files = [i for i in os.listdir(self.data_dir) if i.split("_")[0] == "0"]
        self.sort = sort
        self.drop_last = drop_last
        self.batch_size = batch_size

    def __len__(self):
        return len(self.human_files)

    def __getitem__(self, idx):
        human_fn = self.human_files[idx]
        sample_id = human_fn.split('_')[-1]
        dog_fn = f"1_{sample_id}"
        human_mel = np.load(f"{self.data_dir}/{human_fn}")
        dog_mel = np.load(f"{self.data_dir}/{dog_fn}")
        # assert len(human_mel.shape) == 2 and len(dog_mel.shape) == 2, f"Check shape of mel {sample}!"
        sample = {"sample_id": int(sample_id[:-4]), "human_mel": human_mel, "dog_mel": dog_mel,
                  "audio_size": int(human_mel.shape[0])}
        return sample

    def reprocess(self, data, idxs):
        """
        Batch consists of a List[["sample_id"], [2d human mel spec], [2d dog mel spec], [int, second mel dim size]]
        """
        ids = torch.Tensor([data[idx]["sample_id"] for idx in idxs]).long()
        human_mels = torch.from_numpy(pad_2D([data[idx]["human_mel"] for idx in idxs])).float()
        dog_mels = torch.from_numpy(pad_2D([data[idx]["dog_mel"] for idx in idxs])).float()
        audio_sizes = torch.Tensor([data[idx]["audio_size"] for idx in idxs]).long()
        return ids, human_mels, dog_mels, audio_sizes

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
