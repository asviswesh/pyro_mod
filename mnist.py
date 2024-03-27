import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, functional

from pyro.contrib.examples.util import MNIST

class CVAEMNIST(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.original = MNIST(root, train=train, download=download)
        self.transform = transform

    def __len__(self):
        return len(self.original)

    def __getitem__(self, item):
        image, digit = self.original[item]
        sample = {"original": image, "digit": digit}
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor:
    def __call__(self, sample):
        sample["original"] = functional.to_tensor(sample["original"])
        digit_list = [0] * 10
        digit_list[sample["digit"]] = 1
        sample["digit"] = torch.as_tensor(
            np.asarray(digit_list), dtype=torch.float64
        )
        return sample

def get_data(batch_size):
    transforms = Compose(
        [ToTensor()]
    )
    datasets, dataloaders, dataset_sizes = {}, {}, {}
    for mode in ["train", "val"]:
        datasets[mode] = CVAEMNIST(
            "../data", download=True, transform=transforms, train=mode == "train"
        )
        dataloaders[mode] = DataLoader(
            datasets[mode],
            batch_size=batch_size,
            shuffle=mode == "train",
            num_workers=0,
        )
        dataset_sizes[mode] = len(datasets[mode])
    return datasets, dataloaders, dataset_sizes

