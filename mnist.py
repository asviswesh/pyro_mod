import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
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
        sample["digit"] = torch.as_tensor(
            np.asarray(sample["digit"]), dtype=torch.int64
        )
        return sample

def get_data(batch_size):
    transforms = Compose(
        [ToTensor()]
    )
    dataset = CVAEMNIST("../data", transform=transforms, download=True)
    # Forcing 80-20 test split, can configure this later.
    train_size = int(len(dataset) * 0.8)
    validation_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, validation_size])


    dataloaders = {
        "train": DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        ),
        "val": DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        ),
    }

    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}

    return train_dataset, val_dataset, dataloaders, dataset_sizes

