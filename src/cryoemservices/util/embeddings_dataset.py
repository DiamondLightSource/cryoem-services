import random
from pathlib import Path

import mrcfile
import numpy as np
import torch
from torch.utils.data import Dataset


def _handle_frame(
    data: np.array, crop: tuple[int, int] | None = None, normalise: bool = True
) -> np.array:
    if not crop:
        crop = (np.min(data.shape), np.min(data.shape))
    left = (data.shape[0] - crop[0]) // 2
    top = (data.shape[1] - crop[1]) // 2
    right = left + crop[0]
    bottom = top + crop[1]
    data = data[left:right, top:bottom]
    if normalise:
        mean = np.mean(data)
        sdev = np.std(data)
        sigma_min = mean - 3 * sdev
        sigma_max = mean + 3 * sdev
        data = np.ndarray.copy(data)
        data[data < sigma_min] = sigma_min
        data[data > sigma_max] = sigma_max
        data = data - data.min()
        data = data / data.max()
        data = data * 255
        data = data.astype("uint8")
    return data


def read_img(
    img_path: Path,
    normalise: bool = True,
    crop: tuple[int] | None = None,
    multiframe: bool = False,
) -> np.array:
    data = mrcfile.read(img_path)

    if multiframe:
        adjusted_data = np.zeros(data.shape)
        for i, f in enumerate(data):
            adjusted_data[i] = _handle_frame(f, normalise=normalise)
    else:
        adjusted_data = _handle_frame(data, normalise=normalise)

    return adjusted_data


class ClassDataset(Dataset):
    def __init__(
        self,
        class_image_paths: list[Path],
        skip_indices: list[list[int]] | None = None,
        transform=None,
    ):
        self.imgs: np.array | None = None
        self.num_classes_per_batch = None
        true_index = 0
        for i, cl in enumerate(class_image_paths):
            data = read_img(cl, multiframe=True)
            if not self.num_classes_per_batch:
                self.num_classes_per_batch = data.shape[0]
            if self.imgs is None:
                if skip_indices:
                    total = np.sum([data.shape[0] - len(si) for si in skip_indices])
                    self.imgs = np.zeros((total, data[0].shape[0], data[0].shape[1]))
                    for j, d in enumerate(data):
                        if j not in skip_indices[i]:
                            self.imgs[true_index] = d
                            true_index += 1
                else:
                    self.imgs = np.zeros(
                        (
                            len(class_image_paths) * data.shape[0],
                            data[0].shape[0],
                            data[0].shape[1],
                        )
                    )
                    self.imgs[: data.shape[0]] = data
            else:
                if skip_indices:
                    for j, d in enumerate(data):
                        if j not in skip_indices[i]:
                            self.imgs[true_index] = d
                            true_index += 1
                else:
                    self.imgs[i * data.shape[0] : (i + 1) * data.shape[0]] = data
        if self.imgs is None:
            raise ValueError("ClassDataset contains no images")
        self.labels = list(range(self.imgs.shape[0]))
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()  # type: ignore

        idx2 = random.randint(0, self.__len__() - 1)

        if self.imgs is None:
            raise ValueError("ClassDataset contains no images")

        sample = self.imgs[idx].astype(np.float32)
        sample2 = self.imgs[idx2].astype(np.float32)

        sample = 2.0 * (sample / 255.0) - 1.0
        sample2 = 2.0 * (sample2 / 255.0) - 1.0

        sample = torch.from_numpy(np.expand_dims(sample, axis=0))
        sample2 = torch.from_numpy(np.expand_dims(sample2, axis=0))
        if self.transform:
            sample = self.transform(sample)
            sample2 = self.transform(sample2)

        labs = self.labels[idx]

        labs = torch.from_numpy(np.expand_dims(labs, axis=0))

        sample = {"x1": sample, "x2": sample2, "lab": labs}

        return sample
