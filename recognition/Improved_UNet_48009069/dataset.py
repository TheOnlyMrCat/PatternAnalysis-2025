import os
import torch
import torch.nn as nn
import torchvision
import torchvision.io
import torchvision.transforms.v2 as transforms

import numpy as np
import nibabel as nib
from tqdm import tqdm


class ToOneHotSegmentMap(nn.Module):
    def forward(self, img):
        labels = torch.unique(img)

        channels = []
        for label in labels:
            channel = torch.zeros_like(img, dtype=torch.uint8)
            channel[img == label] = 1
            channels.append(channel)

        new_img = torch.concat(channels)
        return new_img


class OASISDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir: str, segment_dir: str, transform=None, seg_transform=None):
        self.image_dir = image_dir
        self.segment_dir = segment_dir
        self.transform = transform
        self.seg_transform = seg_transform

        self.cases = list(
            {filename.removeprefix("case_") for filename in os.listdir(image_dir)}
        )

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx: int):
        image = torchvision.io.decode_image(
            os.path.join(self.image_dir, f"case_{self.cases[idx]}")
        )
        if self.transform:
            image = self.transform(image)

        segmentation = torchvision.io.decode_image(
            os.path.join(self.segment_dir, f"seg_{self.cases[idx]}")
        )
        if self.seg_transform:
            segmentation = self.seg_transform(segmentation)

        return image, segmentation


def to_channels(arr: np.ndarray, count: int, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (count,), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c : c + 1][arr == c] = 1

    return res


# load medical image functions
def load_data_2D(
    imageNames,
    normImage=False,
    categorical=False,
    channel_count=0,
    dtype=np.float32,
    getAffines=False,
    early_stop=False,
):
    """
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    normImage: bool (normalise the image 0.0-1.0)
    early_stop: Stop loading pre-maturely, leaves arrays mostly empty, for quick loading and testing scripts.
    """
    affines = []

    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching="unchanged")
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # sometimes extra dims , remove
    if categorical:
        first_case = to_channels(first_case, channel_count, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching="unchanged")  # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # sometimes extra dims in HipMRI_study data

        if inImage.shape[1] > images.shape[2]:
            inImage = inImage[:, :images.shape[2]]  # FIXME

        inImage = inImage.astype(dtype)
        if normImage:
            # ~ inImage = inImage / np . linalg . norm ( inImage )
            # ~ inImage = 255. * inImage / inImage . max ()
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = to_channels(inImage, channel_count, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images


class HipMRIDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir: str, segment_dir: str, transform=None, seg_transform=None):
        self.image_dir = image_dir
        self.segment_dir = segment_dir
        self.transform = transform
        self.seg_transform = seg_transform

        self.cases = list(
            {filename.removeprefix("case_") for filename in os.listdir(image_dir)}
        )
        self.images = load_data_2D([os.path.join(self.image_dir, f"case_{case}") for case in self.cases])
        self.segmentations = load_data_2D(
            [os.path.join(self.segment_dir, f"seg_{case}") for case in self.cases],
            categorical=True,
            channel_count=5,
            dtype=np.uint8,
        )

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)

        segmentation = self.segmentations[idx]
        if self.seg_transform:
            segmentation = self.seg_transform(segmentation)

        return torch.from_numpy(np.reshape(image, (1,) + image.shape)), torch.from_numpy(segmentation.transpose((2, 0, 1)))

def load_datasets(root: str) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Load the training and testing OASIS datasets from the given root directory.

    The expected directory structure looks like:

    root
    |- /keras_slices_train
    |- /keras_slices_seg_train
    |- /keras_slices_validate
    |- /keras_slices_seg_validate
    |- /keras_slices_test
    |- /keras_slices_seg_test
    """

    trainset = HipMRIDataset(
        image_dir=os.path.join(root, "keras_slices_train"),
        segment_dir=os.path.join(root, "keras_slices_seg_train"),
    )

    valset = HipMRIDataset(
        image_dir=os.path.join(root, "keras_slices_validate"),
        segment_dir=os.path.join(root, "keras_slices_seg_validate"),
    )

    testset = HipMRIDataset(
        image_dir=os.path.join(root, "keras_slices_test"),
        segment_dir=os.path.join(root, "keras_slices_seg_test"),
    )

    return trainset, valset, testset
