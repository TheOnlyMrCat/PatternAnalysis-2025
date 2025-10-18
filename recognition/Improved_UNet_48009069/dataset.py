import os
import torch
import torchvision
import torchvision.io
import torchvision.transforms.v2 as transforms


class OASISDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir: str, segment_dir: str, transform=None):
        self.image_dir = image_dir
        self.segment_dir = segment_dir
        self.transform = transform

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
        if self.transform:
            segmentation = self.transform(segmentation)

        return image, segmentation


def load_datasets(root: str) -> tuple[OASISDataset, OASISDataset]:
    """
    Load the training and testing OASIS datasets from the given root directory.

    The expected directory structure looks like:

    root
    |- /keras_png_slices_train
    |- /keras_png_slices_seg_train
    |- /keras_png_slices_test
    |- /keras_png_slices_seg_test
    """
    transform = transforms.Compose(
        [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
    )

    trainset = OASISDataset(
        image_dir=os.path.join(root, "keras_png_slices_train"),
        segment_dir=os.path.join(root, "keras_png_slices_seg_train"),
        transform=transform,
    )

    testset = OASISDataset(
        image_dir=os.path.join(root, "keras_png_slices_test"),
        segment_dir=os.path.join(root, "keras_png_slices_seg_test"),
        transform=transform,
    )

    return trainset, testset
