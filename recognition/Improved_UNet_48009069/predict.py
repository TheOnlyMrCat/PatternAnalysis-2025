import argparse
import csv
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
from tqdm import tqdm

import dataset
import modules


@dataclass
class TestResults():
    """
    The results of testing a model against the HipMRI Dataset, broken down by test case and by label.

    Args:
        testset: The dataset this test was run against.
        loss: Overall Dice Loss for each test case
        similarity: Dice Similarity Coefficient for each label of each test case
        accuracy: Overall segmentation accuracy for each test case
    """
    testset: dataset.HipMRIDataset
    loss: list[float]
    similarity: list[list[float]]
    accuracy: list[float]

    def avg_loss(self) -> float:
        return sum(self.loss) / len(self.loss)

    def avg_accuracy(self) -> float:
        return sum(self.accuracy) / len(self.accuracy)

    def print_summary(self):
        loss = self.avg_loss()
        sim = np.asarray(self.similarity)
        accuracy = self.avg_accuracy()
        print(f"Average loss: {loss:.5f}")
        print("Dice Similarity Coefficients (min/avg/max):")
        print(f"- Background: {np.min(sim[:, 0]):.3f}/{np.sum(sim[:, 0]) / sim.shape[0]:.3f}/{np.max(sim[:, 0]):.3f}")
        print(f"- Body: {np.min(sim[:, 1]):.3f}/{np.sum(sim[:, 1]) / sim.shape[0]:.3f}/{np.max(sim[:, 1]):.3f}")
        print(f"- Bone: {np.min(sim[:, 2]):.3f}/{np.sum(sim[:, 2]) / sim.shape[0]:.3f}/{np.max(sim[:, 2]):.3f}")
        print(f"- Bladder: {np.min(sim[:, 3]):.3f}/{np.sum(sim[:, 3]) / sim.shape[0]:.3f}/{np.max(sim[:, 3]):.3f}")
        print(f"- Prostate: {np.min(sim[:, 4]):.3f}/{np.sum(sim[:, 4]) / sim.shape[0]:.3f}/{np.max(sim[:, 4]):.3f}")
        print(f"Accuracy: {accuracy * 100:.1f}%")


def test_model(
    model: torch.nn.Module,
    testset: dataset.HipMRIDataset,
    *,
    device: torch.device,
    progress: bool = True,
    batch_size: int = 1,
    shuffle: bool = False,
) -> TestResults:
    """
    Test an image segmentation model on the given test dataset, recording loss,
    Dice similarity, and accuracy for each batch.

    Args:
        model: The image segmentation model being tested.
        testset: The dataset to test the model against.
        device: The PyTorch device to run the segmentation on.
                The passed `model` should already be on this device.
                Data from `testset` will be loaded to this device before running inference.
        progress: Print a progress bar to stderr.
        batch_size, shuffle: Passed directly to the constructor for `torch.utils.data.DataLoader`.
    """
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
    criterion = modules.MulticlassDiceLoss()

    model.eval()
    losses = []
    similarities = []
    accuracies = []
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(test_loader, disable=not progress)):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            losses.append(criterion(outputs, masks).cpu().item())
            similarities.append([
                1 - modules.DiceLoss()(outputs[:, ch], masks[:, ch]).cpu().item()
                for ch in range(5)
            ])

            seg = torch.argmax(masks, 1)
            predicted_seg = torch.argmax(outputs, 1)
            accuracies.append((torch.sum(seg == predicted_seg) / seg.numel()).cpu().item())

    return TestResults(testset, losses, similarities, accuracies)


def segment_image(model, image, seg):
    model.eval()
    output = torch.squeeze(model(torch.unsqueeze(image, 0)), 0)

    fig, axes = plt.subplots(2, 7, figsize=(18, 6))

    # Plot image
    axes[0][0].imshow(image.cpu().numpy().transpose(1, 2, 0), cmap="gray")
    axes[0][0].set_title("Image")
    axes[0][0].axis("off")
    axes[1][0].imshow(image.cpu().numpy().transpose(1, 2, 0), cmap="gray")
    axes[1][0].set_title("Image")
    axes[1][0].axis("off")

    # Plot each one-hot channel
    channel_names = ["Background", "Body", "Bone", "Bladder", "Prostate"]
    for ch in range(5):
        # Ground truth segmentation
        axes[0][ch + 1].imshow(seg[ch].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[0][ch + 1].set_title(f"Ground Truth {channel_names[ch]}")
        axes[0][ch + 1].axis("off")

        # Predicted segmentation
        axes[1][ch + 1].imshow(output[ch].cpu().detach().numpy(), cmap="gray")
        axes[1][ch + 1].set_title(f"Predicted (DSC: {1 - modules.DiceLoss()(output[ch], seg[ch]):.3f})")
        axes[1][ch + 1].axis("off")

    # Create collated segmentation maps
    seg_full = torch.argmax(seg, 0)
    predict_full = torch.argmax(output, 0)
    accuracy = torch.sum(seg_full == predict_full) / seg_full.numel()

    axes[0][6].imshow(seg_full.cpu().numpy(), cmap="tab10")
    axes[0][6].set_title("Ground Truth Segmentation")
    axes[0][6].axis("off")
    axes[1][6].imshow(predict_full.cpu().numpy(), cmap="tab10")
    axes[1][6].set_title(f"Predicted (accuracy: {accuracy * 100:.1f}%)")
    axes[1][6].axis("off")

    plt.tight_layout()
    plt.show()


def predict_random(model: torch.nn.Module):
    dataset_root = os.getenv("HIPMRI_ROOT")
    if dataset_root is None:
        print("error: need $HIPMRI_ROOT to be set to root path of dataset")
        exit(1)
    testset = dataset.load_test(dataset_root)

    image, seg = random.choice(testset)
    segment_image(model, image, seg)


def predict_idx(model: torch.nn.Module, idx: int):
    dataset_root = os.getenv("HIPMRI_ROOT")
    if dataset_root is None:
        print("error: need $HIPMRI_ROOT to be set to root path of dataset")
        exit(1)
    testset = dataset.load_test(dataset_root)

    image, seg = testset[idx]
    segment_image(model, image, seg)


def test_summary(model: torch.nn.Module, *, device: torch.device):
    dataset_root = os.getenv("HIPMRI_ROOT")
    if dataset_root is None:
        print("error: need $HIPMRI_ROOT to be set to root path of dataset")
        exit(1)
    testset = dataset.load_test(dataset_root)

    test_model(model, testset, device=device).print_summary()


def test_details(model: torch.nn.Module, *, device: torch.device):
    dataset_root = os.getenv("HIPMRI_ROOT")
    if dataset_root is None:
        print("error: need $HIPMRI_ROOT to be set to root path of dataset")
        exit(1)
    testset = dataset.load_test(dataset_root)

    test_results = test_model(model, testset, device=device)

    with open("test.csv", "w", newline="") as out_file:
        writer = csv.writer(out_file)

        # Header row
        writer.writerow([
            "index",
            "image_file",
            "segmentation_file",
            "loss",
            "acc",
            "background_sim",
            "body_sim",
            "bone_sim",
            "bladder_sim",
            "prostate_sim",
        ])

        # Data rows
        writer.writerows([
            [
                i,
                testset.image_file(i),
                testset.seg_file(i),
                loss,
                acc,
                sim[0],
                sim[1],
                sim[2],
                sim[3],
                sim[4],
            ]
            for i, (loss, sim, acc) in enumerate(zip(
                test_results.loss,
                test_results.similarity,
                test_results.accuracy
            ))
        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="model.pt", help="the model to load")
    subparsers = parser.add_subparsers(dest="action")

    parser_random = subparsers.add_parser("random")
    parser_summary = subparsers.add_parser("summary")
    parser_details = subparsers.add_parser("details")

    parser_idx = subparsers.add_parser("idx")
    parser_idx.add_argument("idx", type=int, help="index in the testing set of the image to display")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Warning: CUDA not found. Using CPU")
    model = modules.ImprovedUNet(in_channels=1, out_channels=5, dropout_p=0.2)
    model.load_state_dict(torch.load(args.model, map_location=device))

    match args.action:
        case "random" | None:
            predict_random(model)
        case "idx":
            predict_idx(model, args.idx)
        case "summary":
            test_summary(model, device=device)
        case "details":
            test_details(model, device=device)
