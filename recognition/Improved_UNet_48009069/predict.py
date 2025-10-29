import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
from tqdm import tqdm

import dataset
import modules


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
    for ch in range(5):
        # Ground truth segmentation
        axes[0][ch + 1].imshow(seg[ch].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[0][ch + 1].set_title(f"Seg ch {ch}")
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
    axes[0][6].set_title("Ground truth seg")
    axes[0][6].axis("off")
    axes[1][6].imshow(predict_full.cpu().numpy(), cmap="tab10")
    axes[1][6].set_title(f"Predicted (accuracy: {accuracy * 100:.1f}%)")
    axes[1][6].axis("off")

    plt.tight_layout()
    plt.show()


def predict_random(model):
    dataset_root = os.getenv("HIPMRI_ROOT")
    if dataset_root is None:
        print("error: need $HIPMRI_ROOT to be set to root path of dataset")
        exit(1)
    testset = dataset.load_test(dataset_root)

    image, seg = random.choice(testset)
    segment_image(model, image, seg)


def test_summary(model):
    dataset_root = os.getenv("HIPMRI_ROOT")
    if dataset_root is None:
        print("error: need $HIPMRI_ROOT to be set to root path of dataset")
        exit(1)
    testset = dataset.load_test(dataset_root)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

    criterion = modules.MulticlassDiceLoss()
    model.eval()
    losses = []
    similarities = []
    accuracies = []
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(test_loader)):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            losses.append(criterion(outputs, masks))
            similarities.append([
                1 - modules.DiceLoss()(outputs[:, ch], masks[:, ch])
                for ch in range(5)
            ])

            seg = torch.argmax(masks, 1)
            predicted_seg = torch.argmax(outputs, 1)
            accuracies.append(torch.sum(seg == predicted_seg) / seg.numel())

    avg_loss = sum(losses) / len(losses)
    accuracy = sum(accuracies) / len(accuracies)
    sim = np.asarray(similarities)
    print("Test complete!")
    print(f"Average loss: {avg_loss:.5f}")
    print("Dice Similarity Coefficients (min/avg/max):")
    print(f"- Background: {np.min(sim[:, 0]):.3f}/{np.sum(sim[:, 0]) / sim.shape[0]:.3f}/{np.max(sim[:, 0]):.3f}")
    print(f"- Body: {np.min(sim[:, 1]):.3f}/{np.sum(sim[:, 1]) / sim.shape[0]:.3f}/{np.max(sim[:, 1]):.3f}")
    print(f"- Bone: {np.min(sim[:, 2]):.3f}/{np.sum(sim[:, 2]) / sim.shape[0]:.3f}/{np.max(sim[:, 2]):.3f}")
    print(f"- Bladder: {np.min(sim[:, 3]):.3f}/{np.sum(sim[:, 3]) / sim.shape[0]:.3f}/{np.max(sim[:, 3]):.3f}")
    print(f"- Prostate: {np.min(sim[:, 4]):.3f}/{np.sum(sim[:, 4]) / sim.shape[0]:.3f}/{np.max(sim[:, 4]):.3f}")
    print(f"Accuracy: {accuracy * 100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="model.pt", help="the model to load")
    parser.add_argument("action", choices=["random", "summary"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Warning: CUDA not found. Using CPU")
    model = modules.ImprovedUNet(in_channels=1, out_channels=5, dropout_p=0.2)
    model.load_state_dict(torch.load(args.model, map_location=device))

    match args.action:
        case "random":
            predict_random(model)
        case "summary":
            test_summary(model)
