import argparse
import matplotlib.pyplot as plt
import os
import random
import torch

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
        axes[1][ch + 1].set_title(f"Predicted ch {ch}")
        axes[1][ch + 1].axis("off")

    # Create collated segmentation maps
    seg_full = torch.argmax(seg, 0)
    predict_full = torch.argmax(output, 0)
    accuracy = torch.sum(seg_full == predict_full) / seg_full.numel()

    axes[0][6].imshow(seg_full.cpu().numpy(), cmap="tab10")
    axes[0][6].set_title("Ground truth seg")
    axes[0][6].axis("off")
    axes[1][6].imshow(predict_full.cpu().numpy(), cmap="tab10")
    axes[1][6].set_title(f"Predicted seg (accuracy: {accuracy * 100:.1f}%)")
    axes[1][6].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="model.pt", help="the model to load")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Warning: CUDA not found. Using CPU")
    model = modules.ImprovedUNet(in_channels=1, out_channels=5, dropout_p=0.2)
    model.load_state_dict(torch.load(args.model, map_location=device))

    dataset_root = os.getenv("HIPMRI_ROOT")
    if dataset_root is None:
        print("error: need $HIPMRI_ROOT to be set to root path of dataset")
        exit(1)
    trainset, valset, testset = dataset.load_datasets(dataset_root)

    image, seg = random.choice(testset)
    segment_image(model, image, seg)
