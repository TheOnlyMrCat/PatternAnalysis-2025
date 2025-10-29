import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

import dataset
import modules
import wandb_log

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Warning: CUDA not found. Using CPU")

dataset_root = os.getenv("HIPMRI_ROOT")
if dataset_root is None:
    print("error: need $HIPMRI_ROOT to be set to root path of dataset")
    exit(1)

print(f"> Loading training/validation datasets from {dataset_root}")

trainset, valset = dataset.load_train_val(dataset_root)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(valset, batch_size=16, shuffle=True)

# Hyper-parameters
epochs = 24
learning_rate = 1e-3
dropout_p = 0.3

run = wandb_log.setup(epochs, learning_rate, dropout_p)

model = modules.ImprovedUNet(in_channels=1, out_channels=5, dropout_p=dropout_p)
model.to(device)
criterion = modules.MulticlassDiceLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("> Starting training")
for epoch in range(epochs):
    model.train()

    # Training loop with progress
    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        run.step(loss.item())

    del images, masks, loss, outputs

    # Validate model every epoch
    model.eval()
    losses = []
    accuracies = []
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            losses.append(criterion(outputs, masks))

            seg = torch.argmax(masks, 1)
            predicted_seg = torch.argmax(outputs, 1)
            accuracies.append(torch.sum(seg == predicted_seg) / seg.numel())

    avg_loss = sum(losses) / len(losses)
    accuracy = sum(accuracies) / len(accuracies)
    run.epoch(epoch + 1, avg_loss, accuracy, model)

    print(f"📈 Epoch {epoch+1}/{epochs} Complete: Avg Loss = {avg_loss:.4f}; Accuracy = {accuracy * 100:.2f}%")

del images, masks, losses, outputs
run.finish()

print(f"> Loading test dataset from {dataset_root}")
testset = dataset.load_test(dataset_root)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

print("> Testing model")
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
            1 - modules.DiceLoss()(outputs[:, ch], masks[:, ch]).cpu()
            for ch in range(5)
        ])

        seg = torch.argmax(masks, 1)
        predicted_seg = torch.argmax(outputs, 1)
        accuracies.append(torch.sum(seg == predicted_seg) / seg.numel())

avg_loss = (sum(losses) / len(losses)).cpu()
accuracy = (sum(accuracies) / len(accuracies)).cpu()
sim = np.asarray(similarities)
print("Training complete!")
print(f"Average loss: {avg_loss:.5f}")
print("Dice Similarity Coefficients (min/avg/max):")
print(f"- Background: {np.min(sim[:, 0]):.3f}/{np.sum(sim[:, 0]) / sim.shape[0]:.3f}/{np.max(sim[:, 0]):.3f}")
print(f"- Body: {np.min(sim[:, 1]):.3f}/{np.sum(sim[:, 1]) / sim.shape[0]:.3f}/{np.max(sim[:, 1]):.3f}")
print(f"- Bone: {np.min(sim[:, 2]):.3f}/{np.sum(sim[:, 2]) / sim.shape[0]:.3f}/{np.max(sim[:, 2]):.3f}")
print(f"- Bladder: {np.min(sim[:, 3]):.3f}/{np.sum(sim[:, 3]) / sim.shape[0]:.3f}/{np.max(sim[:, 3]):.3f}")
print(f"- Prostate: {np.min(sim[:, 4]):.3f}/{np.sum(sim[:, 4]) / sim.shape[0]:.3f}/{np.max(sim[:, 4]):.3f}")
print(f"Accuracy: {accuracy * 100:.1f}%")
