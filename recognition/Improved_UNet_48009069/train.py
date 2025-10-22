import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

import dataset
import modules
import wandb_log

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Warning: CUDA not found. Using CPU")

dataset_root = os.getenv("OASIS_ROOT")
if dataset_root is None:
    print("error: need $OASIS_ROOT to be set to root path of dataset")
    exit(1)

print(f"> Load dataset from {dataset_root}")

trainset, testset = dataset.load_datasets(dataset_root)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)

# Hyper-parameters
epochs = 12
learning_rate = 1e-3

run = wandb_log.setup(epochs, learning_rate)

model = modules.ImprovedUNet(in_channels=1, out_channels=4, dropout_p=0.3)
model.to(device)
criterion = modules.DiceLoss()
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

    # Test model every epoch
    model.eval()
    losses = []
    accuracies = []
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
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

run.finish()
