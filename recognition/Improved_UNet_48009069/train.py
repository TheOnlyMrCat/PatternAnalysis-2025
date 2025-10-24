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

dataset_root = os.getenv("HIPMRI_ROOT")
if dataset_root is None:
    print("error: need $HIPMRI_ROOT to be set to root path of dataset")
    exit(1)

print(f"> Load dataset from {dataset_root}")

trainset, valset = dataset.load_train_val(dataset_root)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)

# Hyper-parameters
epochs = 12
learning_rate = 1e-3
dropout_p = 0.3

run = wandb_log.setup(epochs, learning_rate, dropout_p)

model = modules.ImprovedUNet(in_channels=1, out_channels=5, dropout_p=dropout_p)
model.to(device)
criterion = modules.WeightedDiceLoss([1.0, 1.0, 1.0, 1.0, 1.0])
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

run.finish()
