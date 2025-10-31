"""
Functions for handling logging for an experiment.
Logs to W&B if installed, otherwise only saves model to disk.

Author: Max Guppy (48009069)
"""

import os
import torch

try:
    import wandb
except ImportError:
    wandb = None


class ExperimentContext():
    """
    Abstract class which handles logging for an experiment.

    The default behaviour is to do nothing for regular logs, and to save model parameters to `model.pt` every epoch.
    """
    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass

    def step(self, loss):
        self.log({"loss": loss})

    def epoch(self, epoch, loss, accuracy, model):
        self.log({"val/loss": loss, "val/accuracy": accuracy})
        self.save_model(model, aliases=[f"epoch - {epoch}", f"val_accuracy - {accuracy}"])

    def save_model(self, model, aliases=None):
        torch.save(model.state_dict(), "model.pt")


class WandbContext(ExperimentContext):
    """
    ExperimentContext which logs progress to Weights & Biases (https://wandb.ai).

    Logs loss, validation loss, and validation accuracy every step/epoch.
    Uploads each saved `model.pt` as an artifact.
    """
    def __init__(self, run):
        self.run = run

    def log(self, *args, **kwargs):
        self.run.log(*args, **kwargs)

    def finish(self):
        self.run.finish()

    def save_model(self, model, aliases=None):
        super().save_model(model)
        self.run.log_model("model.pt", aliases=aliases)


def setup(epochs: int, learning_rate: float, dropout_p: float) -> ExperimentContext:
    """
    Create a new ExperimentContext based on whether `wandb` is installed or not.

    If `wandb` is installed, creates a `WandbContext` which logs loss and validation
    stats every step. The passed hyperparameters are tracked in the run config. The run
    is created in the project given by the environment variables `WANDB_ENTITY`
    (default: the currently logged-in user) and `WANDB_PROJECT` (default:
    "hipmri-improved-unet-48009069")

    If `wandb` is not installed, creates a default `ExperimentContext` which saves model
    parameters every epoch.
    """
    if wandb is not None:
        entity_name = os.getenv("WANDB_ENTITY")
        project_name = os.getenv("WANDB_PROJECT", "hipmri-improved-unet-48009069")

        run = wandb.init(
            entity=entity_name,
            project=project_name,
            # Track hyperparameters and run metadata.
            config={
                "learning_rate": learning_rate,
                "architecture": "Improved UNet",
                "dataset": "HipMRI",
                "epochs": epochs,
                "dropout_p": dropout_p,
            },
        )
        return WandbContext(run)
    else:
        return ExperimentContext()
