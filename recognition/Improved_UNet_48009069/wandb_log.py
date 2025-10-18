import os
import torch

try:
    import wandb
except ImportError:
    wandb = None


entity_name = os.getenv("WANDB_ENTITY")
project_name = os.getenv("WANDB_PROJECT")


class WandbContext():
    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass

    def step(self, loss):
        self.log({"loss": loss})

    def save_model(self, model):
        torch.save(model.state_dict(), "model.pt")


class ConnectedContext(WandbContext):
    def __init__(self, run):
        self.run = run

    def log(self, *args, **kwargs):
        self.run.log(*args, **kwargs)

    def finish(self):
        self.run.finish()

    def save_model(self, model):
        super().save_model(model)
        self.run.log_model("model.pt")


def setup(epochs: int, learning_rate: float) -> WandbContext:
    if wandb is not None:
        run = wandb.init(
            entity=entity_name,
            project=project_name,
            # Track hyperparameters and run metadata.
            config={
                "learning_rate": learning_rate,
                "architecture": "UNet",
                "dataset": "OASIS",
                "epochs": epochs,
            },
        )
        return ConnectedContext(run)
    else:
        return WandbContext()
