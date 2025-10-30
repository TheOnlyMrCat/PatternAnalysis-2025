## Running

### Dependencies

This project was tested with:

- Python 3.13.7
- NumPy 2.3.4
- Matplotlib 3.10.7
- PyTorch 2.9.0
- TorchVision 0.24.0
- NiBabel 5.3.2
- TQDM 4.67.1

These dependencies are listed in `requirements.txt`, and can be installed with:

```sh
$ pip install -r requirements.txt
```

Optionally, experiments can be logged to [Weights & Biases](https://wandb.ai/).
This is done automatically if `wandb` is installed, which can be done with `pip install wandb`.

(If `wandb` is installed but you do not want to upload runs, you can run `wandb offline` in this directory to turn off run syncing.)

### Training

The model can be trained with:

```sh
$ export HIPMRI_ROOT=/path/to/hipmri/keras_slices_data
$ python train.py
```

This will output a model to `model.pt`.
The model will be saved every epoch, in case training is interrupted for any reason.
If using W&B, the model is also uploaded as an artifact every epoch.

### Inference

The `predict.py` script demonstrates inference with the test set.
It has several subcommands, which can be run as follows:

```sh
$ export HIPMRI_ROOT=/path/to/hipmri/keras_slices_data
$ python predict.py [-m <path/to/model.pt>] <random|idx|summary|details>
```

The modes do the following:

- `random` will choose a random image from the test set and plot the predicted segmentation against the ground truth.
- `idx <INDEX>` will choose the `INDEX`th image from the test set and plot the predicted segmentation against the ground truth.
- `summary` will run inference on the entire test set and show a summary of the accuracy (similar to the summary printed at the end of training).
- `details` will run inference on the entire test set and output a detailed log of loss/similarity/accuracy to `test.csv`
