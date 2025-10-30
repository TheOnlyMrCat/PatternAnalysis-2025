# 2D Improved UNet on the HipMRI Study on Prostate Cancer

Author: Max Guppy (48009069)

This folder implements a 2D Improved UNet based on the brain tumor segmentation model by Isensee et al. [\[1\]](#cite-1).
It differs from a standard UNet in a few ways.
In the encoder stage, pre-activation residual blocks are used between the downscaling stages, and in the decoder stage, outputs from all stages are aggregated to construct the final segmentation.
Both of these features allow gradients to flow through the network more easily, making it faster to train.
Additionally, in the 'context blocks' in the encoder stage, and in the decoder blocks, there are instance normalisation layers and dropout layers in between the convolution layers.
For downsampling, we use 5x5 convolution layers with stride 2.

![
  A diagram of the model's architecture.
  There are four encoder stages, starting with 32 output channels and doubling the number of channels after each stage.
  The three decoder stages feed into each other like in a normal UNet, but also a 1x1 convolution layer.
  The outputs of the three 1x1 convolution layers are upscaled and combined to produce the output.
](static/Improved_UNet.svg)

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

## Results

Training is very rapid for the first two epochs, then plateaus at a loss of about 0.05.

![
  A plot of the training loss of several training runs of the model.
  At the very left, the loss starts very high (at 0.7) but quickly decreases to less than 0.1 for the remainder of the plot.
  Scattered throughout the plateau section are many 'spikes' where the loss jumps to about 0.2 for a few steps and then drops back down.
](static/training_loss.png)

Validation loss plateaus slightly higher, at about 0.08, and fluctuates as training continues.
Validation accuracy (measured by percentage of pixels correctly segmented) tends to remain consistent at just under 0.97.
This suggests not too much overfitting is happening with this learning rate.

![
  A plot of the validation loss of several training runs of the model.
  Values fluctuate by about 0.05 in every epoch, but tend to start at 0.1 and decrease to 0.08.
](static/val_loss.png)

![
  A plot of the validation accuracy of several training runs of the model.
  Values tend to start at 0.95, then plateau at 0.97.
  A couple of runs have downwards spikes by 0.01 or 0.02 some epochs.
](static/val_accuracy.png)

The best run seems to be `celestial-pyramid-12`, which had a learning rate of 0.0005 (compared to the other runs' 0.001).

Here is its output on image 157 in the test set (`case_040_week_2_slice_42.nii.gz`):

![
  A plot showing the output of the model.
  The plot has two rows and seven columns.
  On the left of each row is a slice of an MRI image of a man's pelvis.
  To the right of each image are five images showing which parts of the image are background, body, bone, bladder, and prostate.
  On the very right of each row is a coloured combined segmentation of the entire image.
](static/celestial_pyramid_12_157.png)

## References

<a id="cite-1">[1]</a> F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation
and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,” Feb. 2018. \[Online\].
Available: https://arxiv.org/abs/1802.10508v1
