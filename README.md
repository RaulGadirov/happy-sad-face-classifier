# Happy/Sad Face Classifier

A simple fully-connected neural network built with Keras that classifies grayscale face images as **happy** or **sad**.

## Overview

This project trains a small feedforward neural network on a tiny custom dataset of 32×32 binary (black & white) face images to distinguish between two emotional expressions: happy and sad.

## Dataset

- Images are loaded from a `train/` directory, with filenames prefixed `happy_` or `sad_` to indicate the label.
- Each image is converted to 1-bit black & white mode (`convert('1')`) and flattened into a 1024-length pixel vector (32×32).
- Labels are one-hot encoded: `[1, 0]` for happy, `[0, 1]` for sad.
- Pixel values are normalized to the `[0, 1]` range.

## Model Architecture

A `Sequential` model with fully connected (`Dense`) layers:

| Layer  | Units | Activation |
|--------|-------|------------|
| Input  | 1024  | ReLU       |
| Hidden | 512   | ReLU       |
| Hidden | 256   | ReLU       |
| Hidden | 128   | ReLU       |
| Output | 2     | Softmax    |

- **Loss:** categorical cross-entropy
- **Optimizer:** Adam (learning rate = 0.005)
- **Metric:** accuracy

## Training

```python
model.fit(data, target, epochs=1000, batch_size=32, verbose=1)
```

## Inference

Test images are loaded the same way as training images, normalized, and passed to `model.predict()`:

```python
print(model.predict(test).round())
# [[1. 0.]] -> happy
# [[0. 1.]] -> sad
```

## Requirements

- Python 3
- TensorFlow / Keras
- Pillow (PIL)
- NumPy

## Notes

- This is a minimal proof-of-concept trained on a very small dataset (20 images), intended for learning purposes rather than production use.
- Because the dataset is so small, the model is prone to overfitting — results may not generalize well to new images.
