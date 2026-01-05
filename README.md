# Dog and Cat Classification

A deep learning project for binary image classification to distinguish between dogs and cats using transfer learning with MobileNetV2.

## Overview

This project implements a Convolutional Neural Network (CNN) for classifying images of dogs and cats. It uses transfer learning with a pre-trained MobileNetV2 model as the base, achieving approximately **96.5% accuracy** on the test set.

## Dataset

The project uses the [Dogs vs. Cats dataset from Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data), which contains:
- 25,000 labeled images (12,500 dogs and 12,500 cats) in the training set
- For this project, 2,000 images are used for training (subset for faster training)
- Images are resized to 224x224 pixels for compatibility with MobileNetV2

## Model Architecture

The model uses transfer learning with the following architecture:

1. **Base Model**: MobileNetV2 (pre-trained on ImageNet, with top layers removed)
   - Input shape: (224, 224, 3)
   - Weights frozen during training
2. **Global Average Pooling Layer**: Reduces spatial dimensions
3. **Dense Layer**: 2 output units for binary classification (Cat/Dog)

**Total Parameters**: ~2.26 million (8.62 MB)
**Trainable Parameters**: 2,562 (only the final dense layer)

## Requirements

- Python 3.x
- TensorFlow / Keras
- NumPy
- OpenCV (cv2)
- Pillow (PIL)
- Matplotlib
- Kaggle API (for dataset download)

## Usage

This project is designed to run in **Google Colab**. To get started:

1. Open the notebook `dog_and_cat_classification_.ipynb` in Google Colab
2. Upload your `kaggle.json` API credentials file when prompted
3. Run all cells sequentially to:
   - Download and extract the dataset
   - Preprocess and resize images
   - Train the model
   - Evaluate on test data
   - Make predictions on new images

### Making Predictions

Use the `predict()` function to classify new images:
```python
predict()
# Enter the path to your image when prompted
```

The model will output whether the image is a **Cat** or a **Dog**.

## Results

| Metric | Value |
|--------|-------|
| Training Accuracy (after 5 epochs) | ~98.4% |
| Test Accuracy | ~96.5% |
| Test Loss | 0.083 |

## Project Structure

```
dog-and-cat-classification/
├── README.md                           # This file
└── dog_and_cat_classification_.ipynb   # Main Jupyter notebook with all code
```

## License

This project is open source and available for educational purposes.
