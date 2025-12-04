# Gender-Detection using CNN

A deep learning project that classifies gender from facial images using a custom CNN. Includes real-time webcam detection with Haar Cascade face detection.

## Features

- Custom 5-layer CNN architecture
- Real-time webcam detection with bounding boxes
- 97.1% accuracy on test set
- Lightweight model (1.62 MB)
- Haar Cascade integration for face detection

## Dataset

- **Source**: [Kaggle Gender Dataset](https://www.kaggle.com/datasets/yasserhessein/gender-dataset)
- **Training**: 20,000 images (10K per class)
- **Validation**: 4,000 images (2K per class)
- **Test**: 4,000 images (2K per class)

## Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 97.10% |
| **Female Precision/Recall** | 0.98 / 0.96 |
| **Male Precision/Recall** | 0.96 / 0.98 |
| **Model Size** | 1.62 MB |
| **Parameters** | 423,873 |


## Model Architecture

- 5 Conv blocks (64→128→256→512→512 filters)
- Batch Normalization + Dropout
- Global Average Pooling
- Binary classification output

**Hyperparameters**: 128×128 images, batch size 32, Adam optimizer (lr=0.001)

## Acknowledgments

- Dataset: [Yasir Hussain Shekhar](https://www.kaggle.com/datasets/yasserhessein/gender-dataset)
- Built with TensorFlow and OpenCV
