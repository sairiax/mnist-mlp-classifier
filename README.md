# MNIST Multi-Layer Perceptron (99.43% Accuracy)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Accuracy](https://img.shields.io/badge/Accuracy-99.43%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-grey)

## Project Overview

This repository contains a high-performance implementation of a Multi-Layer Perceptron (MLP) for the MNIST dataset. The objective was to push the limits of a pure fully connected architecture without utilizing Convolutional Neural Networks (CNNs). Through rigorous hyperparameter tuning and modern regularization techniques, this model achieves a **test accuracy of 99.43%**, comparable to standard CNN benchmarks.

## Technical Implementation

The project implements several advanced deep learning strategies to maximize generalization and convergence speed:

*   **Deep Architecture:** A 4-layer MLP with 1024 neurons per hidden layer to capture complex non-linear patterns.
*   **Regularization Pipeline:** Integrated **Batch Normalization** and **Dropout (0.2)** to mitigate overfitting in a high-parameter space.
*   **Data Augmentation:** Applied `RandomRotation` and `RandomAffine` transformations during training to enhance model robustness.
*   **Optimization Strategy:** Utilized the **AdamW** optimizer with **OneCycleLR** scheduling for efficient super-convergence.
*   **Training Control:** Implemented Early Stopping based on validation accuracy to ensure optimal model selection.

## Model Architecture

The network consists of a feed-forward architecture designed for high expressivity:

```python
Input (784) -> [Linear(1024) -> BatchNorm1d -> ReLU -> Dropout] x 3 -> Linear(10)
```

*   **Input:** 784 features (Flattened 28x28 images)
*   **Hidden Layers:** 3 x 1024 units
*   **Output:** 10 classes (Softmax)

## Performance Metrics

The model was evaluated on the standard MNIST test set (10,000 images).

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **99.43%** |
| **Precision** | 99.43% |
| **Recall** | 99.43% |
| **F1-Score** | 99.43% |

## Installation and Usage

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/mnist-mlp-sota.git
    cd mnist-mlp-sota
    ```

2.  **Install dependencies**
    ```bash
    pip install torch torchvision matplotlib seaborn scikit-learn numpy
    ```

3.  **Execute**
    Run the `mnist.ipynb` notebook to reproduce the training pipeline and evaluation metrics.

## Author

**Sergi Cortés Guerrero**
*   [LinkedIn](https://www.linkedin.com/in/scorgue/)
*   [GitHub](https://github.com/sairiax/)
