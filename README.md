# Vision Transformer (ViT) - PyTorch Implementation

This repository offers a comprehensive implementation of the **Vision Transformer (ViT)**, as introduced in the seminal paper:

> **"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"**  
> *Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, et al.*  
> [Paper Link](https://arxiv.org/abs/2010.11929)

## 🚀 Features

- **Complete Implementation**: From scratch ViT model built using PyTorch.
- **Training Capabilities**: Train the ViT model on the CIFAR-10 dataset.
- **Fine-Tuning**: Adapt pre-trained ViT models for specific tasks using CIFAR-10.
- **Inference Pipeline**: Predict classes for input images using the trained model.
- **Configurable Parameters**: Easily adjust hyperparameters and model settings.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Fine-Tuning](#fine-tuning)
  - [Inference](#inference)
- [Dataset Preparation](#dataset-preparation)
- [Results](#results)
- [References](#references)

## 📌 Installation

- Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/ViT-Implementation.git
cd ViT-Implementation
```

* Ensure you have Python 3.7 or higher installed.

- Navigate to your project directory.

- Create a virtual environment (optional but recommended):

   ```bash
   python -m venv env
   ```

- Activate the virtual environment:

  - On Windows:

     ```bash
     env\Scripts\activate
     ```

  - On macOS/Linux:

     ```bash
     source env/bin/activate
     ```

- Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 📂 Project Structure

```
├── model                    # Vision Transformer (ViT) model implementation
├── train.py                 # Training script for ViT
├── finetune.py              # Fine-tuning script for pre-trained ViT
├── inference.py             # Inference script for predicting image classes
├── utils                    # Helper functions for data loading, training, and evaluation
├── config.py                # Configuration settings for hyperparameters
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

---

## Overview

The Vision Transformer (ViT) represents a paradigm shift in image classification by applying transformer architectures, traditionally used in natural language processing, to computer vision tasks. By dividing images into patches and treating them as sequences, ViT models can capture global relationships more effectively than traditional convolutional neural networks (CNNs).

## 📊 Model Architecture

The Vision Transformer (ViT) introduces a novel approach to image recognition by leveraging the Transformer architecture, traditionally used in natural language processing. The core idea is to treat image patches as sequences, akin to word tokens in text.

**Key Components:**

1. **Patch Embedding**: The input image is divided into fixed-size patches (e.g., 16x16 pixels). Each patch is flattened into a vector and linearly projected to form patch embeddings.

2. **Position Embedding**: Since Transformers lack inherent spatial understanding, learnable position embeddings are added to the patch embeddings to retain positional information.

3. **Transformer Encoder**: The sequence of embeddings is processed through multiple Transformer encoder layers, each comprising multi-head self-attention and feed-forward neural networks.

4. **Classification Head**: A special [CLS] token is prepended to the sequence, whose final state serves as the aggregate representation for classification tasks.

**Architecture Diagram:**

![Vision Transformer Architecture](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS44MHuNZGGJ7qsCL5OBKd-DvT1XspFjRlbRA&s)

*Figure: Vision Transformer (ViT) Architecture*

---

## 📖 Usage

### 1️⃣ Training from Scratch

To train a Vision Transformer model on CIFAR-10:

```bash
python train.py --batch_size 64 --learning_rate 3e-4 --num_epochs 50 --dataset_path ./data/CIFAR10
```

This command initializes and trains the ViT model using the specified hyperparameters.

---

### 2️⃣ Fine-Tuning a Pre-trained ViT

To fine-tune a pre-trained Vision Transformer model on CIFAR-10:

```bash
python finetune.py --checkpoint_path path/to/pretrained_model.pth --dataset_path ./data/CIFAR10
```

This process freezes the pre-trained weights except for the classifier head, adapting the model to the CIFAR-10 dataset.

---

### 3️⃣ Inference (Classify a Single Image)

To classify an image using a trained ViT model:

```bash
python inference.py --image_path path/to/image.jpg --checkpoint path/to/model.pth
```

For example:

```bash
python inference.py --image_path sample.jpg --checkpoint vit_cifar10.pth
```

This script loads the trained model and outputs the predicted class for the input image.

---

## 📝 Configuration

Hyperparameters and model configurations can be adjusted in `config.py`:

```python
class Config:
    IMG_SIZE = 224         # Input image size
    PATCH_SIZE = 16        # Patch size
    NUM_LAYERS = 12        # Number of Transformer encoder layers
    D = 768                # Embedding dimension
    FF_D = 3072            # Feed-forward dimension
    NUM_HEADS = 12         # Number of attention heads
    DROPOUT = 0.1          # Dropout rate
    LR = 3e-4              # Learning rate
    EPOCHS = 50            # Number of epochs
    BATCH_SIZE = 64        # Batch size
    NUM_CLASSES = 10       # Number of classes (e.g., CIFAR-10)
    DEVICE = 'cuda'        # Device to use ('cuda' or 'cpu')
    DATASET_PATH = './data/ImageNet'  # Path to dataset
    SAVE_DIR = './checkpoints'       # Directory to save models
```

Modify these parameters to tailor the training and model architecture to your specific requirements.

---

## 📚 Dataset

This project supports the **CIFAR-10** dataset, a widely used benchmark in computer vision.

**CIFAR-10 Dataset:**

- **Description**: Consists of 60,000 color images in 10 classes, with 6,000 images per class. The classes are mutually exclusive and do not overlap.
- **Classes**: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.
- **Image Size**: 32x32 pixels.

The dataset is automatically downloaded and processed when training or fine-tuning the model.

---

## 🔧 Future Improvements

- **Support for Larger Datasets**: Extend compatibility to datasets like ImageNet for more comprehensive training.
- **Data Augmentation Techniques**: Implement advanced augmentation strategies to enhance model robustness.
- **Self-Supervised Pretraining**: Explore self-supervised learning methods to improve feature representations.
- **Efficient Attention Mechanisms**: Investigate and integrate more efficient attention mechanisms to reduce computational overhead.

---

## 👨‍💻 Author

Developed by **[Mohamed Rezq]**

If you found this project useful, please consider ⭐ starring the repository!

---

## 📜 References

- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
