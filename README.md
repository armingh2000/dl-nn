# Text2StyleGAN implementation

This repository is an implementation of the project found at Fast_text2StyleGAN, aimed at reproducing their results in image generation using StyleGAN.

# CLIP (Contrastive Language-Image Pre-Training)
CLIP is a deep learning model trained on image/text pairs so that it can connect images with their related text description. In other words, CLIP aims to encode images and text into the same encoding space. This is achieved by increasing the cosine similarity of the encodings of related pairs and reducing this value for unrelated pairs. An example of how this model works is can be seen in figure.

![A simplified visual example of what CLIP does. At first, it encodes "mountain landscape" using transformers and also encodes the image of the mountain using ViT and ResNet. Now as the tensors on the right side show, the encodings of the text and image are almost similar. This is because the text and image are related to each other.](/assets/clip.png)

# Files Overview
- `dataset.py`: Handles dataset processing and preparation.
- `loss.py`: Contains the implementation of loss functions used in the model.
- `main.py`: The main script for running the training and evaluation processes.
- `model.py`: Defines the neural network model architecture.
- `network.py`: Includes network configurations and settings.

# Getting Started

To get started with this project, clone the repository and install the necessary dependencies.
```bash
git clone https://github.com/armingh2000/dl-nn.git
cd dl-nn
```
# Install Dependencies
- [ ] Update this later



# How to Run?
```bash
python3 main.py
```
