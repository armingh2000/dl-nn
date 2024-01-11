# Image Generation Using Text Input

The goal of this project is to train a neural network to generate face images using text input to the model.

*This repository is an implementation of the project found at [Fast_text2StyleGAN](https://github.com/duxiaodan/Fast_text2StyleGAN), aimed at reproducing their results in image generation using StyleGAN.*

# Summary
We use CLIP(Contrastive Language-Image Pre-Training) model as a base and build on top of it. As we know, CLIP, maps text and image data to the same latent space. We use this leverage to train a neural network to decode any point in the latent space as an output image.

In other words, we use these steps:

    1. Encode a face image using CLIP model.
    2. Train a neural network to decode the encoded vector in the latent space back to the image domain.

We know that CLIP model maps the image and the related text to the same point in the latent space. Hence, we can now input text into the model and recieve the same latent vector for it. Using the decoder network, we can now decode the encoded text (corresponding to the actual image) and get the desired image.

## Results
| Input Text               | Image 1                  | Image 2                  | Image 3                  |
|--------------------------|--------------------------|--------------------------|--------------------------|
| a photo of a cute child                 | ![baby image 1](assets/baby1.png) | ![baby image 2](assets/baby2.png) | ![baby image 3](assets/baby3.png) |
| a photo of a happy woman with long hair | ![baby image 1](assets/woman1.png) | ![baby image 2](assets/woman2.png) | ![baby image 3](assets/woman3.png) |
| a photo of an old man with white hair   | ![baby image 1](assets/man1.png) | ![baby image 2](assets/man2.png) | ![baby image 3](assets/man3.png) |




# Files Overview
- `dataset.py`: Handles dataset processing and preparation.
- `loss.py`: Contains the implementation of loss functions used in the model.
- `main.py`: The main script for running the training and evaluation processes.
- `model.py`: Defines the neural network model architecture.
- `network.py`: Includes network configurations and settings.

# License
The source code for the site is licensed under the MIT license, which you can find in the LICENSE file.
