# Medical Image Denoising

## Introduction

## Ideas

- Use mask so we only select patches inside lungs
- Generateing N patches from each image
- Noise used: 
    - poisson
    - gaussian
    - peper and salt.
- Models:
    - autoencoder
    - UNet
- Loss:
    - MSE
    - ABS
    - [SSIM - structural similarity index measure](https://en.wikipedia.org/wiki/Structural_similarity)

## Questions
- Is the noise distribution is different on different parts of the tissue?
- Do we need to use small amount of corctive noise (-noise)?

## TODO:
- Clean code 
- folder structure
- write main cli 

  