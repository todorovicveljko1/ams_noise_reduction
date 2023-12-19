import torch
import torch.utils.data as data
import numpy as np
import os
import random
from PIL import Image
import torchvision.transforms as transforms


class CustomDataset(data.Dataset):
    """
    Dataset class for training and testing
    """
    def __init__(self, root, noise_gen, transform=None, device='cpu'):
        """
        :param root: root directory of dataset
        :param noise_gen: noise generator
        :param transform: data augmentation
        :param device: device to run on
        """
        self.root = root
        self.noise_gen = noise_gen
        self.transform = transform
        self.imgs = [os.path.join(root, img) for img in os.listdir(root)]
        self.device = device

    def __getitem__(self, index):
        """
        :param index: index of image
        :return: image, noisy image, noise mask
        """
        img = Image.open(self.imgs[index])
        if self.transform is not None:
            img = self.transform(img)
        img = (img - img.min()) / (img.max() - img.min())
        noise = self.noise_gen.noise(img.shape[1], img.shape[2])
        noisy_img = img + noise
        noisy_img = torch.clamp(noisy_img, 0.0, 1.0)

        # to device
        img = img.to(self.device)
        noisy_img = noisy_img.to(self.device)
        noise = noise.to(self.device)

        return img, noisy_img, noise

    def __len__(self):
        return len(self.imgs)
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from noise_gen import GaussianNoiseGen, CompositeNoise, PepperSaltNoiseGen, PoissonNoiseGen
    from torchvision import transforms
    noise_gen = CompositeNoise()
    noise_gen.add(GaussianNoiseGen(0, 0.2), 0, 0.3, 1, 1, -100, 255)
    noise_gen.add(PoissonNoiseGen(2), 0, 0.1, 1, 0.3, -100, 255)
    noise_gen.add(PepperSaltNoiseGen(0.001), 0, 0.5, 1, 1, 0, 255)
    # scale to [0, 1]
    trans = transforms.Compose([transforms.ToTensor()])
    dataset = CustomDataset('data\\dataset\\train', noise_gen, trans)
    img, noisy_img, noise = dataset[0]
    # show img, noise and noise mask next to each other
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img.numpy().transpose(1,2,0))
    plt.subplot(1,3,2)
    plt.imshow(noise.numpy())
    plt.subplot(1,3,3)
    plt.imshow(noisy_img.numpy().transpose(1,2,0))
    plt.show()