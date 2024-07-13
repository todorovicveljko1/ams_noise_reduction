import torch
import torch.utils.data as data
import numpy as np
import os
import random
from PIL import Image
from torchvision.transforms import v2



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
        img = np.array(Image.open(self.imgs[index]), np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        x,y = img.shape[0], img.shape[1]
        noise = self.noise_gen.noise(x,y)
        noisy_img = img + noise
        if self.transform is not None:
            img, noisy_img, noise = self.transform(img, noisy_img, noise)
            #noisy_img = self.transform(noisy_img) #???
           # noise = self.transform(noise)
        # to device
        img = img.to(self.device)

        # mask = torch.bernoulli(torch.full((x,y), 0.5)) # 50% -> 0 or 1
        # noisy_img = noisy_img*mask
        noisy_img = noisy_img.to(self.device)

        noise = noise.to(self.device)

        return img, noisy_img, noise

    def __len__(self):
        return len(self.imgs)
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from noise_gen import GaussianNoiseGen, CompositeNoise, PepperSaltNoiseGen, PoissonNoiseGen

    noise_gen = CompositeNoise(scale=0.1, value_prob=0.5)
    noise_gen.add(GaussianNoiseGen(0, 0.2), scale=0.5, min_clip=-1.0)
    noise_gen.add(PoissonNoiseGen(2), scale = 0.1)
    noise_gen.add(PepperSaltNoiseGen(0.005), scale = 0.5)
    # scale to [0, 1]
    trans = v2.Compose([v2.ToTensor()])
    dataset = CustomDataset('data\\dataset\\train', noise_gen, trans)
    img, noisy_img, noise = dataset[0]
    # show img, noise and noise mask next to each other
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img.numpy().transpose(1,2,0))
    plt.subplot(1,3,2)
    plt.imshow(noise.numpy().transpose(1,2,0))
    plt.subplot(1,3,3)
    plt.imshow(noisy_img.numpy().transpose(1,2,0))
    plt.show()