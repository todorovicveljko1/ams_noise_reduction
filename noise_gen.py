from abc import ABC, abstractmethod
import os
import numpy as np

class NoiseGen(ABC):
    """
    Abstract class for noise generator
    """
    @abstractmethod
    def noise(self, x, y):
        """
        Generate noise with given size
        :param x: row size
        :param y: column size
        :return: noise matrix
        """
        raise NotImplementedError

class PoissonNoiseGen(NoiseGen):
    """
    Poisson noise generator note that this generate integer noise >= 0, so it needs to be scaled   
    """
    def __init__(self, lam):
        self.lam = lam 

    def noise(self, x, y):
        return np.random.poisson(self.lam, (x, y))

class PepperSaltNoiseGen(NoiseGen):
    """
    Pepper and salt noise generator
    """
    def __init__(self, p):
        self.p = p

    def noise(self, x, y):
        return np.random.binomial(1, self.p, (x, y))

class GaussianNoiseGen(NoiseGen):
    """
    Gaussian noise generator
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def noise(self, x, y):
        return np.random.normal(self.mean, self.std, (x, y)) 

class CompositeNoise(NoiseGen):
    """
    Composite noise generator
    """
    def __init__(self, min_clip = 0.0, max_clip = 1.0, scale=1.0, value_prob=1.0) -> None:
        self.noises = []
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.scale = scale
        self.value_prob = value_prob

    def add(self, noise, offset=0.0, scale=1.0, prob=1.0, value_prob=1.0, min_clip=0.0, max_clip=1.0):
        """
        Add noise to composite noise generator
        :param noise: noise generator
        :param offset: offset
        :param scale: scale
        :param prob: probability of applying this noise
        :param value_prob: probability of applying this noise to each pixel
        :param min_clip: minimum clamp
        :param max_clip: maximum clamp
        """
        self.noises.append({
            'noise': noise,
            'offset': offset,
            'scale': scale,
            'prob': prob,
            'value_prob': value_prob,
            'min_clip': min_clip,
            'max_clip': max_clip
        })
    
    def noise(self, x, y):
        noise = np.zeros((x, y), np.float32)
        for n in self.noises:
            if np.random.rand() > n['prob']: # skip this noise
                continue
            temp_noise = n['noise'].noise(x, y) * n['scale'] + n['offset']
            if n['value_prob'] < 1.0:
                temp_noise = np.where(np.random.rand(x, y) < n['value_prob'], temp_noise, 0.0)
            noise += np.clip(temp_noise , n['min_clip'], n['max_clip'])
        if self.value_prob < 1.0:
            noise = np.where(np.random.rand(x, y) < self.value_prob, noise, 0.0)
        return np.clip(self.scale*noise, self.min_clip, self.max_clip)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image

    noise_gen = CompositeNoise(scale=0.1, value_prob=0.5)
    noise_gen.add(GaussianNoiseGen(0, 0.2), scale=0.5, min_clip=-1.0)
    noise_gen.add(PoissonNoiseGen(2), scale = 0.1)
    noise_gen.add(PepperSaltNoiseGen(0.005), scale = 0.5)
    
    noise = noise_gen.noise(224, 224)
    plt.imshow(noise, cmap='gray')
    plt.show()
    plt.hist(noise.flatten(), bins=100)
    plt.show()
    # Check if file exists
    if not os.path.exists('temp/example.png'):
        exit()

    img = Image.open('temp/example.png')
    
    #trans = transforms.Compose([transforms.ToTensor()])
    #img = trans(img)
    img = np.array(img)
    img = (img - img.min()) / (img.max() - img.min())
    # img crop to multiple of 32
    img = img[:(img.shape[0] - img.shape[0]%32), :(img.shape[1] - img.shape[1]%32)]
    #print(img.shape)
    # add noise

    noise = noise_gen.noise(img.shape[0], img.shape[1])

    noisey_img = img + noise

    print('img std', img.std(), 'mean', img.mean())
    print('noisey_img std', noisey_img.std(), 'mean', noisey_img.mean())
    print('noise std', noise.std(), 'mean', noise.mean())

    # show img, noisey_img
    plt.figure()

    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')

    plt.subplot(1,3,2)
    plt.imshow(noisey_img, cmap='gray')

    plt.subplot(1,3,3)
    plt.imshow(noise, cmap='gray')

    plt.show()



