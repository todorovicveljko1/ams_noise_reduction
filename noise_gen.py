from abc import ABC, abstractmethod
import numpy as np
import torch
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
    Poisson noise generator
    """
    def __init__(self, lam):
        self.lam = lam 

    def noise(self, x, y):
        return torch.poisson(torch.ones(1) * self.lam)

class PepperSaltNoiseGen(NoiseGen):
    """
    Pepper and salt noise generator
    """
    def __init__(self, p):
        self.p = p

    def noise(self, x, y):
        return torch.bernoulli(torch.ones(x, y) * self.p)

class GaussianNoiseGen(NoiseGen):
    """
    Gaussian noise generator
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def noise(self, x, y):
        return torch.normal(self.mean, self.std, (x, y))

class CompositeNoise(NoiseGen):
    """
    Composite noise generator
    """
    def __init__(self, min_clip = 0., max_clip = 1., scale=1.) -> None:
        self.noises = []
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.scale = scale

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
        noise = torch.zeros((x, y))
        for n in self.noises:
            if np.random.rand() < n['prob']:
                temp_noise = n['noise'].noise(x, y) * n['scale'] + n['offset']
                if n['value_prob'] < 1.0:
                    temp_noise = np.where(np.random.rand(x, y) < n['value_prob'], temp_noise, 0.0)
                noise += np.clip(temp_noise , n['min_clip'], n['max_clip'])
        return np.clip(self.scale*noise, self.min_clip, self.max_clip)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    noise_gen = CompositeNoise()
    noise_gen.add(GaussianNoiseGen(0, 0.1), 0, 1, 1, 1, -100, 255)
    noise_gen.add(PoissonNoiseGen(1), 0, 1/12, 1, 0.3, -100, 255)
    noise_gen.add(PepperSaltNoiseGen(0.01), 0, 0.7, 1, 1, 0, 255)
    noise = noise_gen.noise(224, 224)
    print(noise)
    plt.imshow(noise.numpy(), cmap='gray')
    plt.show()
    plt.hist(noise.numpy().flatten(), bins=100)
    plt.show()