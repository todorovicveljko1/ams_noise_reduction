from data_loader import CustomDataset
from noise_gen import GaussianNoiseGen, CompositeNoise, PepperSaltNoiseGen, PoissonNoiseGen
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch
from models import UNet, AutoEncoder, DownSampleBlock, UpSampleBlock, UNetConfig
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as snr
from skimage.metrics import mean_squared_error as mean_squared_error
from torchvision.transforms import v2

import numpy as np
from PIL import Image
import leafmap

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trans = v2.Compose([v2.ToTensor()])
    
    model = torch.load('models/unet_without_rot.pth', map_location=device)
    model.eval()

    img = np.array( Image.open('example.png'), np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    img = img[:img.shape[0]//32*32, :img.shape[1]//32*32]
    img = Image.fromarray(np.uint8(img*255))
    img.save('example_cut.png')
    img = trans(img)

    
    with torch.no_grad():
        out = model(img.unsqueeze(0).to(device))
    out = out.squeeze().cpu().numpy()
    out = np.clip(out,0,1)
    out = Image.fromarray(np.uint8(out*255))
    out.save('example_output.png')


    leafmap.image_comparison(
        "example_cut.png",
        "example_output.png",
        label1="Original Image",
        label2="Denoised",
        starting_position=50,
        out_html="image_comparison.html",
    )