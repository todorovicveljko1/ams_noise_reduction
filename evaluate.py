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

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    
    noise_gen = CompositeNoise(scale=0.1, value_prob=0.5)
    noise_gen.add(GaussianNoiseGen(0, 0.2), scale=0.5, min_clip=-1.0)
    noise_gen.add(PoissonNoiseGen(2), scale = 0.1)
    noise_gen.add(PepperSaltNoiseGen(0.005), scale = 0.5)
    
    trans_test = v2.Compose([v2.ToTensor()])
    dataset_test = CustomDataset('data/dataset/test', noise_gen, trans_test, device)
    test_lodaer = DataLoader(dataset=dataset_test, batch_size=batch_size, num_workers=0)


    snr_img = 0
    snr_output = 0
    ssi = 0
    mse = 0
    count = 0
    
    model = torch.load('models/unet_without_rot.pth', map_location=device)
    t = tqdm(test_lodaer)
    model.eval()
    with torch.no_grad():
        for i, (img, noisy_img, noise) in enumerate(t):
            output = model(noisy_img)
            imgs = img.squeeze().cpu().numpy()
            outputs = output.squeeze().cpu().numpy()
            for img, output in zip(imgs, outputs):
                snr_img += snr(img, output)
                #snr_output += snr(output)
                ssi += ssim(img, output, data_range=1)
                mse += mean_squared_error(img, output)
                count += 1
    print("PSNR: ", snr_img / count)
    print("SSIM: ", ssi / count)
    print("MSE", mse / count )
