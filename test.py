import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU()
        )
        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )
        self.conv = nn.Conv2d(64, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(0.2)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        x = self.upconv3(x3)
        x = self.upconv2(x + self.dropout(x2))
        x = x + self.dropout(x1)
        x = self.conv(x)
        x = self.sigmoid(x)
        return x
    
if __name__ == "__main__":
    
    # model = UNet()
    # model.load_state_dict(torch.load("models/l1_unet_model.pth", map_location=torch.device('cpu')))
    # model.eval()
    # img = Image.open('temp/example.png')
    # trans = transforms.Compose([transforms.ToTensor()])
    # img = trans(img)

    # img = (img - img.min()) / (img.max() - img.min())
    # # img crop to multiple of 32
    # img = img[:, :img.shape[1]//32*32, :img.shape[2]//32*32]

    # with torch.no_grad():
    #     out = model(img.unsqueeze(0))

    # # show img, out
    # # plt.figure()
    # # plt.subplot(1,2,1)
    # # plt.imshow(img.numpy().transpose(1,2,0), cmap='gray')
    # # plt.subplot(1,2,2)
    # # plt.imshow(out.squeeze().numpy(), cmap='gray')
    # # plt.show()

    # # save images
    # out = Image.fromarray(np.uint8(out.squeeze().numpy()*255))
    # out.save('temp/l1_example_output.png')

    # load original image and output image and show difference
    img = Image.open('temp/example.png')
    img = np.array(img)
    img = img[:img.shape[0]//32*32, :img.shape[1]//32*32]
    out = Image.open('temp/example_output.png')
    out = np.array(out)
    # sinal to noise ratio for image mean/std
    print(img.mean() / img.std())
    print(out.mean() / out.std())
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(out, cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(np.abs(img-out), cmap='gray')
    plt.show()