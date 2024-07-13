
import datetime
from data_loader import CustomDataset
from models import UNet, UNetConfig
from noise_gen import GaussianNoiseGen, CompositeNoise, PepperSaltNoiseGen, PoissonNoiseGen
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def train_iter(model, train_loader, criterion, optimizer, debug=False):
    model.train()
    train_loss = 0.0
    t = tqdm(train_loader)
    losses = []
    for i, (img, noisy_img, noise) in enumerate(t):
        optimizer.zero_grad()
        output = model(noisy_img)
        # print(output.shape)
        loss = criterion(output, img)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if i % 10 == 0:
            losses.append(loss.item())
            if debug:
                t.write(f" loss {loss.item():.6f} at iter: {i:4d}")
    return train_loss / len(train_loader), losses


def validate_iter(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, (img, noisy_img, noise) in enumerate(val_loader):
            output = model(noisy_img)
            loss = criterion(output, img)
            val_loss += loss.item()
    return val_loss / len(val_loader)


if __name__ == "__main__":
    # PARAMS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    # NOISE GEN
    noise_gen = CompositeNoise(scale=0.1, value_prob=0.5)
    noise_gen.add(GaussianNoiseGen(0, 0.2), scale=0.5, min_clip=-1.0)
    noise_gen.add(PoissonNoiseGen(2), scale=0.1)
    noise_gen.add(PepperSaltNoiseGen(0.005), scale=0.5)

    trans = v2.Compose([v2.ToTensor()])  # , v2.RandomRotation(5)])
    trans_test = v2.Compose([v2.ToTensor()])
    dataset = CustomDataset('data/dataset/train', noise_gen, trans, device)
    dataset_val = CustomDataset('data/dataset/val', noise_gen, trans, device)
    dataset_test = CustomDataset('data/dataset/test', noise_gen, trans_test, device)
    # data set loader
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_lodaer = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=True, num_workers=0)
    test_lodaer = DataLoader(dataset=dataset_test, batch_size=batch_size, num_workers=0)
    # model
    cfg = UNetConfig([1, 32, 64, 128], 0.)  # Best [1, 32, 64, 128], 0.
    model = UNet(cfg)
    # model = AutoEncoder(input_channels=1, base_channels=24, layers=3)
    model.to(device)
    # model.compile()
    # loss function
    criterion = nn.L1Loss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00003)
    # number of params
    # train
    epochs = 4
    for epoch in range(epochs):
        train_loss, losses = train_iter(
            model, train_loader, criterion, optimizer, debug=True)
        print(f'Epoch {epoch+1}/{epochs} train loss: {train_loss}')
        val_loss = validate_iter(model, val_lodaer, criterion)
        print(f'Epoch {epoch+1}/{epochs} val loss: {val_loss}')
    current_datetime = datetime.datetime.now()
    filename = current_datetime.strftime("file_%Y%m%d_%H%M%S.txt")
    # save model
    torch.save(model.state_dict(), f'models/{filename}.pth')

    # load example.png
