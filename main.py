
from data_loader import CustomDataset
from noise_gen import GaussianNoiseGen, CompositeNoise, PepperSaltNoiseGen, PoissonNoiseGen
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import tqdm
import matplotlib.pyplot as plt

def model_autoencoder():
    
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(128, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Dropout2d(0.2),
        nn.Conv2d(256, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(128, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(64, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 1, 3, padding=1),
        nn.Sigmoid()
    )
    return model

def train_iter(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0.0
    for i, (img, noisy_img, noise) in enumerate(tqdm.tqdm(train_loader)):
        optimizer.zero_grad()
        output = model(noisy_img)
        loss = criterion(output, img)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise_gen = CompositeNoise()
    noise_gen.add(GaussianNoiseGen(0, 0.2), 0, 0.3, 1, 1, -100, 255)
    noise_gen.add(PoissonNoiseGen(2), 0, 0.1, 1, 0.3, -100, 255)
    noise_gen.add(PepperSaltNoiseGen(0.001), 0, 0.5, 1, 1, 0, 255)

    
    trans = transforms.Compose([transforms.ToTensor()])
    dataset = CustomDataset('data\\dataset\\train', noise_gen, trans, device)
    dataset_val = CustomDataset('data\\dataset\\val', noise_gen, trans, device)
    # data set loader
    train_loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2)
    val_lodaer = DataLoader(dataset=dataset_val,batch_size=32,shuffle=False,num_workers=2)

    # model
    model = model_autoencoder()
    model.to(device)
    # loss function
    criterion = nn.MSELoss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # number of params
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_params}')
    # train
    epochs = 1
    for epoch in range(epochs):
        train_loss = train_iter(model, train_loader, criterion, optimizer)
        print(f'Epoch {epoch+1}/{epochs} train loss: {train_loss}')
        val_loss = validate_iter(model, val_lodaer, criterion)
        print(f'Epoch {epoch+1}/{epochs} val loss: {val_loss}')

    # save model
    torch.save(model.state_dict(), 'model.pth')

    
    # visualize 8 examples
    model.eval()
    with torch.no_grad():
        for i, (img, noisy_img, noise) in enumerate(tqdm.tqdm(val_lodaer)):
            
            output = model(noisy_img)
            img = img.cpu().numpy()
            noisy_img = noisy_img.cpu().numpy()
            output = output.cpu().numpy()
            noise = noise.cpu().numpy()
            for j in range(8):
                fig, axs = plt.subplots(1, 4, figsize=(20, 20))
                axs[0].imshow(img[j, 0, :, :], cmap='gray')
                axs[0].set_title('Original')
                axs[1].imshow(noisy_img[j, 0, :, :], cmap='gray')
                axs[1].set_title('Noisy')
                axs[2].imshow(output[j, 0, :, :], cmap='gray')
                axs[2].set_title('Denoised')
                axs[3].imshow(noise[j, :, :], cmap='gray')
                axs[3].set_title('Noise')
                plt.show()
            
            break
