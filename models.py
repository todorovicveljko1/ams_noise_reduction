from dataclasses import dataclass, field
import torch.nn as nn


@dataclass
class UNetConfig:
    # number of chanels on first block, every next block is doubled
    channels: list[int] = field(default_factory=lambda: [1, 16, 32, 64])
    # drop out applayed only to dropout
    res_dropout: float = field(default_factory=lambda: 0.2)
    kernel_size: int = field(default_factory=lambda: 3)

    @property
    def n_blocks(self):
        # number of blocks is one less then number of channels
        return len(self.channels) - 1


@dataclass
class BlockConfig:
    channel: tuple[int, int]  # number of in, out chanels in block
    rescale: bool  # use maxpool or upsample
    kernel_size: int


class DownSampleBlock(nn.Module):
    def __init__(self, cfg: BlockConfig):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cfg.channel[0], cfg.channel[1],
                      cfg.kernel_size, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.channel[1], cfg.channel[1],
                      cfg.kernel_size, padding="same"),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2) if cfg.rescale else None

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        return self.block(x)


class UpSampleBlock(nn.Module):
    def __init__(self, cfg: BlockConfig):
        super().__init__()
        self.block = nn.Sequential(
            # nn.BatchNorm2d(cfg.channel[0]),
            nn.Conv2d(cfg.channel[0], cfg.channel[0],
                      cfg.kernel_size, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.channel[0], cfg.channel[1],
                      cfg.kernel_size, padding="same"),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2) if cfg.rescale else None

    def forward(self, x):
        x = self.block(x)
        if self.upsample is not None:
            return self.upsample(x)
        return x


class UNet(nn.Module):
    def __init__(self, cfg: UNetConfig):
        super(UNet, self).__init__()
        self.cfg = cfg
        self.down_blocks = nn.ModuleList([])
        chs = cfg.channels
        for i in range(len(chs[:-1])):  # [1, 16, 32; 64]
            # i!=0 first block will have false for rescale
            self.down_blocks.append(DownSampleBlock(
                BlockConfig((chs[i], chs[i+1]), i != 0, cfg.kernel_size)))

        self.up_blocks = nn.ModuleList([])
        chs = cfg.channels[::-1]
        for i in range(len(chs[:-1])):  # in reverse order [64, 32, 16; 1]
            self.up_blocks.append(UpSampleBlock(BlockConfig(
                (chs[i], chs[i+1]), i != cfg.n_blocks - 1, cfg.kernel_size)))

        self.res_dropout = nn.Dropout2d(cfg.res_dropout)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ress = []
        for i, down in enumerate(self.down_blocks):
            x = down(x)
            if i < self.cfg.n_blocks - 1:
                ress.append(self.res_dropout(x))
        for i, up in enumerate(self.up_blocks):
            if i != 0:  # < self.cfg.n_blocks - 1:
                x = x + ress[-i]
            x = up(x)

        return x


class AutoEncoder(nn.Module):
    def __init__(self, input_channels=1, base_channels=16, layers=3):
        super(AutoEncoder, self).__init__()

        self.encode = self._make_encoder_layers(
            input_channels, base_channels, layers)
        self.decode = self._make_decoder_layers(
            base_channels * (2**(layers-1)), base_channels, layers, input_channels)

    def _make_encoder_layers(self, in_channels, base_channels, num_layers):
        layers = []
        channels = in_channels
        for i in range(num_layers):
            out_channels = base_channels * (2**i)
            layers.append(nn.Conv2d(channels, out_channels,
                          kernel_size=3, padding="same"))
            layers.append(nn.ReLU(True))
            layers.append(nn.MaxPool2d(2))
            channels = out_channels
        return nn.Sequential(*layers)

    def _make_decoder_layers(self, in_channels, base_channels, num_layers, final_out_channels):
        layers = []
        channels = in_channels
        for i in range(num_layers):
            out_channels = base_channels * (2**(num_layers-i-1))
            if i == 0:
                # Batch norm on bottleneck
                layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.Conv2d(channels, out_channels,
                          kernel_size=3, padding="same"))
            layers.append(nn.ReLU(True))
            layers.append(nn.Upsample(scale_factor=2))
            channels = out_channels
        layers.append(nn.Conv2d(base_channels, final_out_channels,
                      kernel_size=3, padding="same"))
        layers.append(nn.ReLU(True))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(self.encode(x))
