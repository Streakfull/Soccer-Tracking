import torch
import torch.nn as nn
from einops import rearrange
import numpy as np


class SiameseNetworkPnet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # elf.pnet = PNet(100)

        self.backbone = nn.Sequential(
            nn.Linear(in_features=34, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=256, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),


            nn.Linear(in_features=2048, out_features=4096),
            # nn.BatchNorm1d(4096),
        )

       # self.up_project = nn.Linear(in_features=2048, out_features=4096)

        self.comb = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=128,
                      kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=128,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=1,
                      kernel_size=3, padding=1),


        )

        self.cls = nn.Linear(4096, 1)

    def forward(self, x):
        x1 = x["x1k"]
        x2 = x["x2k"]
        x1 = self.backbone(x1)
        x2 = self.backbone(x2)
        x1c = x1.unsqueeze(1)
        x2c = x2.unsqueeze(1)
        fc = torch.cat((x1c, x2c), dim=1)

        last_dim = fc.shape[-1]
        conv_dim = int(np.sqrt(last_dim))
        fc = rearrange(fc, 'bs ch (w h) -> bs ch w h', w=conv_dim, h=conv_dim)
        fc = self.comb(fc)
        fc = fc.flatten(start_dim=1)
        fc = self.cls(fc)
        return x1, x2, fc

    def pc_features(self, pc):
        return self.backbone(pc)
