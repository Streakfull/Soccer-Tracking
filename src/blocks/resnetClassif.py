import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
from torchvision import models


class PlayerClassifier(nn.Module):

    def __init__(self, num_classes=7) -> None:
        super(PlayerClassifier, self).__init__()
        self.resnet = self.load_resnet()
        # self.bn_resnet = nn.BatchNorm1d(num_features=2048)
        self.linear = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=256, out_features=num_classes)


        )

    def load_resnet(self):
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])
        # for param in feature_extractor.parameters():
        #     param.requires_grad = False
        resnet_features = feature_extractor.to("cuda:0")
        # resnet_features.eval()
       # resnet_features.to("cuda:0")
        return resnet_features

    def forward(self, x):
        x = self.resnet(x)
        x = x.flatten(start_dim=1)
        # x = self.bn_resnet(x)
        x = self.linear(x)
        return x
