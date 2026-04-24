import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
from torchvision import models


class TeamCluster(nn.Module):

    def __init__(self) -> None:
        super(TeamCluster, self).__init__()

    def forward(self, x):
        pass
