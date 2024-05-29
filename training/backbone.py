# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
We borrow the positional encoding from Detr and modify the ShortChunkCNN_Res as backbone base 
"""

from torch.functional import Tensor
import torch.nn.functional as F
from torch import nn

from position_encoding import build_position_encoding
import torchaudio
from modules import Res_2d

class ShortChunkCNN_Res(nn.Module):
    def __init__(self,
                n_channels=128,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=128):
        super(ShortChunkCNN_Res, self).__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # CNN
        self.layer1 = Res_2d(1, n_channels, stride=2)
        self.layer2 = Res_2d(n_channels, n_channels, stride=2)
        self.layer3 = Res_2d(n_channels, n_channels*2, stride=2)
        self.layer4 = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.layer5 = Res_2d(n_channels*2, n_channels*4, stride=2)

        # Dense
        self.dense1 = nn.Conv2d(n_channels*4, n_channels*4, 1)
        self.bn = nn.BatchNorm2d(n_channels*4)
        self.dense2 = nn.Conv2d(n_channels*4, n_channels*4, 1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Spectrogram
        x = self.spec(x)
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)

        return x

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, input: Tensor):
        xs = self[0](input)
        pos = self[1](xs).to(xs.dtype)
        return xs, pos


def build_backbone(hidden_dim, n_channels=128):
    position_embedding = build_position_encoding(hidden_dim, maxH=4, maxW=8)
    backbone = ShortChunkCNN_Res(n_channels=n_channels)
    model = Joiner(backbone, position_embedding)
    model.num_channels = n_channels*4
    return model
