import torch
from torch import nn as nn
from . import networks

class SSIToSIDepthModel():
    def __init__(self, device):
        # define networks
        input_nc = 5
        self.netG = networks.define_efficient_net(input_nc, outer_activation="sigmoid", backbone="efficientnet_b7")
        self.netG.to(device)

        self.device = device
        
    def set_input(self, outer, inner, rgb):
        inner = torch.from_numpy(inner.copy()).unsqueeze(0).unsqueeze(0)
        outer = torch.from_numpy(outer.copy()).unsqueeze(0).unsqueeze(0)
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0)

        self.real_A = torch.cat((rgb, outer, inner), 1).to(self.device)


    def forward(self):
        self.fake_B = self.netG(self.real_A).unsqueeze(1)

class BMDMergeModel():
    def __init__(self, device):
        # define networks
        def norm_layer(x): return nn.Identity()
        self.netG = networks.UnetGenerator(input_nc=2,output_nc=1, num_downs=10, norm_layer=norm_layer, outer_activation="tanh")
        self.netG.to(device)

        self.device = device
        
    def set_input(self, outer, inner):
        inner = torch.from_numpy(inner.copy()).unsqueeze(0).unsqueeze(0)
        outer = torch.from_numpy(outer.copy()).unsqueeze(0).unsqueeze(0)

        inner = self.normalize(inner)
        outer = self.normalize(outer)

        self.real_A = torch.cat((outer, inner), 1).to(self.device)


    def forward(self):
        self.fake_B = self.netG(self.real_A)


    def normalize(self, input):

        return input * 2 - 1