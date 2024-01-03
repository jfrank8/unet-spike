import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.nn.functional import sigmoid
from torchvision.transforms import v2

class g_net(nn.Module):
    def __init__(self, ann=True):
        super().__init__()
        #first block
        bias = True
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 2, kernel_size=1, padding=0, bias=bias)
        if ann:
            self.lif1 = nn.ReLU()
        else:
            self.lif1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.conv2 = nn.Conv2d(in_channels = 2, out_channels = 12, kernel_size=3, padding=0, bias=bias)
        if ann:
            self.lif2 = nn.ReLU()
        else:
            self.lif2 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.conv3 = nn.Conv2d(in_channels = 12, out_channels = 12, kernel_size=3, padding=0, bias=bias)
        if ann:
            self.lif3 = nn.ReLU()
        else:
            self.lif3 = neuron.IFNode(surrogate_function=surrogate.ATan())

        #downsample
        self.conv4 = nn.Conv2d(in_channels = 12,out_channels = 24, kernel_size=3, stride=2, padding=0, bias=bias)
        #if ann:
        #    self.downlif = nn.ReLU()
        #else:
        #    self.downlif = neuron.IFNode(surrogate_function=surrogate.ATan())
        #upsample
        self.conv5 = nn.ConvTranspose2d(in_channels = 24,out_channels = 12, kernel_size=2, stride=2, bias=bias)
        #if ann:
        #    self.uplif = nn.ReLU()
        #else:
        #    self.uplif = neuron.IFNode(surrogate_function=surrogate.ATan())

        self.conv6 = nn.Conv2d(in_channels = 24,out_channels = 12, kernel_size = 3, padding=0, bias=bias)
        if ann:
            self.lif4 = nn.ReLU()
        else:
            self.lif4 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.conv7 = nn.Conv2d(in_channels = 12,out_channels = 12, kernel_size = 3, padding=0, bias=bias)
        if ann:
            self.lif5 = nn.ReLU()
        else:
            self.lif5 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.conv8 = nn.Conv2d(in_channels=12, out_channels= 2, kernel_size=1, bias=bias)
        if ann:
            self.lif6 = nn.ReLU()
        else:
            self.lif6 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.final = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.lif1(x)
        x = self.conv2(x)
        x = self.lif2(x)
        x = self.conv3(x)
        x = self.lif3(x)
        #crop
        y = v2.CenterCrop(size=54)(x)
        #downsample
        x = self.conv4(x)
        #x = self.downlif(x)
        x = v2.CenterCrop(size=27 )(x)
        z = self.conv5(x)
        #z = self.uplif(z)

        #print(y.size())
        #print(z.size())
        x = torch.concatenate([y,z], axis=1)
        #print(x.size())
        x = self.conv6(x)
        x = self.lif4(x)
        x = self.conv7(x)
        x = self.lif5(x)
        x = self.conv8(x)
        x = self.lif6(x)
        #breakpoint()
        x = self.final(x)

        return x


if __name__ == "__main__":
    # A full forward pass
    im = torch.randn(1, 1, 64, 64)
    model = g_net()
    x = model(im)
    # print(x.shape)
    del model
    del x
    # print(x.shape)
