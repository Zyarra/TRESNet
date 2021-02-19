import torch
import torch.nn as nn


class Residual(nn.Module):
    """
    In a network with residual blocks, each layer feeds into the next layer and directly
    into the layers about 2–3 hops away
    """
    def __init__(self, fn):
        """
        fn: layer function to apply the residual layer on
        """
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNormalize(nn.Module):
    """Applies Layer Normalization over a mini-batch of inputs"""
    def __init__(self, dim, fn):
        """
        dim: input/output dimensions
        fn: layer function to apply the normalization on
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class LambdaLayer(nn.Module):
    """
    Lambda layer
    """
    def __init__(self, lambd):
        """
        lambd: lambda function to iterate the data over
        """
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class Conv2dSame(torch.nn.Module):
    """
    Creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.
    The only difference to the conv2d module is the padding='SAME'
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=torch.nn.ReflectionPad2d,
                 stride=1):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias, stride=stride)
        )

    def forward(self, x):
        return self.net(x)
