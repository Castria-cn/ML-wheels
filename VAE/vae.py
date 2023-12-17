import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, *hiddens):
        super().__init__()
        self.layers = []
        self.activate = nn.ReLU()
        for i in range(len(hiddens) - 1):
            self.layers.append(nn.Linear(hiddens[i], hiddens[i + 1]))
            self.add_module(f'linear{i}', self.layers[-1])
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.activate(x)
        return x

class Encoder(nn.Module):
    """
    VAE Encoder:
    Given x, output mu and sigma of x.
    z = mu(x) * epsilon + sigma(x), epsilon ~ N(0, I)
    """
    def __init__(self, *hiddens):
        super().__init__()
        self.mlp = MLP(*hiddens)
        self.add_module('mlp', self.mlp)
    
    def forward(self, x):
        return self.mlp(x)

class Decoder(nn.Module):
    """
    VAE Decoder:
    Given z, output mu and sigma of z.(which are used for decode x' from z)
    x' ~ p_theta(x|z), p_theta(x|z) = N(x; mu(z), sigma(z)^2(I))
    reparameterization: x' = mu(z) + sigma(z) * epsilon
    """
    def __init__(self, *hiddens):
        super().__init__()
        self.mlp = MLP(*hiddens)
        self.add_module('mlp', self.mlp)
    
    def forward(self, x):
        return self.mlp(x)