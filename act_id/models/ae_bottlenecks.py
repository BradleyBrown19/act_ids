import pdb
from typing import List
import torch
import torch.nn as nn
import ast

class AEBottleneck(nn.Module):
    def __init__(
        self,
        layer_name: str,
        lam: float,
        dims: List,
        activation = nn.ReLU,
        norm_loss = False,
    ):
        super().__init__()

        assert dims[0] == dims[-1]
        assert len(dims) > 2

        self.layer_name = layer_name
        self.lam = lam

        layers = [nn.Linear(dims[0], dims[1]), activation()]
        last_dim = dims[1]

        for dim in dims[2:-1]:
            layers.extend([nn.Linear(last_dim, dim), activation()])
            last_dim = dim
        
        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.layers = nn.Sequential(*layers)

        self.norm_loss = norm_loss
    
    def forward(self, xb):
        return self.layers(xb)
    
    def loss(self, xb):
        if self.norm_loss:
            return ((((self.layers(xb)-xb)**2).sum()) / (xb**2).sum() )*self.lam
        else:
            return ((((self.layers(xb)-xb)**2).mean()) / (xb**2).mean() )*self.lam

class ClassAEBottleneck(nn.Module):
    def __init__(
        self,
        layer_name: str,
        lam: float,
        encoder_dims: List,
        decoder_dims: List,
        class_dims: List,
        activation = nn.ReLU,
        norm_loss = False
    ):
        super().__init__()

        self.layer_name = layer_name
        self.lam = lam

        # Build Encoder
        last_dim = encoder_dims[0]
        encoder = []
        for dim in encoder_dims[1:]:
            encoder.extend([nn.Linear(last_dim, dim), activation()])
            last_dim = dim 

        self.encoder = nn.Sequential(*encoder)

        # Build class layers
        class_layers = []
        for class_dim in class_dims:
            class_layer = []
            class_last_dim = last_dim
            for idx,dim in enumerate(class_dim + [decoder_dims[0]]):
                class_layer.append(nn.Linear(class_last_dim, dim))
                class_layer.append(activation())
                class_last_dim = dim

            class_layer = nn.Sequential(*class_layer)
            class_layers.append(class_layer)
        self.class_layers = nn.ModuleList(class_layers)
       
        # Build Decoder
        last_dim = decoder_dims[0]
        decoder = []
        for idx,dim in enumerate(decoder_dims[1:]):
            decoder.append(nn.Linear(last_dim, dim))
            if idx != len(decoder_dims) - 2:
                decoder.append(activation())
            last_dim = dim 
        self.decoder = nn.Sequential(*decoder)

        self.norm_loss = norm_loss
    
    def class_layers_forward(self, inputs, labels):
        outs = []
        for input,label in zip(inputs,labels):
            outs.append(self.class_layers[label.item()](input))
        return torch.stack(outs)
    
    def layers(self, inputs, labels):
        inputs = self.encoder(inputs)
        inputs = self.class_layers_forward(inputs, labels)
        inputs = self.decoder(inputs)
        return inputs
    
    def loss(self, batch):
        _,labels = batch
        inputs = self.inputs
        if self.norm_loss:
            return ((((self.layers(inputs,labels)-inputs)**2).sum()) / (inputs**2).sum())*self.lam
        else:
            return ((((self.layers(inputs,labels)-inputs)**2).mean()) / (inputs**2).mean())*self.lam
