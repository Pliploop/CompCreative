### MLP class with Ada LayerNorm for conditioning, the conditioning can be done with only a text encoder or text and timesteps

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings
from diffusers.models.normalization import AdaLayerNormContinuous


class Layer(nn.Module):
    def __init__(self, in_channels, out_channels, conditioning = True, dropout = 0.1, activation = 'silu', conditioning_embedding_dim = 512):
        super(Layer, self).__init__()
        
        self.conditioning = conditioning
        self.activation = getattr(F, activation)
        
        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = AdaLayerNormContinuous(out_channels, conditioning_embedding_dim) if self.conditioning else nn.LayerNorm(out_channels)
        self.layernorm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, conditioning = None):
        x = self.linear(x)
        x = self.norm(x, conditioning) if ((self.conditioning==True) and (conditioning is not None)) else self.layernorm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class MLP(nn.Module):
    def __init__(self,
                 in_channels = 512,
                 out_channels = 512,
                 layer_dims = [2048, 2048, 2048],
                 dropout = 0.1,
                 conditioning = [True, True, True],
                 final_conditioning = True,
                 activation = 'silu',
                 conditioning_embedding_dim = 512,
                 **kwargs
                 ):
    
        super(MLP, self).__init__()
        
        self.conditioning = conditioning
        self.final_conditioning = final_conditioning
        
        self.activation = activation
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.layers = nn.ModuleList()
            
        self.conditioning_embedder = CombinedTimestepTextProjEmbeddings(embedding_dim = conditioning_embedding_dim, pooled_projection_dim=conditioning_embedding_dim)
        
        for i, (dim, cond) in enumerate(zip(layer_dims, conditioning)):
            layer = Layer(in_channels if i == 0 else layer_dims[i-1], dim, cond, dropout, activation, conditioning_embedding_dim)
            self.layers.append(layer)
            
        self.final_layer = nn.Linear(layer_dims[-1], out_channels)
        self.final_norm = AdaLayerNormContinuous(out_channels, conditioning_embedding_dim) if self.final_conditioning else nn.LayerNorm(out_channels)
        
    def forward(self, x, conditioning = None, timesteps = None):
        
        squeeze_ = x.dim()==2
        if squeeze_: x = x.unsqueeze(1)
        
        if timesteps is not None and conditioning is not None:
            conditioning = self.conditioning_embedder(timestep = timesteps, pooled_projection = conditioning)
        
        for layer in self.layers:
            x = layer(x, conditioning)
        
        x = self.final_layer(x)
        x = self.final_norm(x, conditioning) if self.final_conditioning else self.final_norm(x)
        
        if squeeze_: x = x.squeeze(1)
        
        return x
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)