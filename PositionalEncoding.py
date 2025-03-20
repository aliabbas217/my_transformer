import torch
import torch.nn as nn
import math
from InputEmbeddings import InputEmbeddings

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        # Create a matrix of zeros of dimensions max_seq_length by d_model
        pe = torch.zeros((max_seq_length, d_model)) # just try to visualize things
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        # Perform the sine and cosine calculations
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) #as mentioned in paper, learnable PE and fixed PE performed almost similar, so making it fixed
        # buffer will make part of the model's state_dict but these parameters will not be learnable
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    # here I'm returning the input + pos_encoding, 
    # alternatingly we can also return only pos_encodings and add them explicitly with input