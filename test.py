from model import myGPT,myGPTConfig
import numpy as np
import torch.nn as nn
import sentencepiece as spm
import sys
import torch
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from data_set import load_tokenizer
from torch.nn import functional as F

class testConfig:
    train_data_split_ratio:float = 0.8
    batch_size: int = 4
    vocab_size: int = 20000
    time_sequence_length: int = 8
    embedding_depth: int = 16
    layer_numbers: int = 4
    head_numbers: int = 4
    dropout: float = 0.0
    bias: bool = True

train_data = np.memmap('train.dat', dtype=np.int32, mode='r')
target_data = np.memmap('target.dat', dtype=np.int32, mode='r')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LayerNorm(nn.Module):
    def __init__(self,ndim,bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self,input):
        return F.layer_norm(input,self.weight.shape,self.weight,self.bias,1e-5)

def get_batch(split, config:testConfig):
    mydata = train_data if split == 'train' else target_data
    ix = torch.randint(len(mydata) - config.time_sequence_length, (config.batch_size,))
    x = torch.stack([torch.from_numpy((mydata[i:i + config.time_sequence_length]).astype(np.int32)) for i in ix])
    y = torch.stack([torch.from_numpy((mydata[i + 1:i + 1 + config.time_sequence_length]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def main1():
    config = testConfig()
    x,y = get_batch('train',config)
    token_embedding_table = nn.Embedding(config.vocab_size, config.embedding_depth)
    xEmbed = token_embedding_table(x)
    print(x[0])
    print(xEmbed[0])
    my_layer_norm = LayerNorm(config.embedding_depth,bias = config.bias)
    xLayerMy = my_layer_norm(xEmbed)
    print(xLayerMy[0])
    nn_layer_norm = nn.LayerNorm(config.embedding_depth,bias = config.bias)
    xLayernn = nn_layer_norm(xEmbed)
    print(xLayernn[0])

if __name__ == '__main__':
    main1()