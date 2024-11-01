
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class myGPTConfig:
    train_data_split_ratio:float = 0.8
    batch_size: int = 64
    vocab_size: int = 20000
    time_sequence_length: int = 128
    embedding_depth: int = 128
    layer_numbers: int = 4
    head_numbers: int = 4
    dropout: float = 0.1
    bias: bool = True

#position embedding with sin and cos
class PositionEmbedding(nn.Module):
    def __init__(self,config:myGPTConfig):
        super().__init__()
        d_model = config.embedding_depth
        seq_len = config.time_sequence_length
        position_embedding = torch.zeros(seq_len,d_model)
        #print(position_embedding)
        #position = torch.arrange(0,seq_len,dtype=torch.float)
        #pos = torch.arrange(0,seq_len,dtype=torch.float)
        #print(pos) #arange
        #print(pos.unsqueeze(0)) #Tensor[1,16]
        #print(pos.unsqueeze(1)) #Tensor[16,1]
        pos = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        #print(pos.shape)
        parameter_divided = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))
        #print(parameter_divided) #arrange
        position_embedding[:,0::2] = torch.sin(pos * parameter_divided) #boardcast mechanism
        position_embedding[:,1::2] = torch.cos(pos * parameter_divided) #pos[16,1] parameter_divided:arrange
        #print(position_embedding.shape)
        #print(position_embedding) #[seq_len,embedding_depth]
        #print(position_embedding.unsqueeze(0).shape) #[1,seq_len,embedding_depth]
        position_embedding = position_embedding.unsqueeze(0) #[1,seq_len,embedding_depth]
        self.register_buffer('sinusoid_pe',position_embedding)
    
    def forward(self,x):
        return self.sinusoid_pe[:,:x.shape[1],:]
        
class SelfAttention(nn.Module):
    def __init__(self, config:myGPTConfig):
        super().__init__()
        self.attention = nn.Linear(config.embedding_depth, 3 * config.embedding_depth, bias=config.bias)
        self.proj = nn.Linear(config.embedding_depth, config.embedding_depth, bias=config.bias)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.head_numbers = config.head_numbers
        self.embedding_depth = config.embedding_depth
        self.dropout = config.dropout
        self.register_buffer("mask",torch.tril(torch.ones(config.time_sequence_length, config.time_sequence_length))
                                                .view(1, 1, config.time_sequence_length, config.time_sequence_length))
        #print(torch.tril(torch.ones(config.time_sequence_length, config.time_sequence_length))
        #                        .view(1, 1, config.time_sequence_length, config.time_sequence_length))

    def forward(self,x):
        B,T,C = x.size() #B:batch size T:sequence length C:embedding dimensionality
        q,k,v = self.attention(x).split(self.embedding_depth,dim=2)
        q = q.view(B, T, self.head_numbers, C // self.head_numbers).transpose(1, 2)
        k = k.view(B, T, self.head_numbers, C // self.head_numbers).transpose(1, 2)
        v = v.view(B, T, self.head_numbers, C // self.head_numbers).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)           
        att = self.attention_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(y)

#GPT framework:forward and generate
class myGPT(nn.Module):
    def __init__(self,config:myGPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size,config.embedding_depth)
        self.position_embedding_table = PositionEmbedding(config)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.layer_numbers)])
        self.layer_norm = nn.LayerNorm(config.embedding_depth, bias=config.bias)
        self.final_linear = nn.Linear(config.embedding_depth, config.vocab_size, bias=False)

        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, features, targets=None):
        tok_emb = self.token_embedding_table(features) 
        pos_emb = self.position_embedding_table(tok_emb)  
        x = tok_emb + pos_emb   
        for block in self.blocks:
            x = block(x)

        x = self.layer_norm(x)
        logits = self.final_linear(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, seq, max_new_tokens):
        for _ in range(max_new_tokens):
            seq = seq[:, -self.config.time_sequence_length:]
            logits, loss = self.forward(seq)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            seq_next = torch.multinomial(probs, num_samples=1)
            seq = torch.cat((seq, seq_next), dim=1)
        return seq




class FeedForward(nn.Module):
    """ a two-layers mlp """
    def __init__(self, config:myGPTConfig):
        super().__init__()
        d_model = config.embedding_depth
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self,config:myGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embedding_depth,bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln2 = nn.LayerNorm(config.embedding_depth, bias=config.bias)
        self.ffn = FeedForward(config)


    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))                                              #
        return x


