import torch.nn as nn

from torch import Tensor
from torch.nn import Module
from torch.nn import Linear
from torch.nn import Dropout
from torch.nn import Embedding


from . import layers



class GPT(Module):
    def __init__(self, vocab_size: int, n_layers: int, n_heads: int, 
                 d_model: int, d_ff: int, max_seq_length: int, dropout: float=0.1):
        super(GPT, self).__init__()

        self.token_embedding = Embedding(vocab_size, d_model)
        self.embedding = layers.Embeddings(max_seq_length, d_model)

        self.dropout = Dropout(dropout)

        #Transformer Blocks(Decoder Blocks)
        self.layers = nn.ModuleList([
            layers.TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        self.out_linear = Linear(d_model, vocab_size)



    def forward(self, x: Tensor, attention_mask: Tensor=None) -> Tensor:

        out = self.token_embedding(x)
        out = self.embedding(out)
        out = self.dropout(out)

        
        #Transformer Decoder Block
        for layer in self.layers:
            out = layer(out, attention_mask)
        
        out = self.out_linear(out)
        return out


class GPT2(Module):
    '''
    Add & Norm Layer를 앞으로 이동 시킴
    '''
    def __init__(self, vocab_size: int, n_layers: int, n_heads: int, 
                 d_model: int, d_ff: int, max_seq_length: int, dropout: float=0.1):
        super(GPT2, self).__init__()

        self.token_embedding = Embedding(vocab_size, d_model)
        self.embedding = layers.Embeddings(max_seq_length, d_model)

        self.dropout = Dropout(dropout)

        #Transformer Blocks(Decoder Blocks)
        self.layers = nn.ModuleList([
            layers.PreNormTransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        self.out_linear = Linear(d_model, vocab_size)



    def forward(self, x: Tensor, attention_mask: Tensor=None) -> Tensor:

        out = self.token_embedding(x)
        out = self.embedding(out)
        out = self.dropout(out)

        
        #Transformer Decoder Block
        for layer in self.layers:
            out = layer(out, attention_mask)
        
        out = self.out_linear(out)
        return out
    

class ALiBiGPT(Module):
    '''
    ALiBi Positional Embedding을 사용
    Pre Norm Layer 사용
    '''

    def __init__(self, vocab_size: int, n_layers: int, n_heads: int, 
                 d_model: int, d_ff: int, max_seq_length: int, dropout: float=0.1):
        super(ALiBiGPT, self).__init__()

        self.token_embedding = Embedding(vocab_size, d_model)
        self.embedding = layers.EmbeddingsWithoutPosition(d_model)

        self.dropout = Dropout(dropout)


    def forward(self, x: Tensor, attention_mask: Tensor=None) -> Tensor:
        pass