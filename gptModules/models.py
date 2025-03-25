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

        #Transformer Blocks(Decoder Blocks)
        self.layers = nn.ModuleList([
            layers.ALiBiTransformerBlock(d_model, n_heads, d_ff, max_seq_length, dropout) for _ in range(n_layers)
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
    

class LLaMA(Module):
    '''
    GQA(4 head group)
    RoPE(theta=500_000)
    RMS Norm
    '''
    def __init__(self, vocab_size: int, n_layers: int, n_heads: int, 
                 d_model: int, d_ff: int, n_groups: int, max_seq_length: int, base: int, dropout: float=0.1):
        super(LLaMA, self).__init__()
        self.token_embedding = Embedding(vocab_size, d_model)
        self.embedding = layers.EmbeddingsWithoutPosition(d_model)

        self.dropout = Dropout(dropout)

        #Transformer Blocks(Decoder Blocks)
        self.layers = nn.ModuleList([
            layers.GroupedQueryTransformerBlock(d_model, n_heads, d_ff, n_groups, max_seq_length, dropout, base) for _ in range(n_layers)
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
    

class DeepSeek(Module):
    '''
    MoE
    자체적인 RoPE
    Multi Head Latent Attention
    RMS Norm
    '''
    def __init__(self, vocab_size: int, n_layers: int, n_heads: int, 
                 d_model: int, d_ff: int, max_seq_length: int, 
                 n_shared: int, n_expert: int, top_k: int, d_kv_comp: int, d_rope: int, rope_base: int, dropout: float=0.1):
        super(DeepSeek, self).__init__()

        self.token_embedding = Embedding(vocab_size, d_model)
        self.embedding = layers.EmbeddingsWithoutPosition(d_model)

        self.dropout = Dropout(dropout)

        #Transformer Blocks(Decoder Blocks)
        transformer_args = {
            'd_model': d_model,
            'n_heads': n_heads,
            'd_ff': d_ff,
            'max_seq_length': max_seq_length,
            'n_shared': n_shared,
            'n_expert': n_expert,
            'top_k': top_k,
            'd_kv_comp': d_kv_comp,
            'd_rope': d_rope,
            'rope_base': rope_base,
        }
        self.layers = nn.ModuleList([
            layers.DeepseekTransformerBlock(**transformer_args) for _ in range(n_layers)
        ])

        self.out_linear = Linear(d_model, vocab_size)        


    def forward(self, x: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        '''
        return out, aux_loss
        '''
        out = self.token_embedding(x)
        out = self.embedding(out)
        out = self.dropout(out)

        #Transformer Decoder Block
        total_aux_loss = 0.0
        for layer in self.layers:
            out = layer(out, attention_mask)
            total_aux_loss += layer.ffn.aux_loss


        out = self.out_linear(out)
        return out, total_aux_loss