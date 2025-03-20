
import torch
from torch import nn

from gptModules import layers

from torch import Tensor
from torch.nn import Module
from torch.nn import Linear
from torch.nn import Dropout


class GPT(Module):
    def __init__(self, vocab_size: int, n_layers: int, n_heads: int, 
                 d_model: int, d_ff: int, max_seq_length: int, dropout: float=0.1):
        super(GPT, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
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


if __name__ == '__main__':
    
    x = torch.randint(0, 15, (1, 3))

    model = GPT(
        vocab_size=15,
        n_layers=3,
        n_heads=2,
        d_model=16,
        d_ff=64,
        max_seq_length=25,
    )

    out = model(x)

    print(out, out.shape)