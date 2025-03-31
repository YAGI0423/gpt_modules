import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F


from torch.nn import Module
from torch.nn import Linear
from torch.nn import Dropout
from torch.nn import Embedding
from torch.nn import LayerNorm
from torch import Size, Tensor

import numbers
from typing import Union, List



#<Embedding Layer>=====================================
class Embeddings(Module):
    def __init__(self, max_seq_length: int, d_model: int):
        super(Embeddings, self).__init__()
        
        self.segment_embedding = Embedding(2, d_model) #seg(0, 1)
        self.position_embedding = Embedding(max_seq_length, d_model)
    

    def forward(self, x: Tensor, segment_input_ids: Tensor=None) -> Tensor:
        device = x.device

        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        
        if segment_input_ids is not None:
            x = x + self.segment_embedding(segment_input_ids)
        
        return x + pos_emb


class EmbeddingsWithoutPosition(Module):
    def __init__(self, d_model: int):
        super(EmbeddingsWithoutPosition, self).__init__()

        self.segment_embedding = Embedding(2, d_model) #seg(0, 1)

    
    def forward(self, x: Tensor, segment_input_ids: Tensor=None) -> Tensor:
        if segment_input_ids is None:
            return x
        
        return x + self.segment_embedding(segment_input_ids)


class RotaryPositionalEmbeddings(Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: int=10_000):
        super(RotaryPositionalEmbeddings, self).__init__()
        
        #Meta Data
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        self.h_half = head_dim // 2


        self.cache: Tensor
        self.rope_init()


    def register_parameter(self):
        self.rope_init()


    def rope_init(self) -> None:
        arange_ = torch.arange(0, self.head_dim, 2)[: self.h_half].float()
        theta = 1.0 / (self.base ** (arange_ / self.head_dim))

        seq_idx = torch.arange(self.max_seq_len, dtype=theta.dtype)

        #(max_seq_len x h_half(head_dim // 2))
        idx_theta = torch.einsum('i, j -> ij', seq_idx, theta).float()

        #(max_seq_len x h_half x 2)
        cache = torch.stack((torch.cos(idx_theta), torch.sin(idx_theta)), dim=-1)
        self.register_buffer('cache', cache, persistent=False)


    def forward(self, x: Tensor) -> Tensor:
        #x (batch x seq x n_heads x head_dim)
        batch, seq, n_heads, _ = x.shape


        #(seq x h_half(h_model // 2) x 2)
        rope_cache = self.cache[:seq] #seq 만큼만 cache에서 indexing


        #(batch x seq x n_heads x h_half(head_dim // 2) x 2)
        x_shaped = x.float().view(batch, seq, n_heads, self.h_half, 2)


        #(batch(1) x seq x 1 x h_half x 2)
        rope_cache = rope_cache.view(1, seq, 1, self.h_half, 2)


        #(batch x seq x n_heads x h_half x 2)
        x_out = (
            x_shaped[..., 0] * rope_cache[..., 0] - x_shaped[..., 1] * rope_cache[..., 1],
            x_shaped[..., 1] * rope_cache[..., 0] + x_shaped[..., 0] * rope_cache[..., 1],
        )
        x_out = torch.stack(x_out, dim=-1)

        #(batch x seq x n_heads x h_half)
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


class ALiBiEmbeddings(Module):
    def __init__(self, n_heads: int, max_seq_length: int):
        super(ALiBiEmbeddings, self).__init__()
        self.alibi_bias: Tensor
        self.register_buffer(
            'alibi_bias', 
            self.create_alibi_bias(n_heads, max_seq_length), #Tensor
            persistent=False,
        )


    @staticmethod
    def create_alibi_bias(n_heads:int, max_seq_len: int) -> Tensor:
        get_scale = lambda idx: 2**(-8*(idx / n_heads))

        #Distance Matrix (max_seq_len x max_seq_len)
        distance = torch.arange(max_seq_len)
        distance = distance[:, None] - distance[None, :]
        distance = distance[None, :, :].clamp(min=0) #음수 값 제거


        #Head별 Scaling Factor (n_heads x 1 x 1)
        slopes = torch.tensor([get_scale(i) for i in range(n_heads)])
        slopes = slopes[:, None, None]

        #Calcuate Bias (n_heads x max_seq_len x max_seq_len)
        return -(distance * slopes)


    def forward(self, x: Tensor) -> Tensor:
        '''
        x = (K * R^T) #(batch x n_heads x seq x seq)

        return (batch x n_heads x q_seq x k_seq)
        '''
        seq = x.size(-1)

        alibi_bias = self.alibi_bias[:, :seq, :seq]
        return x + alibi_bias #(batch x n_heads x q_seq x k_seq)
#End===================================================




#<Multi Head Attention Layer>==========================
class MaskedMultiHeadAttention(Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super(MaskedMultiHeadAttention, self).__init__()

        #Meta Data
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads #각 head의 차원
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])) #sqrt(d_k)

        self.query = Linear(d_model, d_model, bias=False)
        self.key = Linear(d_model, d_model, bias=False)
        self.value = Linear(d_model, d_model, bias=False)

        self.dropout = Dropout(dropout)
        self.fc_out = Linear(d_model, d_model)

    
    @staticmethod
    def __masked_fill(tensor: Tensor, mask: Tensor, fill_value='-inf') -> Tensor:
        return tensor.masked_fill(mask==0, float(fill_value))
    

    @staticmethod
    def __get_causal_mask(size: int) -> Tensor:
        causal_mask = torch.ones(size, size)
        causal_mask = torch.tril(causal_mask)
        return causal_mask



    def scaledDotProductAttention(self, Q: Tensor, K: Tensor, 
                                  V: Tensor, attention_mask: Tensor=None) -> Tensor:
        
        device = Q.device

        #(batch x n_heads x seq x head_dim)
        seq_len = Q.size(-2)


        #(batch x n_heads x seq x seq)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(device)


        #Attention Mask
        if attention_mask is not None:
            scores = self.__masked_fill(scores, attention_mask[:, None, None, :])
        
        #Causal Mask
        causal_mask = self.__get_causal_mask(seq_len).to(device).view(1, 1, seq_len, seq_len)
        scores = self.__masked_fill(scores, causal_mask)


        attention_weights = F.softmax(scores, dim=-1) #Q(행)를 기준으로 softmax
        attention_weights = self.dropout(attention_weights)
        return torch.matmul(attention_weights, V)
    


    def forward(self, Q: Tensor, K: Tensor, V: Tensor, 
                attention_mask: Tensor=None) -> Tensor:

        batch, seq, _ = Q.shape

        #(batch x seq x d_model) -> (batch x seq x n_heads x head_dim)
        Q = self.query(Q).view(batch, seq, self.n_heads, self.head_dim)
        K = self.key(K).view(batch, seq, self.n_heads, self.head_dim)
        V = self.value(V).view(batch, seq, self.n_heads, self.head_dim)
        
        #(batch x seq x n_heads x head_dim) -> (batch x n_heads x seq x head_dim)
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        out = self.scaledDotProductAttention(
            Q=Q,
            K=K,
            V=V,
            attention_mask=attention_mask,
        )

        #(batch x n_heads x seq x head_dim) -> (batch x seq x d_model)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch, seq, self.d_model)
        return self.fc_out(out)


class ALiBiAttenion(Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_length: int, dropout: float):
        super(ALiBiAttenion, self).__init__()

        #Meta Data
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads #각 head의 차원
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])) #sqrt(d_k)

        self.alibi_emb = ALiBiEmbeddings(n_heads, max_seq_length)

        self.query = Linear(d_model, d_model, bias=False)
        self.key = Linear(d_model, d_model, bias=False)
        self.value = Linear(d_model, d_model, bias=False)

        self.dropout = Dropout(dropout)
        self.fc_out = Linear(d_model, d_model)

    
    @staticmethod
    def __masked_fill(tensor: Tensor, mask: Tensor, fill_value='-inf') -> Tensor:
        return tensor.masked_fill(mask==0, float(fill_value))
    

    @staticmethod
    def __get_causal_mask(size: int) -> Tensor:
        causal_mask = torch.ones(size, size)
        causal_mask = torch.tril(causal_mask)
        return causal_mask


    def scaledDotProductAttention(self, Q: Tensor, K: Tensor, 
                                  V: Tensor, attention_mask: Tensor=None) -> Tensor:
        '''
        Q, K, V ∈ (batch x seq x n_heads x head_dim)
        attention_mask ∈(batch x seq)

        out ∈ (batch x seq x d_model)
        '''
        device = Q.device

        seq_len = Q.size(1)

        #(batch x n_head x q_seq x k_seq)
        scores = torch.einsum('bnhd, bmhd -> bhnm', Q, K) / self.scale.to(device)


        #ALiBi Emb(batch x n_head x q_seq x k_seq)
        scores = self.alibi_emb(scores)
        

        #Attention Mask
        if attention_mask is not None:
            att_mask = attention_mask[:, None, None, :] #(batch x 1 x 1 x seq)
            scores = self.__masked_fill(scores, att_mask)

        
        #Causal Mask
        causal_mask = self.__get_causal_mask(seq_len).to(device).view(1, 1, seq_len, seq_len)
        scores = self.__masked_fill(scores, causal_mask)


        attention_weights = F.softmax(scores, dim=-1) #Q(행)를 기준으로 softmax
        attention_weights = self.dropout(attention_weights)

        #(batch x seq x n_heads x head_dim)
        return torch.einsum('bhnm, bmhd -> bnhd', attention_weights, V)


    def forward(self, Q: Tensor, K: Tensor, V: Tensor, 
                attention_mask: Tensor=None) -> Tensor:
        '''
        Q, K, V ∈ (batch x seq x d_model)
        attention_mask ∈(batch x seq)

        out ∈ (batch x seq x d_model)
        '''
        batch, seq, _ = Q.shape

        #(batch x seq x d_model) -> (batch x seq x n_heads x head_dim)
        Q = self.query(Q).view(batch, seq, self.n_heads, self.head_dim)
        K = self.key(K).view(batch, seq, self.n_heads, self.head_dim)
        V = self.value(V).view(batch, seq, self.n_heads, self.head_dim)

        out = self.scaledDotProductAttention(Q, K, V, attention_mask)
        
        #(batch x seq x n_heads x head_dim) -> (batch x seq x d_model)
        out = out.contiguous().view(batch, seq, self.d_model)
        return self.fc_out(out)


class RoPEAttenion(Module):
    '''
    RoPE를 적용한 Attention
    '''
    def __init__(self, d_model: int, n_heads: int, max_seq_length: int, dropout: float, base: int):
        super(RoPEAttenion, self).__init__()

        #Meta Data
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads #각 head의 차원
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])) #sqrt(d_k)

        #RoPE Embedding
        self.q_rot_emb = RotaryPositionalEmbeddings(self.head_dim, max_seq_length, base)
        self.k_rot_emb = RotaryPositionalEmbeddings(self.head_dim, max_seq_length, base)


        self.query = Linear(d_model, d_model, bias=False)
        self.key = Linear(d_model, d_model, bias=False)
        self.value = Linear(d_model, d_model, bias=False)

        self.dropout = Dropout(dropout)
        self.fc_out = Linear(d_model, d_model)

    
    @staticmethod
    def __masked_fill(tensor: Tensor, mask: Tensor, fill_value='-inf') -> Tensor:
        return tensor.masked_fill(mask==0, float(fill_value))
    

    @staticmethod
    def __get_causal_mask(size: int) -> Tensor:
        causal_mask = torch.ones(size, size)
        causal_mask = torch.tril(causal_mask)
        return causal_mask


    def scaledDotProductAttention(self, Q: Tensor, K: Tensor, 
                                  V: Tensor, attention_mask: Tensor=None) -> Tensor:
        '''
        Q, K, V ∈ (batch x seq x n_heads x head_dim)
        attention_mask ∈(batch x seq)

        out ∈ (batch x seq x d_model)
        '''
        device = Q.device

        seq_len = Q.size(1)

        #(batch x n_head x q_seq x k_seq)
        scores = torch.einsum('bnhd, bmhd -> bhnm', Q, K) / self.scale.to(device)  

        #Attention Mask
        if attention_mask is not None:
            att_mask = attention_mask[:, None, None, :] #(batch x 1 x 1 x seq)
            scores = self.__masked_fill(scores, att_mask)

        
        #Causal Mask
        causal_mask = self.__get_causal_mask(seq_len).to(device).view(1, 1, seq_len, seq_len)
        scores = self.__masked_fill(scores, causal_mask)


        attention_weights = F.softmax(scores, dim=-1) #Q(행)를 기준으로 softmax
        attention_weights = self.dropout(attention_weights)

        #(batch x seq x n_heads x head_dim)
        return torch.einsum('bhnm, bmhd -> bnhd', attention_weights, V)


    def forward(self, Q: Tensor, K: Tensor, V: Tensor, 
                attention_mask: Tensor=None) -> Tensor:
        '''
        Q, K, V ∈ (batch x seq x d_model)
        attention_mask ∈(batch x seq)

        out ∈ (batch x seq x d_model)
        '''
        batch, seq, _ = Q.shape

        #(batch x seq x d_model) -> (batch x seq x n_heads x head_dim)
        Q = self.query(Q).view(batch, seq, self.n_heads, self.head_dim)
        K = self.key(K).view(batch, seq, self.n_heads, self.head_dim)
        V = self.value(V).view(batch, seq, self.n_heads, self.head_dim)

        #RoPE
        Q = self.q_rot_emb(Q)
        K = self.k_rot_emb(K)

        out = self.scaledDotProductAttention(Q, K, V, attention_mask)
        
        #(batch x seq x n_heads x head_dim) -> (batch x seq x d_model)
        out = out.contiguous().view(batch, seq, self.d_model)
        return self.fc_out(out)


class GroupedQueryAttention(Module):
    '''
    RoPE 적용된 GQA
    '''
    def __init__(self, d_model: int, n_heads: int, n_groups: int, max_seq_length: int, dropout: float, base: int):
        super(GroupedQueryAttention, self).__init__()
        assert n_heads % n_groups == 0, '`n_heads` must be divisible by `n_groups`'

        #Meta Data
        self.d_model = d_model #32
        self.n_heads = n_heads #8
        self.n_groups = n_groups #2

        self.head_dim = d_model // n_heads #32 // 8 = 4
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])) #sqrt(d_k)

        #embedding
        self.q_rot_emb = RotaryPositionalEmbeddings(self.head_dim, max_seq_length, base)
        self.k_rot_emb = RotaryPositionalEmbeddings(self.head_dim, max_seq_length, base)

        #Layer
        self.w_q = Linear(d_model, d_model, bias=False) #(32 x 32)

        #share K와 V
        #head_dim(4) // n_groups(2) = 8
        self.w_k = Linear(d_model, self.head_dim * n_groups, bias=False) #(32 x 8)
        self.w_v = Linear(d_model, self.head_dim * n_groups, bias=False) #(32 x 8)


        self.dropout = Dropout(dropout)
        self.fc_out = Linear(d_model, d_model) #(32 x 32)


    @staticmethod
    def __masked_fill(tensor: Tensor, mask: Tensor, fill_value='-inf') -> Tensor:
        return tensor.masked_fill(mask==0, float(fill_value))
    

    @staticmethod
    def __get_causal_mask(size: int) -> Tensor:
        causal_mask = torch.ones(size, size)
        causal_mask = torch.tril(causal_mask)
        return causal_mask


    def scaledDotProductAttention(self, Q: Tensor, K: Tensor, 
                                  V: Tensor, attention_mask: Tensor) -> Tensor:
        '''
        Q ∈ (batch x seq x n_group x H/G(n_heads//group) x head_dim)
        K, V ∈ (batch x seq x n_groups x head_dim)
        attention_mask ∈(batch x seq)

        out ∈ (batch x seq x n_group x H/G(n_heads//group) x head_dim)
        '''
        device = Q.device

        seq = Q.size(1)

        #(batch x seq x n_group x H/G x seq)
        scores = torch.einsum('bnghd, bmgd -> bnghm', Q, K) / self.scale.to(device)

        #Attention Mask
        if attention_mask is not None:
            scores = self.__masked_fill(scores, attention_mask[:, None, None, None, :])

        #Causal Mask
        causal_mask = self.__get_causal_mask(seq).to(device).view(1, seq, 1, 1, seq)
        scores = self.__masked_fill(scores, causal_mask)


        attention_weights = F.softmax(scores, dim=-1) #Q(행)를 기준으로 softmax
        attention_weights = self.dropout(attention_weights)

        out = torch.einsum('bnghm,bmgd -> bnghd', attention_weights, V)
        return out

        
    def forward(self, Q: Tensor, K: Tensor, V: Tensor,
                attention_mask: Tensor=None) -> Tensor:
        '''
        Q, K, V ∈ (batch x seq x d_model)
        attention_mask ∈(batch x seq)

        out ∈ (batch x seq x d_model)
        '''
        batch, seq, _ = Q.shape

        #(batch x seq x d_model) -> (batch x seq x n_heads x head_dim)
        Q = self.w_q(Q).view(batch, seq, self.n_heads, self.head_dim)
        Q = self.q_rot_emb(Q)

        # (batch x seq x n_group x H/G(n_heads//group) x head_dim)
        Q = Q.view(batch, seq, self.n_groups, self.n_heads // self.n_groups, self.head_dim)


        #(batch x seq x d_model) -> (batch x seq x n_groups x head_dim)
        K = self.w_k(K).view(batch, seq, self.n_groups, self.head_dim)
        K = self.k_rot_emb(K)

        V = self.w_v(V).view(batch, seq, self.n_groups, self.head_dim)


        #(batch x seq x n_groups x H/G x head_dim) -> (batch x seq x d_model)
        out = self.scaledDotProductAttention(Q, K, V, attention_mask)
        out = out.contiguous().view(batch, seq, self.d_model)

        #(batch x seq x d_model)
        out = self.fc_out(out)
        return out


class GroupedQueryAttentionWithoutRoPE(Module):
    '''
    RoPE 미적용 GQA
    '''
    def __init__(self, d_model: int, n_heads: int, n_groups: int, dropout: float):
        super(GroupedQueryAttentionWithoutRoPE, self).__init__()
        assert n_heads % n_groups == 0, '`n_heads` must be divisible by `n_groups`'

        #Meta Data
        self.d_model = d_model #32
        self.n_heads = n_heads #8
        self.n_groups = n_groups #2

        self.head_dim = d_model // n_heads #32 // 8 = 4
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])) #sqrt(d_k)


        #Layer
        self.w_q = Linear(d_model, d_model, bias=False) #(32 x 32)

        #share K와 V
        #head_dim(4) // n_groups(2) = 8
        self.w_k = Linear(d_model, self.head_dim * n_groups, bias=False) #(32 x 8)
        self.w_v = Linear(d_model, self.head_dim * n_groups, bias=False) #(32 x 8)


        self.dropout = Dropout(dropout)
        self.fc_out = Linear(d_model, d_model) #(32 x 32)


    @staticmethod
    def __masked_fill(tensor: Tensor, mask: Tensor, fill_value='-inf') -> Tensor:
        return tensor.masked_fill(mask==0, float(fill_value))
    

    @staticmethod
    def __get_causal_mask(size: int) -> Tensor:
        causal_mask = torch.ones(size, size)
        causal_mask = torch.tril(causal_mask)
        return causal_mask


    def scaledDotProductAttention(self, Q: Tensor, K: Tensor, 
                                  V: Tensor, attention_mask: Tensor) -> Tensor:
        '''
        Q ∈ (batch x seq x n_group x H/G(n_heads//group) x head_dim)
        K, V ∈ (batch x seq x n_groups x head_dim)
        attention_mask ∈(batch x seq)

        out ∈ (batch x seq x n_group x H/G(n_heads//group) x head_dim)
        '''
        device = Q.device

        seq = Q.size(1)

        #(batch x seq x n_group x H/G x seq)
        scores = torch.einsum('bnghd,bmgd -> bnghm', Q, K) / self.scale.to(device)


        #Attention Mask
        if attention_mask is not None:
            #attention_mask view 수정할 필요 있음
            scores = self.__masked_fill(scores, attention_mask[:, None, None, None, :])


        #Causal Mask
        causal_mask = self.__get_causal_mask(seq).to(device).view(1, seq, 1, 1, seq)
        scores = self.__masked_fill(scores, causal_mask)


        attention_weights = F.softmax(scores, dim=-1) #Q(행)를 기준으로 softmax
        attention_weights = self.dropout(attention_weights)

        out = torch.einsum('bnghm,bmgd -> bnghd', attention_weights, V)
        return out

        
    def forward(self, Q: Tensor, K: Tensor, V: Tensor,
                attention_mask: Tensor=None) -> Tensor:
        '''
        Q, K, V ∈ (batch x seq x d_model)
        attention_mask ∈(batch x seq)

        out ∈ (batch x seq x d_model)
        '''
        batch, seq, _ = Q.shape

        #(batch x seq x d_model) -> (batch x seq x n_heads x head_dim) -> \
        # (batch x seq x n_group x H/G(n_heads//group) x head_dim)
        Q = self.w_q(Q).view(batch, seq, self.n_heads, self.head_dim)
        Q = Q.view(batch, seq, self.n_groups, self.n_heads // self.n_groups, self.head_dim)


        #(batch x seq x d_model) -> (batch x seq x n_groups x head_dim)
        K = self.w_k(K).view(batch, seq, self.n_groups, self.head_dim)
        V = self.w_v(V).view(batch, seq, self.n_groups, self.head_dim)


        #(batch x seq x n_groups x H/G x head_dim) -> (batch x seq x d_model)
        out = self.scaledDotProductAttention(Q, K, V, attention_mask)
        out = out.contiguous().view(batch, seq, self.d_model)

        #(batch x seq x d_model)
        out = self.fc_out(out)
        return out


class MultiHeadLatentAttention(Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_length: int, 
                 d_kv_comp: int, d_rope: int, dropout: float=0.1,  rope_base: int=10_000):
        super(MultiHeadLatentAttention, self).__init__()

        #Meta Data
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.max_seq_length = max_seq_length
        self.scale = torch.sqrt(torch.FloatTensor([self.d_head])) #sqrt(d_k)


        self.d_kv_comp = d_kv_comp #압축(compress)된 Key, Value 값에 대한 잠재적 차원
        self.d_rope = d_rope #Query, Key Head에 적용된 RoPE 차원
        self.d_split = self.d_head - d_rope #RoPE가 적용 되지 않는 q_c, k_c의 차원

        self.rope_base = rope_base


        #Q projection
        self.W_dq = Linear(d_model, d_kv_comp, bias=False)
        self.W_uq = Linear(d_kv_comp, n_heads * self.d_split, bias=False)


        #KV projection
        self.W_dkv = Linear(d_model, d_kv_comp, bias=False)
        self.W_uk = Linear(d_kv_comp, n_heads * self.d_split, bias=False)
        self.W_uv = Linear(d_kv_comp, n_heads * self.d_head, bias=False)


        #RoPE
        self.cos_cached: Tensor
        self.sin_cached: Tensor

        self.W_qr = Linear(d_kv_comp, n_heads * d_rope, bias=False)
        self.W_kr = Linear(d_model, n_heads * d_rope, bias=False)


        self.q_layernorm = RMSNorm(d_kv_comp)
        self.kv_layernorm = RMSNorm(d_kv_comp)

        self.dropout = Dropout(dropout)
        self.fc_out = Linear(d_model, d_model)

        self.rope_init()


    def rope_init(self) -> None:
        h_half = self.d_head // 2

        arange_ = torch.arange(0, self.d_head, 2)[: h_half].float()
        theta = 1.0 / (self.rope_base ** (arange_ / self.d_head))

        seq_idx = torch.arange(self.max_seq_length, dtype=theta.dtype)

        #(max_seq_len x h_half[head_dim // 2])
        idx_theta = torch.einsum('i, j -> ij', seq_idx, theta).float()


        #(1 x max_seq_len x 1 x h_half)
        cos_cached = idx_theta.cos().view(1, self.max_seq_length, 1, h_half)
        sin_cached = idx_theta.sin().view(1, self.max_seq_length, 1, h_half)

        #regist buffer
        self.register_buffer('cos_cached', cos_cached, persistent=False)
        self.register_buffer('sin_cached', sin_cached, persistent=False)


    @staticmethod
    def __get_causal_mask(size: int) -> Tensor:
        causal_mask = torch.ones(size, size)
        causal_mask = torch.tril(causal_mask)
        return causal_mask


    @staticmethod
    def __masked_fill(tensor: Tensor, mask: Tensor, fill_value='-inf') -> Tensor:
        return tensor.masked_fill(mask==0, float(fill_value))
    
    
    def __apply_rope_x(self, x: Tensor) -> Tensor:
        #x(batch x seq x n_heads x d_rope) -> x(batch x seq x n_heads x d_rope)
        seq = x.size(1)
        half_d = self.d_rope // 2

        #2묶음 단위로 회전하므로 repeat(1, 1, 1, 2) 수행
        cos = self.cos_cached[:, :seq, :, :half_d].repeat(1, 1, 1, 2)
        sin = self.sin_cached[:, :seq, :, :half_d].repeat(1, 1, 1, 2)

        rot_x = torch.cat(x.chunk(2, dim=-1), dim=-1) #rotate_half
        return (x * cos) + (rot_x * sin)
    

    def scaledDotProductAttention(self, Q: Tensor, K: Tensor, 
                                  V: Tensor, attention_mask: Tensor=None) -> Tensor:
        '''
        Q (batch x seq x n_heads x head_dim)
        K (batch x seq x n_heads x head_dim)
        V (batch x seq x n_heads x head_dim)

        return (batch x seq x n_heads x head_dim)
        '''
        device = Q.device

        seq_len = Q.size(1)


        #(batch x n_head x q_seq x k_seq)
        scores = torch.einsum('bnhd, bmhd -> bhnm', Q, K) / self.scale.to(device)
        

        #Attention Mask
        if attention_mask is not None:
            att_mask = attention_mask[:, None, None, :] #(batch x 1 x 1 x seq)
            scores = self.__masked_fill(scores, att_mask)

        
        #Causal Mask
        causal_mask = self.__get_causal_mask(seq_len).to(device).view(1, 1, seq_len, seq_len)
        scores = self.__masked_fill(scores, causal_mask)


        attention_weights = F.softmax(scores, dim=-1) #Q(행)를 기준으로 softmax
        attention_weights = self.dropout(attention_weights)

        #(batch x seq x n_heads x head_dim)
        return torch.einsum('bhnm, bmhd -> bnhd', attention_weights, V)


    def forward(self, x: Tensor, attention_mask: Tensor=None) -> Tensor:
        '''
        x(batch x seq x d_model)
        kv_cache(
            c_kv_cache(batch x seq x d_model),
            k_rot_cache(batch x seq x d_model),
        )


        return Tensor
        '''
        batch, seq, _ = x.shape

        #Q projection
        c_q = self.q_layernorm(self.W_dq(x))

        #q_base (batch x seq x n_heads x d_split)
        q_base = self.W_uq(c_q).view(batch, seq, self.n_heads, self.d_split)

        #q_R (batch x seq x n_heads x d_rope)
        q_rot = self.W_qr(c_q).view(batch, seq, self.n_heads, self.d_rope)
        q_rot = self.__apply_rope_x(q_rot)


        #get c_kv, k_rot
        c_kv = self.kv_layernorm(self.W_dkv(x))
        k_rot = self.W_kr(x)


        #K projection
        #k_base (batch x seq x n_heads x d_split)
        k_base = self.W_uk(c_kv).view(batch, seq, self.n_heads, self.d_split)

        #k_R (batch x seq x n_heads x d_rope)
        k_rot = k_rot.view(batch, seq, self.n_heads, self.d_rope)
        k_rot = self.__apply_rope_x(k_rot)


        #V projection
        #v (batch x seq x n_heads x d_head)
        v = self.W_uv(c_kv).view(batch, seq, self.n_heads, self.d_head)


        #split into multiple heads
        #(batch x seq x n_heads x d_head)
        q = torch.cat((q_base, q_rot), dim=-1)
        k = torch.cat((k_base, k_rot), dim=-1)


        #Scaled Dot Product (batch x seq x n_heads x d_head)
        out = self.scaledDotProductAttention(q, k, v, attention_mask)
        out = out.contiguous().view(batch, seq, self.d_model)


        #apply projection
        out = self.fc_out(out)
        return out


class MultiHeadLatentAttentionWithoutRoPE(Module):
    '''
    Roray Position Embedding을 적용하지 않은 MLA
    '''
    def __init__(self, d_model: int, n_heads: int, 
                 d_kv_comp: int, dropout: float=0.1):
        super(MultiHeadLatentAttentionWithoutRoPE, self).__init__()

        #Meta Data
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.scale = torch.sqrt(torch.FloatTensor([self.d_head])) #sqrt(d_k)

        self.d_kv_comp = d_kv_comp #압축(compress)된 Key, Value 값에 대한 잠재적 차원


        #Q projection
        self.W_dq = Linear(d_model, d_kv_comp, bias=False)
        self.W_uq = Linear(d_kv_comp, n_heads * self.d_head, bias=False)


        #KV projection
        self.W_dkv = Linear(d_model, d_kv_comp, bias=False)
        self.W_uk = Linear(d_kv_comp, n_heads * self.d_head, bias=False)
        self.W_uv = Linear(d_kv_comp, n_heads * self.d_head, bias=False)


        self.q_layernorm = RMSNorm(d_kv_comp)
        self.kv_layernorm = RMSNorm(d_kv_comp)

        self.dropout = Dropout(dropout)
        self.fc_out = Linear(d_model, d_model)


    @staticmethod
    def __get_causal_mask(size: int) -> Tensor:
        causal_mask = torch.ones(size, size)
        causal_mask = torch.tril(causal_mask)
        return causal_mask


    @staticmethod
    def __masked_fill(tensor: Tensor, mask: Tensor, fill_value='-inf') -> Tensor:
        return tensor.masked_fill(mask==0, float(fill_value))
    

    def scaledDotProductAttention(self, Q: Tensor, K: Tensor, 
                                  V: Tensor, attention_mask: Tensor=None) -> Tensor:
        '''
        Q (batch x seq x n_heads x head_dim)
        K (batch x seq x n_heads x head_dim)
        V (batch x seq x n_heads x head_dim)

        return (batch x seq x n_heads x head_dim)
        '''
        device = Q.device
        q_seq = Q.size(1)


        #(batch x n_head x q_seq x k_seq)
        scores = torch.einsum('bnhd, bmhd -> bhnm', Q, K) / self.scale.to(device)

        #Attention Mask
        if attention_mask is not None:
            att_mask = attention_mask[:, None, None, :] #(batch x 1 x 1 x seq)
            scores = self.__masked_fill(scores, att_mask)
        
        #Causal Mask
        causal_mask = self.__get_causal_mask(q_seq).to(device).view(1, 1, q_seq, q_seq)
        scores = self.__masked_fill(scores, causal_mask)

        attention_weights = F.softmax(scores, dim=-1) #Q(행)를 기준으로 softmax
        attention_weights = self.dropout(attention_weights)

        #(batch x seq x n_heads x head_dim)
        return torch.einsum('bhnm, bmhd -> bnhd', attention_weights, V)


    def forward(self, x: Tensor, attention_mask: Tensor=None) -> Tensor:
        '''
        x(batch x seq x d_model)
        kv_cache(
            c_kv_cache(batch x seq x d_model),
            k_rot_cache(batch x seq x d_model),
        )

        return Tensor #attention_out
        '''
        batch, seq, _ = x.shape

        #Q projection
        c_q = self.q_layernorm(self.W_dq(x))
        q = self.W_uq(c_q).view(batch, seq, self.n_heads, self.d_head)


        #get c_kv
        #c_kv_new: 현재 시점에서의 c_kv
        #c_kv_out: norm을 수행한 최종 c_kv
        #c_kv_cache: c_kv cache
        c_kv = self.kv_layernorm(self.W_dkv(x))

        #K projection
        #k (batch x seq[old + new] x n_heads x d_head)
        k = self.W_uk(c_kv).view(batch, seq, self.n_heads, self.d_head)

        #V projection
        #v (batch x seq[old + new] x n_heads x d_head)
        v = self.W_uv(c_kv).view(batch, seq, self.n_heads, self.d_head)

        #Scaled Dot Product (batch x seq x n_heads x d_head)
        out = self.scaledDotProductAttention(q, k, v, attention_mask)
        out = out.contiguous().view(batch, seq, self.d_model)

        #apply projection
        out = self.fc_out(out)
        return out
#End===================================================




#<Add & Norm Layer>====================================
class RMSNorm(nn.Module):
    '''
    pytorch의 `RMSNorm` Code
    '''
    def __init__(self, normalized_shape: Union[int, List[int], Size], eps: float = 1e-5, bias: bool = False) -> None:
        super(RMSNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(self.normalized_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        var = input.pow(2).mean(dim=-1, keepdim=True) + self.eps
        input_norm = input * torch.rsqrt(var)

        rmsnorm = self.weight * input_norm
        
        if self.bias is not None:
            rmsnorm = rmsnorm + self.bias

        return rmsnorm
#End===================================================




#<FFNN Layer>==========================================
class PositionWiseFeedForward(Module):
    def __init__(self, d_model: int, d_ff: int, droput: float):
        super(PositionWiseFeedForward, self).__init__()

        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)

        self.dropout = Dropout(droput)
        self.activation = nn.GELU()


    def forward(self, x: Tensor) -> Tensor:
        out = self.activation(self.fc1(x))
        out = self.dropout(out)
        return self.fc2(out)


class DeepseekMoE(Module):
    def __init__(self, d_model: int, d_ff: int, 
                 n_shared: int, n_expert: int, top_k: int, aux_alpha: float=0.003):
        super(DeepseekMoE, self).__init__()

        #Meta Data
        self.n_expert = n_expert
        self.top_k = top_k
        
        self.aux_alpha = aux_alpha
        self.aux_loss = torch.tensor([0.0])


        ffnn = lambda d_model, d_ff: PositionWiseFeedForward(d_model, d_ff, droput=0.0)
        self.shared_experts = nn.ModuleList([ffnn(d_model, d_ff) for _ in range(n_shared)])
        self.routed_experts = nn.ModuleList([ffnn(d_model, d_ff) for _ in range(n_expert)])
        
        self.gate = Linear(d_model, n_expert)


    def forward(self, x: Tensor) -> Tensor:
        '''
        x (batch x seq x d_model)
        '''
        device= x.device

        batch, seq, d_model = x.shape

        #Shared experts (batch x seq x d_model)
        shared_out = sum(expert(x) for expert in self.shared_experts)
        

        #Routing (batch x seq x n_expert)
        routed_logits = self.gate(x)
        probs = F.softmax(routed_logits, dim=-1)
        
        top_k_probs, top_k_indices = probs.topk(self.top_k, dim=-1)


        #Expert balance loss
        expert_counts = torch.zeros(batch, seq, self.n_expert).to(device)
        src = torch.ones(batch, seq, self.n_expert, dtype=torch.float).to(device) #(n_expert, )
        expert_counts.scatter_add_(index=top_k_indices, src=src, dim=-1)

        self.aux_loss = self.aux_loss.to(device) + expert_counts.float().var() * self.aux_alpha #aux_alpha = 0.003

        
        #Sparse computation
        #(batch x seq x d_model)
        routed_out = torch.zeros(batch, seq, d_model).to(device)

        for k in range(self.top_k):
            #채택된 expert_k(x) 계산 과정
            expert_contrib = torch.zeros(batch, seq, d_model).to(device) #=expert_k(x)

            #(batch x seq)
            expert_mask = top_k_indices[..., k] #batch, seq별 expert_k의 번호(indice)

            for expert_idx in range(self.n_expert):
                #expert 번호를 순회하며 채택된 expert가 존재할 시(expert_mask == expert_idx),
                #
                mask = (expert_mask == expert_idx)
                if mask.any(): # OR(mask) *하나라도 True가 있으면,
                    #x[mask] (len[x_b,s is True] x d_model) -> (len[x_b,s is True] x d_model)
                    expert_out = self.routed_experts[expert_idx](x[mask]) #expert_k(x)


                    #probs_out (len[x_b,s is True], ) ex) Tensor([0.1, 0.6, 0.3])
                    probs_out = top_k_probs[..., k][mask] 

                    expert_contrib[mask] = torch.einsum('nd, n -> nd', expert_out, probs_out)
                    
                    '''
                    batch, seq, d_model = 2, 3, 4

                    expert_cotrib = Tensor([ #(batch x seq x d_model)
                        [ b0
                            [0., 0., 0., 0.], s0
                            [0., 0., 0., 0.], s1
                            [0., 0., 0., 0.], s2
                        ],
                        [ b1
                            [0., 0., 0., 0.], s0
                            [0., 0., 0., 0.], s1
                            [0., 0., 0., 0.], s2
                        ],
                    ])

                    mask = Tensor([ #(batch x seq)
                          s0     s1     s2
                        [True, False, False], b0
                        [False, True, False], b1
                    ])
                    >=================================================================<

                    expert_out = Tensor([   #(len[x_b,s is True] x d_model)
                        [0.01, 0.02, 0.05, 0.08],
                        [0.07, 0.06, 0.01, 0.08],
                    ])

                    
                    probs_out = Tensor([0.5, 0.8])  #(len[x_b,s is True], )
                        
                    
                    expert_cotrib[mask] = torch.einsum('nd, n -> nd', expert_out, probs_out) -> \
                        Tensor([ #(len[x_b,s is True] x d_model)
                            [0.5 * 0.01, 0.5 * 0.02, 0.5 * 0.05, 0.5 * 0.08],
                            [0.8 * 0.07, 0.8 * 0.06, 0.8 * 0.01, 0.8 * 0.08],
                        )

                    >=================================================================<

                    expert_cotrib[mask] = Tensor([ #(batch x seq x d_model)
                        [0., 0., 0., 0.], b0, s0
                        [0., 0., 0., 0.], b1, s1
                    ])
                    

                    expert_cotrib = Tensor([ #(batch x seq x d_model)
                        [ b0
                            [0.5 * 0.01, 0.5 * 0.02, 0.5 * 0.05, 0.5 * 0.08], s0
                            [0., 0., 0., 0.], s1
                            [0., 0., 0., 0.], s2
                        ],
                        [ b1
                            [0., 0., 0., 0.], s0
                            [0.8 * 0.07, 0.8 * 0.06, 0.8 * 0.01, 0.8 * 0.08], s1
                            [0., 0., 0., 0.], s2
                        ],
                    ])
                    '''
            
            #routed_out = expert_0(x) + expert_1(x) + ... + expert_k(x)
            routed_out += expert_contrib
        
        return shared_out + routed_out
#End===================================================




#<Transforemr Block>===================================
class TransformerBlock(Module):
    '''
    Original Layer
    '''
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super(TransformerBlock, self).__init__()

        self.attention = MaskedMultiHeadAttention(d_model, n_heads, dropout)
        self.norm_att = LayerNorm(d_model)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm_ffn = LayerNorm(d_model)

        self.dropout = Dropout(dropout)

    
    def forward(self, x: Tensor, attention_mask: Tensor=None) -> Tensor:
        #Masked Multi Head Attention
        att_out = self.attention(Q=x, K=x, V=x, attention_mask=attention_mask)
        x = self.norm_att(x + self.dropout(att_out))

        #FFNN
        ffn_out = self.ffn(x)
        x = self.norm_ffn(x + self.dropout(ffn_out))
        return x
    

class PreNormTransformerBlock(Module):
    '''
    Add & Norm(Layer Normalization) 레이어가 Attention 및 FFNN 앞에 위치하는
    Pre-Norm 구조
    '''
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, eps: float=1e-6):
        super(PreNormTransformerBlock, self).__init__()
        
        self.norm_att = LayerNorm(d_model, eps=eps)
        self.attention = MaskedMultiHeadAttention(d_model, n_heads, dropout)

        self.norm_ffn = LayerNorm(d_model, eps=eps)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.dropout = Dropout(dropout)

    
    def forward(self, x: Tensor, attention_mask: Tensor=None) -> Tensor:
        #Masked Multi Head Attention
        norm_x = self.norm_att(x)
        att_out = self.attention(Q=norm_x, K=norm_x, V=norm_x, attention_mask=attention_mask)
        x = x + self.dropout(att_out)

        #FFNN
        norm_x = self.norm_ffn(x)
        ffn_out = self.ffn(norm_x)

        x = x + self.dropout(ffn_out)
        return x


class ALiBiTransformerBlock(Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 max_seq_length: int, dropout: float, eps: float=1e-6):
        super(ALiBiTransformerBlock, self).__init__()
        self.norm_att = LayerNorm(d_model, eps=eps)
        self.attention = ALiBiAttenion(d_model, n_heads, max_seq_length, dropout)

        self.norm_ffn = LayerNorm(d_model, eps=eps)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.dropout = Dropout(dropout)


    def forward(self, x: Tensor, attention_mask: Tensor=None) -> Tensor:
        #ALiBi Attention
        norm_x = self.norm_att(x)
        att_out = self.attention(Q=norm_x, K=norm_x, V=norm_x, attention_mask=attention_mask)
        x = x + self.dropout(att_out)

        #FFNN
        norm_x = self.norm_ffn(x)
        ffn_out = self.ffn(norm_x)

        x = x + self.dropout(ffn_out)
        return x


class RoPETransformerBlock(Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 max_seq_length: int, base: int, dropout: float, eps: float=1e-6):
        super(RoPETransformerBlock, self).__init__()
        self.norm_att = LayerNorm(d_model, eps=eps)
        self.attention = RoPEAttenion(d_model, n_heads, max_seq_length, dropout, base)

        self.norm_ffn = LayerNorm(d_model, eps=eps)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.dropout = Dropout(dropout)


    def forward(self, x: Tensor, attention_mask: Tensor=None) -> Tensor:
        #RoPE Attention
        norm_x = self.norm_att(x)
        att_out = self.attention(Q=norm_x, K=norm_x, V=norm_x, attention_mask=attention_mask)
        x = x + self.dropout(att_out)

        #FFNN
        norm_x = self.norm_ffn(x)
        ffn_out = self.ffn(norm_x)

        x = x + self.dropout(ffn_out)
        return x


class GroupedQueryTransformerBlock(Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 n_groups: int, max_seq_length: int, dropout: float, base: int, eps: float=1e-6):
        super(GroupedQueryTransformerBlock, self).__init__()
        self.norm_att = RMSNorm(d_model, eps=eps)
        self.attention = GroupedQueryAttention(d_model, n_heads, n_groups, max_seq_length, dropout, base)

        self.norm_ffn = RMSNorm(d_model, eps=eps)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.dropout = Dropout(dropout)


    def forward(self, x: Tensor, attention_mask: Tensor=None) -> Tensor:
        #GQ Attention
        norm_x = self.norm_att(x)
        att_out = self.attention(Q=norm_x, K=norm_x, V=norm_x, attention_mask=attention_mask)
        x = x + self.dropout(att_out)

        #FFNN
        norm_x = self.norm_ffn(x)
        ffn_out = self.ffn(norm_x)

        x = x + self.dropout(ffn_out)
        return x


class DeepseekTransformerBlock(Module):
    '''
    `n_shared`: 항상 활성화 되는 expert의 수
    `n_expert`: Routing에 의해 선택적으로 활성화되는 expert의 수
    `top_k`: 선택할 expert의 수
    `d_kv_comp`: 압축(compress)된 Key, Value 값에 대한 잠재적 차원
    `d_rope`: Query, Key Head에 적용된 RoPE 차원
    `aux_alpha`: MoE Layer의 aux_loss에 사용되는 상수

    >>> attention_layer = DeepseekTransformerBlock(...,)
    out = attention_layer(x: Tensor, kv_cache: tuple=None, attention_mask: Tensor=None)
    out = decoder_out #Tensor (batch x seq x d_model)

    `Deep Seek의 `Multi Head Latent Attention`을 적용한 Transfrmer Block
    'RoPE` 적용
    '''
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_length: int, 
                 n_shared: int, n_expert: int, top_k: int, d_kv_comp: int, d_rope: int, 
                 aux_alpha: float=0.003, dropout: float=0.1, rope_base: int=10_000):
        super(DeepseekTransformerBlock, self).__init__()

        self.attention = MultiHeadLatentAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_length=max_seq_length,
            d_kv_comp=d_kv_comp,
            d_rope=d_rope,
            dropout=dropout,
            rope_base=rope_base,
        )
        self.norm_att = RMSNorm(d_model)
        self.ffn = DeepseekMoE(d_model, d_ff, n_shared, n_expert, top_k, aux_alpha)
        self.norm_ffn = RMSNorm(d_model)

        self.dropout = Dropout(dropout)


    def forward(self, x: Tensor, attention_mask: Tensor=None) -> Tensor:
        '''
        >>> return decoder_out #Tensor (batch x seq x d_model)
        '''
        norm_x = self.norm_att(x)
        att_out = self.attention(x, attention_mask=attention_mask)
        x = x + self.dropout(att_out)

        #FFNN
        norm_x = self.norm_ffn(x)
        ffn_out = self.ffn(norm_x)

        x = x + self.dropout(ffn_out)
        return x


class DeepseekTransformerBlockWithoutRoPE(Module):
    '''
    `d_kv_comp`: 압축(compress)된 Key, Value 값에 대한 잠재적 차원
    `d_rope`: Query, Key Head에 적용된 RoPE 차원

    >>> attention_layer = DeepseekTransformerBlock(...,)
    out = attention_layer(x: Tensor, kv_cache: tuple=None, attention_mask: Tensor=None)
    out = decoder_out #Tensor (batch x seq x d_model)


    `Deep Seek의 `Multi Head Latent Attention`을 적용한 Transfrmer Block
    'RoPE` 적용
    '''
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, 
                 n_shared: int, n_expert: int, top_k: int, d_kv_comp: int, 
                 aux_alpha: float=0.003, dropout: float=0.1):
        super(DeepseekTransformerBlockWithoutRoPE, self).__init__()

        self.attention = MultiHeadLatentAttentionWithoutRoPE(
            d_model=d_model,
            n_heads=n_heads,
            d_kv_comp=d_kv_comp,
            dropout=dropout,
        )
        self.norm_att = RMSNorm(d_model)
        self.ffn = DeepseekMoE(d_model, d_ff, n_shared, n_expert, top_k, aux_alpha)
        self.norm_ffn = RMSNorm(d_model)

        self.dropout = Dropout(dropout)


    def forward(self, x: Tensor, attention_mask: Tensor=None) -> Tensor:
        '''
        >>> return decoder_out #Tensor (batch x seq x d_model)
        '''
        norm_x = self.norm_att(x)
        att_out = self.attention(x, attention_mask=attention_mask)
        x = x + self.dropout(att_out)

        #FFNN
        norm_x = self.norm_ffn(x)
        ffn_out = self.ffn(norm_x)

        x = x + self.dropout(ffn_out)
        return x
#End===================================================





