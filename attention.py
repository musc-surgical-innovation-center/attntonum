import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn


def get_scaled_dot_product_attn_weights(q:torch.Tensor, k:torch.Tensor, 
        keep_mask:torch.Tensor=None)->torch.Tensor:
    context_len = k.shape[2]

    scores = torch.bmm(
        q.transpose(2, 1), #n_numbers by Q by e
        k #n_numbers by e by c_len
    )/np.sqrt(context_len) # n_numbers by Q by c_len

    if keep_mask is not None:
        ignore_terms_(scores, keep_mask)

    return F.softmax(scores, dim = -1) #n_numbers by Q by c_len


def ignore_terms_(scores:torch.Tensor, keep_mask:torch.Tensor) -> torch.Tensor:
    """
    >>> import torch
    >>> n_number, Q, c_len = 3, 2, 5
    >>> scores = torch.arange(30, dtype=torch.float).view( (n_number, Q, c_len) )
    >>> scores[0]
    tensor([[0., 1., 2., 3., 4.],
            [5., 6., 7., 8., 9.]])
    >>> keep_mask = (torch.arange(c_len) % 2).repeat( (n_number, 1) ) == 0
    >>> keep_mask[0]
    tensor([ True, False,  True, False,  True])
    >>> ignore_terms_(scores, keep_mask)
    >>> scores
    tensor([[[ 0., -inf,  2., -inf,  4.],
             [ 5., -inf,  7., -inf,  9.]],
    <BLANKLINE>
            [[10., -inf, 12., -inf, 14.],
             [15., -inf, 17., -inf, 19.]],
    <BLANKLINE>
            [[20., -inf, 22., -inf, 24.],
             [25., -inf, 27., -inf, 29.]]])
    """
    Q = scores.shape[1]
    scores[
        (keep_mask==False) #n_numbers by c_len
        .unsqueeze(dim = 1) #n_numbers by 1 by c_len
        .expand(-1, Q, -1) #n_numbers by Q by c_len
    ] = -torch.inf


class AttentionHead(nn.Module):
    def __init__(self, embed_dim:int, head_dim:int):
        super().__init__()
        self.query_lin = nn.Linear(embed_dim, head_dim)
        self.key_lin = nn.Linear(embed_dim, head_dim)
        self.value_lin = nn.Linear(embed_dim, head_dim)
    
    def forward(self, context_embeds:torch.Tensor, keep_mask:torch.Tensor, 
                number_ix:int=2) -> torch.Tensor:
        q, k, v = self.get_qkv(context_embeds, number_ix)
        weights = get_scaled_dot_product_attn_weights(
            q, k, keep_mask) #n_numbers by Q by c_len
        
        return torch.bmm(
            v, #n_numbers by e_h by c_len
            weights.transpose(2, 1), #n_numbers by c_len by Q
        ) #n_numbers by e_h by Q
    
    def get_qkv(self, context_embeds:torch.Tensor, number_ix:int=None):
        q, k, v = [
            lin.forward(
                context_embeds.transpose(2, 1) #n_numbers by c_len by e
            ).transpose(2, 1) #n_numbers by e_h by c_len
            for lin in [self.query_lin, self.key_lin, self.value_lin]
        ]
        if number_ix is not None:
            q = q[:, :, number_ix].unsqueeze(dim = 2) #n_numbers by e_h by Q
        
        return q,k,v


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim:int, n_heads:int) -> None:
        super().__init__()
        head_dim = int(embed_dim/n_heads)

        assert head_dim * n_heads == embed_dim
        
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(n_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, context_embeds:torch.Tensor, keep_mask:torch.Tensor, number_ix:int=2)->torch.Tensor:
        attns = [_.forward(context_embeds, keep_mask, number_ix).squeeze() for _ in self.heads]
        attn = torch.cat(attns, dim=-1)
        return self.output_linear(attn)


if __name__=='__main__':
    import doctest
    doctest.testmod()
