from abc import ABC, abstractmethod
from typing import Callable

from toolz import identity, partial, pipe as p
import torch
from torch.nn import functional as F
import torch.nn as nn

import attention as attn
import datasets as ds


class NumberFillEmbed(nn.Module):
    def __init__(self, embed_size:int, post_lin_activation:str, do_pre_lin_log:bool):
        super().__init__()
        self.number_embed = nn.Linear(1, embed_size)
        self.do_log = do_pre_lin_log
        self.post_lin_activation = post_lin_activation
        self.post_lin_activation:Callable[[torch.Tensor], torch.Tensor] = (
            torch.sigmoid if self.post_lin_activation=='sigmoid'
            else (
            torch.relu if self.post_lin_activation=='ReLU'
            else identity
            )
        )
    
    def forward(self, embeds:torch.Tensor, x: dict[ds.NumberAwareKeys, torch.Tensor], **_) -> torch.Tensor:
        numbers = x['numbers'] #n_sample by seq_len
        is_numbers = x['is_numbers'] #n_sample by seq_len

        number_embeds = self.run_number_embed(
            numbers[is_numbers] #n_numbers by 1
        ) #n_numbers by e

        #gradient error on directly set
        embeds[is_numbers] = 0*embeds[is_numbers] + number_embeds

        return embeds#n_sample by seq_len by e

    @abstractmethod
    def run_number_embed(self, numbers:torch.Tensor)->torch.Tensor:
        pass


class ScaleNumEmbed(NumberFillEmbed):
    def __init__(self, embed_size:int, 
                 post_lin_activation:str, do_pre_lin_log:bool, n_number_lins:int):
        super().__init__(
            embed_size=embed_size, post_lin_activation=post_lin_activation,
            do_pre_lin_log=do_pre_lin_log, 
        )
        self.number_lins = nn.ModuleList(
            [nn.Linear(1, embed_size) for _ in range(n_number_lins)]
        )


    def run_number_embed(self, numbers: torch.Tensor) -> torch.Tensor:
        ret = None
        for nl in self.number_lins:
            curr = self.run_number_lin(numbers, nl) #n_numbers by e
            if ret is None:
                ret = curr
            else:
                ret = curr + ret
        return ret


    def run_number_lin(self, numbers: torch.Tensor, ix:int|nn.Linear) -> torch.Tensor:
        lin = self.number_lins[ix] if isinstance(ix, int) else ix

        return p(
            numbers, #n_numbers
            torch.log if self.do_log else identity,
            partial(torch.unsqueeze, dim=1), #n_numbers by 1
            lin, #n_numbers by e
            self.post_lin_activation,
        )

    
class AttnToNumEmbed(nn.Module):
    def __init__(self, n_left:int, n_right:int, 
                 embed_size:int, n_attn_heads:int):
        super().__init__()
        self.n_left=n_left
        self.n_right = n_right
        self.attn_head = attn.MultiHeadAttention(embed_size, n_attn_heads)
    
    def forward(self, embeds:torch.Tensor, is_numbers:torch.Tensor, **_) -> torch.Tensor:
        if is_numbers.any():
            embeds[
                is_numbers #n_sample by seq_len
                ] = (
                    0*embeds[is_numbers] + 
                    self.calc_attn(embeds, is_numbers).squeeze()
                )
        
        return embeds #n_sample by seq_len by e


    def calc_attn(self, embeds:torch.Tensor, is_numbers:torch.Tensor)->torch.Tensor:
        context_embeds, keep_mask = self.get_context_and_keep_mask(
            embeds, is_numbers
        )

        return self.attn_head.forward(
            context_embeds, 
            keep_mask, 
            number_ix=self.n_left
        ) #n_numbers by e
    

    def get_context_and_keep_mask(self, embeds:torch.Tensor, is_numbers:torch.Tensor):
        n_left = self.n_left
        n_right = self.n_right

        number_locs_sample, number_locs_time = get_padded_number_locations(is_numbers, 
            n_left=n_left, n_right=n_right) #n_numbers by 2 (padded locations)
        
        embeds_pad = pad_embeds_left_right(embeds, 
            self.n_left, self.n_right) #n_sample by (seq_len + n_left + n_right) by e
        
        def get_context(i:int) -> torch.Tensor:
            return (
                embeds_pad[(number_locs_sample, number_locs_time+i)] #n_numbers by e
                .unsqueeze(dim=2) #n_numbers by e by 1
            )
        
        shifts = range(-self.n_left, self.n_right+1)
        context_embeds = torch.concat(
            [get_context(_) for _ in shifts], dim=2
        ) #n_numbers by e by c_len

        seq_len = is_numbers.shape[1]
        
        keep_mask = make_keep_mask(number_locs_time, seq_len, 
            n_left=n_left, n_right=n_right) #n_numbers by c_len

        return context_embeds, keep_mask


def pad_embeds_left_right(embeds:torch.Tensor, n_left:int, n_right:int) -> torch.Tensor:
    """
    >>> import torch
    >>> batch_embeds = torch.rand(3, 300, 50)
    >>> padded = pad_embeds_left_right(batch_embeds, 2, 1)
    >>> padded.size()
    torch.Size([3, 303, 50])
    >>> (padded[:, :2, :] == 0).all()
    tensor(True)
    >>> (padded[:, -1, :] == 0).all()
    tensor(True)
    """
    return F.pad(
            embeds,  #n_sample by seq_len by e
            (0, 0, n_left, n_right)
        ) #n_sample by (seq_len + n_left + n_right) by e


def get_padded_number_locations(is_numbers:torch.Tensor, n_left:int, n_right:int):
    """
    >>> import torch
    >>> is_numbers = torch.tensor([[False, True, False], [True, False, True]])
    >>> number_locs_sample, number_locs_seq = get_padded_number_locations(is_numbers, 2, 1)
    >>> number_locs_sample
    tensor([0, 1, 1])
    >>> number_locs_seq
    tensor([3, 2, 4])
    """
    is_numbers_pad = F.pad(is_numbers, (n_left, n_right)) #n_sample by (seq_len + n_left + n_right)
    return torch.nonzero(is_numbers_pad, 
        as_tuple=True) #n_numbers by 2


def are_in_pre_pad(context_i:int, post_pad_locs_time:torch.Tensor, pre_pad_seq_len:int, 
                  n_left:int=2) -> torch.Tensor:
    """
    >>> import torch
    >>> is_numbers = torch.tensor([[True, False, False], [False, True, False], [False, False, True]])
    >>> _, number_locs_seq = get_padded_number_locations(is_numbers, 2, 1)
    >>> number_locs_seq
    tensor([2, 3, 4])
    >>> are_in_pre_pad(0, number_locs_seq, is_numbers.shape[1])
    tensor([True, True, True])
    >>> are_in_pre_pad(1, number_locs_seq, is_numbers.shape[1])
    tensor([ True,  True, False])
    >>> are_in_pre_pad(-2, number_locs_seq, is_numbers.shape[1])
    tensor([False, False,  True])
    """
    pre_pad_seq_locs = post_pad_locs_time-n_left #n_numbers
    context_locs = pre_pad_seq_locs+context_i
    return (
        torch.logical_and(
            context_locs > -1, 
            context_locs < pre_pad_seq_len 
        )#n_numbers
    )


def make_keep_mask(post_pad_seq_locs_time:torch.Tensor, pre_pad_seq_len:int,
                   n_left=2, n_right=1):
    """
    >>> import torch
    >>> is_numbers = torch.tensor(
    ...    [[True, False, False, False], 
    ...     [False, True, False, False], 
    ...     [False, False, True, False],
    ...     [False, False, False, True]]
    ... )
    >>> _, number_locs_seq = get_padded_number_locations(is_numbers, 2, 1)
    >>> number_locs_seq
    tensor([2, 3, 4, 5])
    >>> make_keep_mask(number_locs_seq, is_numbers.shape[1])
    tensor([[False, False,  True,  True],
            [False,  True,  True,  True],
            [ True,  True,  True,  True],
            [ True,  True,  True, False]])
    """
    
    def are_in_pre_pad_curry(context_i:int) -> torch.Tensor:
        return are_in_pre_pad(
            context_i, post_pad_seq_locs_time, pre_pad_seq_len, 
            n_left=n_left
        ).unsqueeze(dim = 1)

    context_is = range(-n_left, n_right+1)

    return torch.concat(
        [
            are_in_pre_pad_curry(_) #n_numbers by 1
            for _ in context_is], dim=1
    ) #n_numbers by c_len


if __name__ == "__main__":
    import doctest
    doctest.testmod()
