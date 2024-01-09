from typing import Callable

from toolz import identity, pipe as p
import torch
from torch import nn

import datasets as ds
import embeds as e
import hyperparam_config as config


class TextCnn(nn.Module):
    def __init__(self, sequence_len:int, n_terms:int,
                model_params: config.TextCnnParams = config.TextCnnParams(),
        ):
        super(TextCnn, self).__init__()

        mp = model_params
        self.do_bias = mp.do_bias
        self.do_batch_norm = mp.do_batch_norm
        self.embed = nn.Embedding(n_terms, mp.embed_size)
        self.embed_size = mp.embed_size
        self.kernel_sizes = mp.kernel_sizes
        self.n_filters = mp.n_filters
        self.n_kernels = len(self.kernel_sizes)

        def conv1d(ks:int)->nn.Conv1d:
            return nn.Conv1d(in_channels=self.embed_size,
                             out_channels=self.n_filters,
                             kernel_size=ks,
                             bias=self.do_bias)

        def max_pool1d(ks:int): return nn.MaxPool1d(
            sequence_len - ks + 1,
        )

        def map_kernels(fn:Callable[[int], nn.Module], agg = nn.ModuleList):
            return agg(
                [fn(ks) for ks in self.kernel_sizes]
            )

        self.convolutions = map_kernels(conv1d)
        self.pools = map_kernels(max_pool1d)
        
        if self.do_batch_norm:
            self.norms = map_kernels(
                lambda _: nn.BatchNorm1d(self.n_filters))

        self.linear = nn.Linear(self.n_filters*self.n_kernels, 2)

        self.dropout = nn.Dropout(p=mp.dropout)


    def forward(self, x:torch.Tensor, **_)->torch.Tensor:
        return p(
            x,
            self.calc_pre_conv,
            self.calc_conv,
            self.calc_logits
        )


    def calc_pre_conv(self, x:torch.Tensor, **_)->torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        return (
            self.embed
            .forward(x) #n x seq_len x e
        )
    

    def calc_conv(self, pre_conv:torch.Tensor, 
                  pool_lo:Callable[[int], nn.Module]=None):
        if pool_lo is None:
            pool_lo = lambda _: self.pools[_]

        conv_outputs = torch.cat(
            [
                self.apply_conv_max(
                    pre_conv.swapaxes(2, 1), #n x e x seq_len
                    i, pool = pool_lo(i)
                ) 
                for i in range(self.n_kernels)
            ],
            dim=1)

        return conv_outputs
    

    def apply_conv_max(self, pre_conv:torch.Tensor, i:int, pool:nn.Module)->torch.Tensor:
        return p(
            pre_conv,
            self.convolutions[i],
            pool,
            self.norms[i] if self.do_batch_norm else identity
        )


    def calc_logits(self, conv_outputs:torch.Tensor)->torch.Tensor:
        return p(
            conv_outputs.squeeze(),
            self.dropout,
            self.linear
        )


class ScanCnn1Logit(TextCnn):
    def __init__(self, sequence_len:int, n_terms:int,
                model_params:config.TextCnnParams = config.TextCnnParams(),
        ):
        super(ScanCnn1Logit, self).__init__(
            sequence_len, model_params=model_params,
            n_terms=n_terms)

        self.linear = nn.Linear(self.n_filters*self.n_kernels, 1)
    
    def forward(self, x:torch.Tensor, **_):
        return p(
            x,
            super(ScanCnn1Logit, self).forward,
            self.mk_logit_pair
        )#self.mk_logit_pair(super(ScanCnn1Logit, self).forward(x))
    
    def mk_logit_pair(self, base_out:torch.Tensor):
        if base_out.dim() == 1:
            base_out = base_out.unsqueeze(0)
        return torch.cat([-base_out, base_out], dim = 1)


class TextCnnScaleNum(ScanCnn1Logit):
    def __init__(self, sequence_len:int, n_terms:int,
                 model_params: config.TextCnnScaleNumParams = config.TextCnnScaleNumParams()):
        super().__init__(sequence_len, n_terms, model_params)

        self.number_embed = e.ScaleNumEmbed(
            model_params.embed_size,
            model_params.post_lin_activation, model_params.do_pre_lin_log,
            n_number_lins=model_params.n_number_lins,
            do_resid_conn=model_params.do_resid_conn_scale_num,
        )
    
    def calc_pre_conv(self, x: dict[ds.NumberAwareKeys, torch.Tensor], **_) -> torch.Tensor:
        embeds = ScanCnn1Logit.calc_pre_conv(self, x['nums'])
        return self.number_embed.forward(embeds, x)


class TextCnnAttnToNum(TextCnnScaleNum):
    def __init__(self, sequence_len:int, n_terms:int,
                 model_params: config.TextCnnAttnToNumParams = config.TextCnnAttnToNumParams()):
        """
        >>> import torch
        >>> seq_len = 10
        >>> n_sample = 5
        >>> mx_vocab = 20
        >>> nums = torch.randint(mx_vocab, (n_sample, seq_len))
        >>> numbers = torch.randint(5, (n_sample, seq_len)) - 3
        >>> is_numbers = numbers > 0 #n_sample by seq_len
        >>> model = TextCnnAttnToNumModelParams(seq_len, mx_vocab)
        >>> x = dict(nums=nums, numbers=numbers, is_numbers=is_numbers)
        >>> embeds = model.embed(x['nums'])
        >>> attn_embeds = model.calc_pre_conv(x) #n_sample x seq_len x e
        >>> calc_same = attn_embeds[:, :, 0] == embeds[:, :, 0]
        >>> is_words = torch.logical_not(is_numbers)
        >>> (calc_same == is_words).all()
        tensor(True)
        >>> (torch.logical_not(calc_same) == is_numbers).all()
        tensor(True)
        """
        super().__init__(sequence_len, n_terms, model_params)
        mp = model_params
        self.attn_embed = e.AttnToNumEmbed(
            mp.n_left, mp.n_right,
             mp.embed_size,
             mp.n_attn_heads,
             do_resid_conn=mp.do_resid_conn_attn_to_num
        )

    def calc_pre_conv(self, x: dict[ds.NumberAwareKeys, torch.Tensor], **_) -> torch.Tensor:
        """
        >>> import torch
        >>> seq_len = 10
        >>> n_sample = 5
        >>> mx_vocab = 20
        >>> nums = torch.randint(mx_vocab, (n_sample, seq_len))
        >>> numbers = torch.randint(5, (n_sample, seq_len)) - 3
        >>> is_numbers = numbers > 0 #n_sample by seq_len
        >>> model = TextCnnAttnToNum(seq_len, mx_vocab)
        >>> x = dict(nums=nums, numbers=numbers, is_numbers=is_numbers)
        >>> embeds = model.embed(x['nums']) #n_sample x seq_len x e
        >>> number_embeds = model.number_embed(embeds, x) #n_sample x seq_len x e
        >>> attn_embeds = model.calc_pre_conv(x) #n_sample by seq_len by e
        >>> calc_same = attn_embeds[:, :, 0] == embeds[:, :, 0]
        >>> is_words = torch.logical_not(is_numbers)
        >>> (calc_same == is_words).all()
        tensor(True)
        >>> (torch.logical_not(calc_same) == is_numbers).all()
        tensor(True)
        >>> msg="Test pad embeds left right"
        >>> embeds_2_1 = model.attn_embed.pad_embeds_left_right(number_embeds)
        >>> embeds_2_1.shape
        torch.Size([5, 13, 50])
        >>> (embeds_2_1[:, :2, :] == 0).all()
        tensor(True)
        >>> (embeds_2_1[:, -1, :] == 0).all()
        tensor(True)
        >>> (embeds_2_1[:, 2:-1, :] == embeds).all()
        tensor(True)
        >>> import hyperparam_config as config
        >>> mp = config.TextCnnAttnToNumModelParams(n_left=1, n_right=0)
        >>> model_1_0 = TextCnnAttnToNum(seq_len, mx_vocab, mp)
        >>> embeds_1_0 = model_1_0.attn_embed.pad_embeds_left_right(embeds)
        >>> (embeds_1_0[:, 0, :] == 0).all()
        tensor(True)
        >>> (embeds_1_0[:, 1:, :] == embeds).all()
        tensor(True)
        >>> embeds_1_0.shape
        torch.Size([5, 11, 50])
        """
        is_numbers = x['is_numbers'] #n_sample by seq_len

        num_embeds = TextCnnScaleNum.calc_pre_conv(self, x) #n_sample by seq_len by e
        return self.attn_embed.forward(num_embeds, is_numbers)


class CnnLogitExtracts(nn.Module):
    def __init__(self, cnn_model:TextCnn,
                seq_length:int = 3000):
        super(CnnLogitExtracts, self).__init__()
        self.cnn_model = cnn_model
        self.cnn_model.train(mode = False)
        self.pools = [
            nn.MaxPool1d(kernel_size = seq_length - ks + 1,
                return_indices=True) 
            for ks in self.cnn_model.kernel_sizes
        ]
        self.train(mode = False)
    
    def forward(self, x:torch.Tensor, **_):
        conv_in = (
            self
            .cnn_model
            .calc_pre_conv(x)
            .swapaxes(2, 1) # n x e x seq_len
        )
        loc_activs = [
            self.cnn_model.apply_conv_max(conv_in, i, self.pools[i])
            for i in range(self.cnn_model.n_kernels)
        ]
        def cat(ix:int):
            return torch.cat([
                _[ix].squeeze().unsqueeze(dim = 1)
                for _ in loc_activs
            ], dim = 1)
        locs = cat(1)
        activs = cat(0)

        return locs, activs


if __name__=='__main__':
    import doctest
    doctest.testmod()

