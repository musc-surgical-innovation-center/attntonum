from toolz import pipe as p
import torch
from torch import nn

import datasets as ds
import embeds as e
import hyperparam_config as config


class TextLstm(nn.Module):
    def __init__(self, sequence_len:int, n_terms:int, 
                 model_params:config.TextLstmParams=config.TextLstmParams()):
        super(TextLstm, self).__init__()

        dropout = model_params.dropout
        embed_size = model_params.embed_size
        lstm_dim = model_params.lstm_dim

        self.embed = nn.Embedding(n_terms, embed_size)
        self.lstm = nn.LSTM(input_size = embed_size, hidden_size = lstm_dim,
                            dropout = dropout, bidirectional = True,
                            batch_first = True)
        #self.avg = nn.AvgPool1d(sequence_len)
        self.maxpool = nn.MaxPool1d(sequence_len)

        #cat_len = lstm_dim * 2 * 2 # 2 directions, 2 pool
        self.linear = nn.Linear(lstm_dim*2, 1)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, **_):
        pre_pool = p(
            x,
            self.run_embed,
            self.lstm,
            lambda _: _[0], #output, not hidden state
            lambda _: _.swapaxes(2, 1)
        )

        mx = self.maxpool(pre_pool).squeeze()

        return p(mx,
            self.linear,
            self.dropout,
            self.mk_logit_pair,
        )
    
    def run_embed(self, x:torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.embed.forward(x)
    
    def mk_logit_pair(self, base_out:torch.Tensor):
        if base_out.dim() == 1:
            base_out = base_out.unsqueeze(0)
        return torch.cat([-base_out, base_out], dim = 1)


class TextLstmScaleNum(TextLstm):
    def __init__(self, sequence_len: int, n_terms: int, 
            model_params: config.LstmScaleNumParams = config.LstmScaleNumParams()):
        super().__init__(sequence_len, n_terms, model_params)

        self.number_embed = e.ScaleNumEmbed(
            model_params.embed_size,
            model_params.post_lin_activation, model_params.do_pre_lin_log,
            n_number_lins=model_params.n_number_lins
        )
    
    def run_embed(self, x:dict[ds.NumberAwareKeys, torch.Tensor],):
        embeds =  TextLstm.run_embed(self, x['nums'])
        return self.number_embed.forward(embeds, x)
    

class TextLstmAttnToNum(TextLstmScaleNum):
    def __init__(self, sequence_len:int, n_terms:int,
                 model_params: config.AttnToNumParams = config.AttnToNumParams()):
        super().__init__(sequence_len, n_terms, model_params)
        mp = model_params
        self.attn_embed = e.AttnToNumEmbed(
            mp.n_left, mp.n_right,
            mp.embed_size,
            mp.n_attn_heads
        )
    
    def run_embed(self, x: dict[ds.NumberAwareKeys, torch.Tensor]):
        embeds = TextLstmScaleNum.run_embed(self, x)
        return self.attn_embed.forward(embeds, x['is_numbers'])


if __name__=='__main__':
    import doctest
    doctest.testmod()

