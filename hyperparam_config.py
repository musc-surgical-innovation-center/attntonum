from dataclasses import dataclass
from typing import Literal


@dataclass
class ClassifyParams():
    dropout:float = 0
    do_batch_norm:bool = False
    do_bias: bool = True


@dataclass
class EmbedParams():
    embed_size:int=50

@dataclass
class ScaleNumParams():
    do_pre_lin_log:bool=True
    post_lin_activation:Literal['sigmoid', 'identity', 'ReLU']='sigmoid'
    n_number_lins:int = 5
    do_resid_conn_scale_num:bool=False


@dataclass
class AttnToNumParams():
    n_left:int = 2
    n_right:int = 1
    n_attn_heads:int = 5
    do_resid_conn_attn_to_num:bool=False


@dataclass
class TextCnnParams(EmbedParams, ClassifyParams):
    n_filters: int = 36
    kernel_sizes: tuple = (1, 2, 3, 5)


@dataclass
class TextCnnScaleNumParams(TextCnnParams, ScaleNumParams):
    _filler:str=''


@dataclass
class TextCnnAttnToNumParams(TextCnnParams, ScaleNumParams, AttnToNumParams):
    _filler:str=''
    do_scale_num:bool=True


@dataclass
class TextLstmParams(EmbedParams, ClassifyParams):
    lstm_dim:int = 88


@dataclass
class LstmScaleNumParams(TextLstmParams, ScaleNumParams):
    _filler:str=''


@dataclass
class TextLstmAttnToNumParams(TextLstmParams, ScaleNumParams, AttnToNumParams):
    _filler:str=''
    do_scale_num:bool=True