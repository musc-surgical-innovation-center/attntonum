import re
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import tqdm

import datasets as ds


def default_tokenizer(txt:str) -> list[str]:
    return txt.lower().split()

Y = bool|int|float

def collate_batch(batch:list[Tuple[np.ndarray, Y]], 
        device:str='cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    mk_tensor = lambda ix: torch.tensor(
        [_[ix] for _ in batch]
    ).to(device)

    return mk_tensor(0), mk_tensor(1)


def collate_batch_number_aware(batch:list[Tuple[dict[ds.NumberAwareKeys, np.ndarray], Y]],
        device:str='cuda')->Tuple[dict[ds.NumberAwareKeys, torch.Tensor], torch.Tensor]:
    
    xs = {
        k:torch.concat(
            [torch.tensor(x[k]).unsqueeze(0) for x, _ in batch]
        ).to(device)
        for k in ['nums', 'numbers', 'is_numbers']
    }

    ys = torch.tensor([_[1] for _ in batch]).to(device)

    xs['tokens'] = [x['tokens'] for x, _ in batch]

    return xs, ys


def pull_concordance(df:pd.DataFrame, in_col:str, token:str,
                     out_col:str='concordance',
                     buffer:int=20, is_verbose:bool=True):
    """
    >>> import pandas as pd
    >>> df = pd.DataFrame(['creatinine of 4.3 and bun of 5.6'], columns=['txt'])
    >>> df2 = pull_concordance(df, 'txt', 'creatinine', is_verbose=False, buffer=7)
    >>> df2.concordance.values
    array(['creatinine of 4.3'], dtype=object)
    >>> df2 = pull_concordance(df2, 'txt', 'bun', out_col='bun_c', is_verbose=False, buffer=7)
    >>> df2.bun_c.values
    array(['.3 and bun of 5.6'], dtype=object)
    """
    samples = df[in_col].values

    if is_verbose:
        print(f'get concordance')
        samples = tqdm.tqdm(samples)
    
    reg = re.compile(rf"\b{token}\b")

    return df.assign(
        **{out_col: [pull_concordance_txt(_, token, buffer, reg) for _ in samples]}
    )


def pull_concordance_txt(txt:str, token:str, buffer:int=20, reg=None) -> str:
    if reg is None:
        reg = rf"\b{token}\b"

    match = re.search(reg, txt)
    if match:
        start, stop = match.span()
        return txt[
            max(start-buffer, 0):stop+buffer
        ]

    return ""


if __name__ == '__main__':
    import doctest
    doctest.testmod()
