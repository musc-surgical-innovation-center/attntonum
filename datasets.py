from typing import Literal, Tuple

import numpy as np
import pandas as pd
from toolz import pipe as p
from torch.utils.data import Dataset
import tqdm

import vocabs as v


class TextDs(Dataset):
    def __init__(self, 
        data:pd.DataFrame,
        vocab:v.Vocab,
        sample_len:int,
        y_col:str,
        x_col:str='txt',
    ):
        self.data = data
        self.vocab = vocab
        self.sample_len = sample_len
        self.x_col = x_col
        self.y_col = y_col
    
    def __len__(self)->int:
        return len(self.data)
    
    def __getitem__(self, idx:int) -> Tuple[np.ndarray, bool|int|float]:
        sample:str = self.data[self.x_col].values[idx]
        y = self.data[self.y_col].values[idx]
        return (
            self.numericize_sample(sample),
            y
        )

    def numericize_sample(self, sample:str) -> np.ndarray:
        return p(sample, self.vocab.numericize, np.array, self.pad_nums)
    
    def pad_nums(self, nums:np.ndarray, pad_val:int=0) -> np.ndarray:
        return pad_nums(nums, self.sample_len, pad_val)


NumberAwareKeys = Literal['tokens', 'nums', 'numbers', 'is_numbers']
NumberAwareSample = dict[NumberAwareKeys, np.ndarray|Tuple[np.ndarray]]
class NumberAwareDs(TextDs):
    def  __init__(self, data: pd.DataFrame, vocab: v.NumberAwareMixin, sample_len: int, 
                  y_col: str, x_col: str = 'txt', 
                  augment_vary_ratio:float=None, is_verbose: bool = True,
                  do_cache:bool=True):
        super().__init__(data, vocab, sample_len, y_col, x_col)
        self.augment_vary_ratio = augment_vary_ratio
        self.number_locations_lo:dict[int, np.ndarray] = dict()
        self.number_values_lo:dict[int, np.ndarray] = dict()

        if do_cache:
            self._cache_locations_values(is_verbose)


    def numericize_sample(self, idx:int) -> NumberAwareSample:
        vocab:v.NumberAwareMixin = self.vocab
        sample:str = self.data[self.x_col].values[idx]
        
        tokens, is_numbers, numbers = vocab.number_aware_tokenize(sample,
            number_locations=self.number_locations_lo[idx],
            number_values=self.number_values_lo[idx],
            augment_vary_ratio=self.augment_vary_ratio
        )
        nums:np.ndarray = p(tokens, self.vocab.numericize, np.array)

        return {
            'tokens': tokens,
            'nums': self.pad_nums(nums),
            'numbers': self.pad_nums(numbers, 0),
            'is_numbers': self.pad_nums(is_numbers, False)
        }
    

    def __getitem__(self, idx:int) -> Tuple[np.ndarray, bool|int|float]:
        return (
            self.numericize_sample(idx),
            self.data[self.y_col].values[idx]
        )
    

    def _cache_locations_values(self, is_verbose:bool):
        samples = self.data[self.x_col].values
        samples_e = enumerate(samples)
        samples_l = (
            tqdm.tqdm(samples_e, total=len(samples))
            if is_verbose
            else samples_e
        )

        vocab:v.NumberAwareMixin = self.vocab

        for idx, sample in samples_l:
            _, is_numbers, numbers = vocab.number_aware_tokenize(sample)
            
            self.number_locations_lo[idx] = np.flatnonzero(is_numbers) #n_numbers
            self.number_values_lo[idx] = numbers[is_numbers] #n_numbers


def pad_nums(nums:np.ndarray, sample_len:int, pad_val:int=0) -> np.ndarray:
    """
    >>> import numpy as np
    >>> nums = np.array([3, 4, 5])
    >>> pad_nums(nums, 4, -1)
    array([ 3,  4,  5, -1])
    >>> pad_nums(nums, 2)
    array([3, 4])
    """
    n_nums = len(nums)
    pad = sample_len - n_nums

    return (
        nums[:sample_len] if pad <= 0
        else np.pad(nums, pad_width = (0, pad), constant_values = pad_val)
    )


if __name__ == "__main__":
    import doctest
    doctest.testmod()
