from abc import ABC, abstractmethod
from typing import Iterable, Tuple

import numpy as np
import torch
import tqdm

import number_helpers as nh


class Vocab(ABC):
    def __init__(self, samples:Iterable[str],
                 cutoff_freq:int=4, is_verbose=False):
        
        vocab_counts_no_cutoff:dict[str, int] = {}

        if is_verbose:
            samples = tqdm.tqdm(samples, total=len(samples))

        for sample in samples:
            tokens = [_ for _ in self.tokenize(sample)]

            for t in tokens:
                vocab_counts_no_cutoff[t] = vocab_counts_no_cutoff.get(t, 0) + 1
        
        vocab_counts = {k: v 
            for k, v in vocab_counts_no_cutoff.items() 
            if v >= cutoff_freq
        }

        count_ranks = sorted(vocab_counts, key=vocab_counts.get, reverse=True)
        numericizer:dict[str, int] = {k: i for i, k in enumerate(count_ranks)}

        unk_num = max(numericizer.values()) + 1

        self.numericizer = numericizer
        self.unk_num = unk_num
        self.vocab_rev:dict[int, str] = {v:k for k, v in numericizer.items()}
        self.vocab_rev[unk_num] = 'UNK'
    
    @staticmethod
    @abstractmethod
    def tokenize(txt:str) -> list[str]:
        pass
    
    def numericize(self, tokens:Iterable[str]|str) -> list[int]:
        if type(tokens) is str:
            return self.numericize(self.tokenize(tokens))
        
        return [self.numericizer.get(t, self.unk_num) for t in tokens]

    def denumericize(self, ixs:Iterable[int]):
        return [self.vocab_rev[ix] for ix in ixs]

    def __len__(self) -> int:
        return len(self.numericizer)


class LowercaseVocab(Vocab):
    @staticmethod
    def tokenize(txt:str) -> list[str]:
        return txt.lower().split()


class NumberAwareMixin(LowercaseVocab):
    def __init__(self, samples: Iterable[str], 
                 cutoff_freq: int = 4, is_verbose=False,
                 num_token:str='_NUM_'):
        
        self.num_token = num_token
        
        super().__init__(samples, cutoff_freq, is_verbose)
    
    def tokenize(self, txt:str, **_):
        return self.number_aware_tokenize(txt, **_)[0]

    def number_aware_tokenize(self, txt:str, 
            clamp:Tuple[int, int]=(1, 1000),
            number_locations:Iterable[int]=None,
            number_values:Iterable[float]=None,
            augment_vary_ratio:float=None,
            do_replace_num:bool=True) -> Tuple[list[str], np.ndarray, np.ndarray]:
        """
        >>> sample = 'hello world bun 3.5 </s> To finalize bun 30'
        >>> samples = [sample]
        >>> nav = NumberAwareMixin(samples, cutoff_freq=0)
        >>> tokens, is_numbers, numbers = nav.number_aware_tokenize(samples[0])
        >>> tokens
        ['hello', 'world', 'bun', '_NUM_', '</s>', 'to', 'finalize', 'bun', '_NUM_']
        >>> is_numbers.nonzero()[0]
        array([3, 8])
        >>> numbers[[3, 8]]
        array([ 3.5, 30. ], dtype=float32)
        >>> nav.number_aware_tokenize(samples[0], do_replace_num=False)[0]
        ['hello', 'world', 'bun', '3.5', '</s>', 'to', 'finalize', 'bun', '30']
        >>> avr = .1
        >>> get_augmented_numbers = lambda: nav.number_aware_tokenize(sample, augment_vary_ratio=avr)[2][[3,8]]
        >>> import numpy as np
        >>> tok_many = np.array([get_augmented_numbers() for _ in range(200)])
        >>> mns = tok_many.mean(axis = 0)
        >>> e_mns = np.array([3.5, 30.])
        >>> mns < e_mns * 1.1
        array([ True,  True])
        >>> mns > e_mns * .9
        array([ True,  True])
        >>> e_sds = np.array([_ * avr for _ in [3.5, 30.]])
        >>> e_sds
        array([0.35, 3.  ])
        >>> sds = np.std(tok_many, axis=0)
        >>> (e_sds < sds * 1.1).all()
        True
        >>> (e_sds > sds * .9).all()
        True
        >>> nav.number_aware_tokenize('hello world bun 3.5 </s> To finalize creatinine 5.5')[0]
        ['hello', 'world', 'bun', '_NUM_', '</s>', 'to', 'finalize', 'creatinine', '_NUM_']
        >>> len(nav)
        7
        >>> nav.number_aware_tokenize('hello world')[0]
        ['hello', 'world']
        >>> nav.number_aware_tokenize('hello world bun 3.5 </s> To finalize bun 30', do_replace_num=False)[0]
        ['hello', 'world', 'bun', '3.5', '</s>', 'to', 'finalize', 'bun', '30']
        """
        tokens = (
            super().tokenize(txt) if isinstance(txt, str)
            else txt
        )   

        if number_locations is None:
            is_numbers:np.ndarray = np.array([nh.is_num(_) for _ in tokens])
            number_locations:np.ndarray = np.flatnonzero(is_numbers)
        if number_locations is not None:
            is_numbers = np.zeros(len(tokens), dtype=bool)
            is_numbers[number_locations] = True

        if number_values is None:
            number_values = np.array(
                [
                    nh.to_float(tokens[_]) for _ in number_locations
                ],
                dtype=np.single
            )
        
        if do_replace_num:
            for nl in number_locations:
                tokens[nl] = self.num_token
        
        if augment_vary_ratio is not None:
            augments = np.single(
                np.random.normal(1, augment_vary_ratio, len(number_values))
            )
            number_values = number_values * augments
            
        if clamp is not None:
            number_values = np.clip(number_values, clamp[0], clamp[1])
        
        numbers=np.zeros(len(tokens), dtype=np.single)
        numbers[number_locations] = number_values
        
        return tokens, is_numbers, numbers


    def denumericize(self, nums:torch.Tensor, is_numbers:np.ndarray, numbers: np.ndarray) -> str:
        if 'denumericizer' not in dir(self):
            self.denumericizer = {v:k for k,v in self.numericizer.items()}
        
        return [
                str(numbers[ix]) if is_numbers[ix]
                else self.denumericizer.get(num, '_UNK_')
                for ix, num in enumerate(nums)
            ]


class NumberContextAwareVocab(NumberAwareMixin):
    def __init__(self, samples: Iterable[str], 
                 context_words:list[str],
                 n_left:int = 2, n_right:int = 1,
                 cutoff_freq: int = 4, is_verbose=False,
                 context_num_token:str='_NUM_',
                 non_context_num_token='_INUM_',
                 do_keep_non_context_nums = False,
                 do_keep_context_nums=False):
        
        self.context_words = context_words
        self.n_left = n_left
        self.n_right = n_right
        self.context_num_token = context_num_token
        self.non_context_num_token = non_context_num_token
        self.do_keep_context_nums=do_keep_context_nums
        self.do_keep_non_context_nums = do_keep_non_context_nums
        
        super().__init__(samples, cutoff_freq, is_verbose, num_token=context_num_token)

    def number_aware_tokenize(self, txt:str, 
            clamp:Tuple[int, int]=(1, 1000),
            number_locations:Iterable[int]=None,
            number_values:Iterable[float]=None,
            augment_vary_ratio:float=None) -> Tuple[list[str], np.ndarray, np.ndarray]:
        """
        >>> sample = 'hello world bun 3.5 </s> To finalize bun 30 and a birth date in 2020'
        >>> samples = [sample]
        >>> nav = NumberContextAwareVocab(samples, cutoff_freq=0, context_words = ['bun', 'creatinine'])
        >>> tokens, is_numbers, numbers = nav.number_aware_tokenize(samples[0])
        >>> tokens
        ['hello', 'world', 'bun', '_NUM_', '</s>', 'to', 'finalize', 'bun', '_NUM_', 'and', 'a', 'birth', 'date', 'in', '_INUM_']
        >>> is_numbers.nonzero()[0]
        array([3, 8])
        >>> numbers[[3, 8]]
        array([ 3.5, 30. ], dtype=float32)
        >>> avr = .1
        >>> get_augmented_numbers = lambda: nav.number_aware_tokenize(sample, augment_vary_ratio=avr)[2][[3,8]]
        >>> import numpy as np
        >>> tok_many = np.array([get_augmented_numbers() for _ in range(200)])
        >>> mns = tok_many.mean(axis = 0)
        >>> e_mns = np.array([3.5, 30.])
        >>> mns < e_mns * 1.1
        array([ True,  True])
        >>> mns > e_mns * .9
        array([ True,  True])
        >>> e_sds = np.array([_ * avr for _ in [3.5, 30.]])
        >>> e_sds
        array([0.35, 3.  ])
        >>> sds = np.std(tok_many, axis=0)
        >>> (e_sds < sds * 1.1).all()
        True
        >>> (e_sds > sds * .9).all()
        True
        >>> nav.number_aware_tokenize('hello world bun 3.5 </s> To finalize creatinine 5.5')[0]
        ['hello', 'world', 'bun', '_NUM_', '</s>', 'to', 'finalize', 'creatinine', '_NUM_']
        >>> len(nav)
        13
        >>> nav.number_aware_tokenize('hello world')[0]
        ['hello', 'world']
        >>> nav_nums = NumberContextAwareVocab(
        ...    samples, cutoff_freq=0, context_words = ['bun', 'creatinine'], 
        ...    do_keep_non_context_nums=True)
        >>> tokens, is_numbers, numbers = nav_nums.number_aware_tokenize('there are 10 apples and bun 3.0')
        >>> tokens
        ['there', 'are', '10', 'apples', 'and', 'bun', '_NUM_']
        >>> numbers
        array([ 0.,  0., 10.,  0.,  0.,  0.,  3.], dtype=float32)
        >>> nav_con = NumberContextAwareVocab(
        ...    samples, cutoff_freq=0, context_words = ['bun', 'creatinine'], 
        ...    do_keep_non_context_nums=False, do_keep_context_nums=True)
        >>> tokens, is_numbers, numbers = nav_con.number_aware_tokenize('there are 10 apples and bun 3.0')
        >>> tokens
        ['there', 'are', '_INUM_', 'apples', 'and', 'bun', '3.0']
        >>> numbers
        array([ 0.,  0., 10.,  0.,  0.,  0.,  3.], dtype=float32)
        """
        tokens, is_numbers, numbers = super().number_aware_tokenize(
            txt, clamp=clamp, 
            number_locations=number_locations, number_values=number_values,
            augment_vary_ratio=augment_vary_ratio,
            do_replace_num=not self.do_keep_context_nums)

        token_locs = np.flatnonzero(
            [t in self.context_words for t in tokens]
        )

        number_locs = np.flatnonzero(is_numbers)
        number_window_locs = [(_ - self.n_left, _, _ + self.n_right) for _ in number_locs]
        
        def has_any(left, right):
            return any(i >= left and i <= right for i in token_locs)
        
        passing_locs = np.array([
            loc for l, loc, r in number_window_locs
            if has_any(l, r)
        ])

        is_non_context_numbers = is_numbers.copy()

        is_numbers[:] = False
        if(len(passing_locs) > 0):
            is_numbers[passing_locs] = True
            is_non_context_numbers[passing_locs] = False

        if not self.do_keep_non_context_nums:
            for _ in np.flatnonzero(is_non_context_numbers):
                tokens[_] = self.non_context_num_token

        if not self.do_keep_context_nums:
            for _ in np.flatnonzero(is_numbers):
                tokens[_] = self.context_num_token
        
        return tokens, is_numbers, numbers


    def denumericize(self, nums:torch.Tensor, is_numbers:np.ndarray, numbers: np.ndarray) -> str:
        if 'denumericizer' not in dir(self):
            self.denumericizer = {v:k for k,v in self.numericizer.items()}
        
        return [
                str(numbers[ix]) if is_numbers[ix]
                else self.denumericizer.get(num, '_UNK_')
                for ix, num in enumerate(nums)
        ]


if __name__ == '__main__':
    import doctest
    doctest.testmod()
