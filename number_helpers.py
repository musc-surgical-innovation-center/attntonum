import re

import numbers


number = re.compile("^-?\d*?\.?\d*?$")
def is_num(s:str):
    """
    >>> is_num('3.5')
    True
    >>> is_num('3')
    True
    >>> is_num('the')
    False
    >>> is_num('.3')
    True
    >>> is_num('')
    False
    >>> is_num(' ')
    False
    >>> is_num('30.5')
    True
    >>> is_num('-1')
    True
    >>> is_num('- 1')
    False
    >>> is_num('0.20')
    True
    >>> is_num('.')
    False
    >>> is_num('-.')
    False
    >>> is_num('-')
    False
    >>> is_num('(30.0)')
    True
    """
    if len(s) == 0:
        return False
    if not has_any_number(s):
        return False
    return number.match(s.strip('()')) is not None


any_number = re.compile('.*\d.*')
def has_any_number(s:str):
    """
    >>> has_any_number('the')
    False
    >>> has_any_number('the 3')
    True
    >>> has_any_number('-')
    False
    >>> has_any_number('hello 42 world')
    True
    """
    return any_number.match(s) is not None


def to_float(token:str|numbers.Number) -> float:
    """
    >>> to_float('3.8')
    3.8
    >>> to_float('1')
    1.0
    >>> to_float(1)
    1.0
    >>> to_float('7.8..')
    7.8
    >>> to_float('(203)')
    203.0
    >>> to_float('0.03()')
    0.03
    """
    if isinstance(token, float):
        return token
    elif isinstance(token, numbers.Number):
        return float(token)
    else:
        return float(
            token
            .rstrip('.')
            .replace('(','')
            .replace(')','')
        )
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()