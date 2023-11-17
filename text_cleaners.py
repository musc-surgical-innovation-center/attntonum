import codecs
import re
from typing import Callable, Iterable, Tuple
from toolz import identity, pipe as p
import unicodedata


def cleanTxt(x:str, 
    do_w2v = False, 
    do_remove_all_nums:bool=False, do_remove_decimals:bool=False,
    do_remove_large_ints:bool=True,
    sep_letters_numbers=False,
    sep_numbers_age=False,
    sep_numbers_weight=False,
    debug_stop_pt:int = None)->str:
  clean_txt_fxns =  [
    replaceSpecialUnicode,
    toAscii, 
    lambda _: _.lower(),
    removeUnderscore,
    replaceDateNumbers,
    replaceDateStrings,
    replaceTimes,
    removeAbbreviationPeriods,
    removeTitlePeriods,
    separateDashes,
    separateSpecialCharacters,
    removeNumberCommas,
    removePunctuation,
    removeBracketsStarQuote,
    separateNumbersAge if sep_numbers_age else identity,
    separateNumbersWeight if sep_numbers_weight else identity,
    separateLettersNumbers if sep_letters_numbers else identity,
    replaceNums if do_remove_all_nums else identity,
    removeDecimals if do_remove_decimals else identity,
    replaceLargeInts if do_remove_large_ints else identity,
    removeApostrophes,
    w2vPrep if do_w2v else putSentMarkers
  ]

  fxns = clean_txt_fxns if debug_stop_pt is None else clean_txt_fxns[0:debug_stop_pt]
  for fn in fxns:
      x = fn(x)
  
  return x


def replaceSpecialUnicode(sent:str):
  return sent.replace('×', 'x').replace('™','').replace('ï', 'i')#.replace('é','e')


def makeSpaceHandler(enc_error) -> Tuple[str, int]:
  repl = ' '
  start_again = min(enc_error.end, len(enc_error.object) - 1)
  return (repl, start_again)


codecs.register_error('spacereplace', makeSpaceHandler)


def toAscii(uni_str:str):
  return unicodedata.normalize(
    'NFKD', uni_str).encode(
      'ascii','replace').decode()


def removeUnderscore(sent:str):
  return re.sub('_', ' ', sent)


#range_regex = r"((?<=[^\w\-/])|^|\.)(\d+ ?(-|/) ?\d+)(?=[^\w\-/]|$)"


date_regex1 = r"\b\d{1,2}/\d{1,2}/(\d\d){1,2}\b"
date_regex2 = r"\b\d{1,2}-\d{1,2}-(\d\d){1,2}\b"
date_token = "_date_"
def replaceDateNumbers(sent:str):
  """
  >>> sent = 'who was admitted on 11/15/2024 for a cabg on 2-3-20'
  >>> replaceDateNumbers(sent)
  'who was admitted on _date_ for a cabg on _date_'
  """
  sent = re.sub(date_regex1, date_token, sent)
  return re.sub(date_regex2, date_token, sent)


rm_date2 = r"(\b)([A-Za-z]{3,9})(\s+)([0-9][0-9]*)(,)(\s+)([0-9]{4})"
def replaceDateStrings(sent:str):
  """
  >>> sent = 'who was admitted on June 15, 2024 for a cabg on June 3, 2020'
  >>> replaceDateStrings(sent)
  'who was admitted on _date_ for a cabg on _date_'
  """
  sent = re.sub(rm_date2, date_token, sent)
  
  return sent
  

time_token = "_time_"
time_regex = r"(\d{1,2}:\d{2}(:\d{2})?)"
def replaceTimes(sent:str):
  return re.sub(time_regex, time_token, sent)
  

abbrev_period_regex = r"(?<!\w)([a-z])\."
def removeAbbreviationPeriods(sent:str):
  return re.sub(abbrev_period_regex, r"\1", sent)


title_periods_regex = r"(dr|mr|ms|mrs|sr|jr)\."
def removeTitlePeriods(sent:str):
  return re.sub(title_periods_regex, r"\1", sent)


separate_dashes_regex = r"(?<=)(\S)-"
separate_dashes_replace = r"\1 -"
def separateDashes(sent:str):
  return re.sub(separate_dashes_regex, separate_dashes_replace, sent)


separate_special_characters_regex = r"([-±~])(\S)"
separate_special_characters_replace = r"\1 \2"
def separateSpecialCharacters(sent:str):
  return re.sub(separate_special_characters_regex, 
    separate_special_characters_replace, 
    sent)


remove_number_commas_regex = r"(\d),(\d)"
remove_number_commas_replace = r"\1\2"
def removeNumberCommas(sent:str):
  return re.sub(remove_number_commas_regex,
    remove_number_commas_replace,
    sent)
    

puncts = r"/(){}$+?@!|&%:,;<>=^#~"
remove_puncts_regex = "[" + puncts + "]"
remove_puncts_replace = r" "
def removePunctuation(sent:str):
  return re.sub(remove_puncts_regex,
    remove_puncts_replace,
    sent)


brackets_star_quote = ["[", "]", "*", '"']
remove_brackets_star_quote_regex = r"[\[*\"\]]"
remove_brackets_star_quote_replace = r" "
def removeBracketsStarQuote(sent:str):
  return re.sub(remove_brackets_star_quote_regex,
    remove_brackets_star_quote_replace,
    sent)


separate_numbers_age_regex = r"(\d)(y[.|a-zA-Z]+)"
separate_numbers_age_replace = r"\1 \2"
def separateNumbersAge(sent:str):
  """
  >>> separateNumbersAge('a 72yo person who weighs 75kg and is not 82y.o. or 83yrs')
  'a 72 yo person who weighs 75kg and is not 82 y.o. or 83 yrs'
  """
  return re.sub(separate_numbers_age_regex,
    separate_numbers_age_replace,
    sent)


separate_numbers_kg_regex = r"(\d)(k[.|g]+)"
separate_numbers_kg_replace = r"\1 \2"
def separateNumbersWeight(sent:str):
  """
  >>> separateNumbersWeight('a 72yo person who weighs 75kg and is not 82y.o.')
  'a 72yo person who weighs 75 kg and is not 82y.o.'
  """
  return re.sub(separate_numbers_kg_regex,
    separate_numbers_kg_replace,
    sent)


separate_letters_numbers_regex = r"([a-z]+)([0-9]+)"
separate_letters_numbers_replace = r"\1 \2"
def separateLettersNumbers(sent:str):
  return re.sub(separate_letters_numbers_regex,
    separate_letters_numbers_replace,
    sent)
    

replace_nums_regex = r'[0-9]+'
replace_nums_replace = '#'
def replaceNums(sent:str):
  return re.sub(replace_nums_regex,
    replace_nums_replace,
    sent)


remove_decimals_regex = r'(\s)-?\d*\.\d+'
remove_decimals_replace = r'\1'
def removeDecimals(sent:str):
  return re.sub(remove_decimals_regex,
    remove_decimals_replace,
    sent)


replace_large_ints_regex = r'\d{4,}'
replace_large_ints_replace = '_lgnum_'
def replaceLargeInts(sent:str):
  """
  >>> txt = 'throat mrs: 123454321'
  >>> replaceLargeInts(txt)
  'throat mrs: _lgnum_'
  >>> replaceLargeInts('hello 123')
  'hello 123'
  >>> replaceLargeInts('hello 1234')
  'hello _lgnum_'
  """
  return re.sub(replace_large_ints_regex,
    replace_large_ints_replace,
    sent)
    

remove_apostrophes_regex = r"('|`)s?"
remove_apostrophes_replace = " "
def removeApostrophes(sent:str):
  return re.sub(remove_apostrophes_regex,
    remove_apostrophes_replace,
    sent)


def strSquish(sent:str):
  return " ".join(sent.split())


def putSentMarkers(sent:str):
  sent = re.sub(r"\.(?!\d)", " </s> ", sent)
  sent = re.sub(r"\n", " </s> ", sent)
  sent = re.sub(r"\r", "", sent)
  sent = sent.rstrip()
  sent = re.sub(r"(?<!</s>)$", " </s>", sent)
  return strSquish(sent)


def regexMapFlat(re_fn:Callable[[str], Iterable[str]], strings:Iterable[str]) -> list[str]: 
  return [s for string in strings for s in re_fn(string)]

def regexMap(re_fn:Callable[[str], str], strings:Iterable[str]) -> list[str]: 
  return [re_fn(string) for string in strings]
def w2vPrep(text:str) -> list[str]:
  return p(
    text, 
    lambda _: re.split(r"(?<=[\?\.])\s{1,2}(?=[A-Z|a-z])", _),
    lambda sent_vec: regexMapFlat(lambda s: re.split(r"\n", s), sent_vec),
    lambda sent_vec: regexMap(
      lambda s: re.sub(r"((?<![0-9])\.)|(\.(?![0-9]))", " ", s), ## replace periods after words, but not numbers
      sent_vec),
    lambda sent_vec: regexMap(strSquish, sent_vec)
    ) 


rm_number = r"(?<=^| )[-.]*\d+(?:\.\d+)?(?= |\.?$)|\d+(?:,\d{3})+(\.\d+)*"


pat_name_str = 'patient name'
_not_found_rpn = 100000
def removePatientName(text_lc:str):
  if not pat_name_str in text_lc:
    return text_lc
  
  def get_ix(s, start_ix = 0, def_val = _not_found_rpn):
    return (
      text_lc[start_ix:].index(s) + start_ix
      if s in text_lc[start_ix:]
      else def_val
    )
    
  start_ix = get_ix(pat_name_str)
  stop_ix = p(
    get_ix('mrn', start_ix), 
    lambda _: min(_, get_ix('medical record number', start_ix)),
    lambda _: min(_, get_ix('patient account', start_ix))
  )

  if stop_ix < 100:
    return text_lc.replace(text_lc[start_ix:stop_ix], ' ')
  else:
    return text_lc


if __name__ == "__main__":
  import doctest
  doctest.testmod()
