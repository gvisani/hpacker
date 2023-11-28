
import argparse
from typing import *


def comma_sep_int_list(astr: Union[None, str]) -> List[int]:
    if astr is None or astr == 'None':
        return []
    else:
        return list(map(lambda x: int(x), astr.split(',')))

def comma_sep_float_list(astr) -> List[float]:
    if astr is None or astr == 'None':
        return []
    else:
        return list(map(lambda x: float(x), astr.split(',')))

def comma_sep_str_list(astr) -> List[str]:
    if astr is None or astr == 'None':
        return []
    else:
        return list(map(lambda x: str(x), astr.split(',')))

def str_to_bool(astr: Union[bool, str]) -> bool:
    if type(astr) == bool:
        return astr
    if astr.lower() == 'true':
        return True
    elif astr.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('%s is not a bool' % (astr))

def str_to_bool_or_float(astr: Union[bool, str]) -> bool:
    if type(astr) == bool or type(astr) == float:
        return astr
    if astr.lower() == 'true':
        return True
    elif astr.lower() == 'false':
        return False
    else:
        return float(astr)

def str_to_str_or_bool(astr: Union[bool, str]):
    if type(astr) == bool:
        return astr
    elif type(astr) == str:
        if astr.lower() == 'true':
            return True
        elif astr.lower() == 'false':
            return False
        else:
            return astr
    else:
        raise argparse.ArgumentTypeError('%s is not a str nor a bool' % (astr))

def str_to_str_or_bool_or_comma_sep_tuple_of_both(astr: Union[bool, str, Tuple[bool, str]]):
    if type(astr) == bool:
        return astr
    elif type(astr) == str:
        if len(astr.split(',')) == 2:
            return (str_to_bool(astr.split(',')[0]), astr.split(',')[1])
        else:
            return str_to_str_or_bool(astr)
    else:
        raise argparse.ArgumentTypeError('%s is not a str nor a bool, nor a tuple of two elements.' % (astr))

def optional_str(astr: Union[None, str]) -> Union[None, str]:
    if astr == 'None':
        return None
    else:
        return astr

def args_to_dict(args: argparse.ArgumentParser, ignore_params: Optional[Set] = None):
    adict = {}
    for arg in vars(args):
        if ignore_params is not None and arg in ignore_params:
            continue
        value = getattr(args, arg)
        if type(value) == list:
            value = ','.join(list(map(lambda x: str(x), value)))
        adict[arg] = value
    return adict