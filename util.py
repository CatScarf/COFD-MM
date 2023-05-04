import os
import re
import sys
import time
import random
import pickle
import hashlib
from typing import Any, List, Optional, Tuple, TypeVar, Union

import torch
import psutil
import numpy as np
from tqdm import tqdm
from torch import Tensor
from functools import wraps
from os.path import exists, join

real_dir = os.path.dirname(__file__)

def arg_str(arg: Any):
    if isinstance(arg, list):
        return str(arg[:3])
    else:
        return str(arg)

def cache(func):
    """Cache the result of a function call."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the cache dir.
        cache_dir = join(real_dir, 'cache')
        if not exists(cache_dir):
            os.mkdir(cache_dir)

        # Get the cache file path.
        md = hashlib.md5()
        args_str = ','.join(arg_str(arg) for arg in args)
        kwargs_str = ','.join(f'{k}={arg_str(v)}' for k, v in kwargs.items())
        md.update(args_str.encode('utf-8'))
        md.update(kwargs_str.encode('utf-8'))
        args_hash = md.hexdigest()

        args_hash = str(args_hash)[-10:]
        cache_file_name = f'{func.__name__}-{args_hash}.pkl'
        cache_path = join(cache_dir, cache_file_name)

        # If the cache file exists, load it.
        if exists(cache_path):
            # print(f'[cache] Cache file {cache_file_name} exists, loading...', end='\r')
            start_time = time.time()
            with open(cache_path, 'rb') as f:
                res = pickle.load(f)
                duration = f'{time.time() - start_time:.2f}s'
                # print(f'[cache] Cache file {cache_file_name} exists, loading... Done({duration}).')
                return res
        # Otherwise, call the function and save the result.
        else:
            # print(f'[cache] No cache found for {cache_file_name}, calling...')
            result = func(*args, **kwargs)
            # print(f'[cache] Writing cache {cache_file_name}...', end='\r')
            start_time = time.time()
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            duration = f'{time.time() - start_time:.2f}s'
            # print(f'[cache] Writing cache {cache_file_name}... Done({duration}).')
            return result

    return wrapper

def frequency(x: Union[Tensor, np.ndarray]):
    """Compute the frequency of each element in x."""
    if isinstance(x, torch.Tensor):
        y: np.ndarray = x.detach().cpu().numpy()
    else:
        y: np.ndarray = np.array(x)
    res: List[Tuple[int, int]] = sorted(tuple(zip(*np.unique(y, return_counts=True))))
    res_str = ', '.join(f'{k}:{v}' for k, v in res)

    return res_str

def test_eq(expected: Any, actual: Any):
    """Test if two object's value are equal."""
    expected_str, actual_str = str(expected), str(actual)
    if expected_str == actual_str:
        res = f'Pass: {expected_str} == {actual_str}'
    else:
        res = f'Fail: {expected_str} != {actual_str}'
    res = re.sub(r'\s+', ' ', res)
    print(res if len(res) < 60 else res[:60] + ' ...')
    return res

def save_file(file_path: str, x: Any, head: Optional[str], mode: str):
    """Save to file."""
    with open(file_path, mode) as f:
        if head is not None:
            f.write(f'{head}:\n')
        if isinstance(x, list):
            for line in tqdm(x, desc=f'Saving {head}', leave=False):
                f.write(str(line) + '\n')
            f.write('\n')
        else:
            f.write(str(x) + '\n')

def set_seed(seed: int):
    """Set the seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def readable_bit(bit: int) -> str:
    """Convert bit to readable format."""
    if bit < 1024:
        return f'{bit}B'
    elif bit < 1024 * 1024:
        return f'{bit / 1024:.2f}KB'
    elif bit < 1024 * 1024 * 1024:
        return f'{bit / 1024 / 1024:.2f}MB'
    else:
        return f'{bit / 1024 / 1024 / 1024:.2f}GB'

def mem_usage(obj: Any = None):
    """Get memory usage."""
    all_memory = psutil.virtual_memory().total
    if obj is None:
        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss
    else:
        memory = sys.getsizeof(obj)
    ratio = memory / all_memory
    return f'{readable_bit(memory)}({ratio:.2%})'


def update_pbar(pbar: Optional[tqdm], close: bool = False):
    """Update tqdm progress bar with memory usage."""
    if pbar is not None:
        pbar.set_postfix_str(f'{mem_usage()}')
        pbar.update()
        if close:
            pbar.close()

array = TypeVar('array', np.ndarray, Tensor)
def norm(x: array, dim: int = 0) -> array:
    if isinstance(x, np.ndarray):
        x = (x - x.mean(axis=dim, keepdims=True)) / (x.std(axis=dim, keepdims=True) + 1e-6)
    elif isinstance(x, Tensor):
        x = (x - x.mean(dim=dim, keepdim=True)) / (x.std(dim=dim, keepdim=True) + 1e-6)
    else:
        raise TypeError(f'x must be Tensor or np.ndarray, not {type(x)}')
    return x