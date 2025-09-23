import torch.nn as nn
from functools import reduce

def get_nested_attr(obj, names):
    """
    Accesses a nested attribute (e.g., 'heads.head') from an object.
    """
    return reduce(getattr, names.split('.'), obj)

def set_nested_attr(obj, names, value):
    """
    Sets a nested attribute (e.g., 'heads.head') on an object.
    """
    parts = names.split('.')
    parent = reduce(getattr, parts[:-1], obj)
    setattr(parent, parts[-1], value)