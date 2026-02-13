from __future__ import print_function, division
import sys
import os
import argparse
import copy
import yaml
import json
import inspect

import numpy as np

from collections import OrderedDict, namedtuple
import torch
import torch.nn.parallel
import torch.utils.data
import torch.autograd as autograd
import torch.nn as nn
from .logger import Logger as Log


def mkdir(path):
    if is_list_or_tuple(path):
        for p in path:
            if not os.path.exists(p):
                os.makedirs(p)
    elif is_str(path):
        if not os.path.exists(path):
            os.makedirs(path)
        # os.makedirs(path, exist_ok=True)  # explicitly set exist_ok when multi-processing

def save_args(args, path, modelpara, pipepara, trainpara, filename='args.json'):
    # args_dict = vars(args)
    cfg = copy.deepcopy(args)
    cfg = vars(cfg)  # convert to dict

    mkdir(path)

    cfg_dict = {"model": {}, "pipe": {}, "trainer": {}, "extra": {}}
    model_key = vars(modelpara).keys()
    pip_key = vars(pipepara).keys()
    train_key = vars(trainpara).keys()
    for key, value in cfg.items():
        if key in model_key or "_" + key in model_key:
            cfg_dict["model"][key] = cfg[key]
        elif key in pip_key or "_" + key in pip_key:
            cfg_dict["pipe"][key] = cfg[key]
        elif key in train_key or "_" + key in train_key:
            cfg_dict["trainer"][key] = cfg[key]
        else:
            cfg_dict['extra'][key] = cfg[key]
        if key == "data_name":
            open(os.path.join(path, 'data_name: ' + value), 'a').close()

    save_path = os.path.join(path, filename)

    with open(save_path, 'w') as f:
        json.dump(cfg_dict, f, indent=4, sort_keys=False)


def str2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
    
def int2bool(v):
    """ Usage:
    parser.add_argument('--x', type=int2bool, nargs='?', const=True,
                        dest='x', help='Whether to use pretrained models.')
    """
    if int(v) == 1:
        return True
    elif int(v) == 0:
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
    
    
def is_str(x):
    return isinstance(x, str)

def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))

def isNum(n):
    try:
        # n=eval(n)
        if type(n)==int or type(n)==float or type(n)==complex:
            return True
    except NameError:
        return False