"""
tools to convert specified type
"""
import torch as tc
import numpy as np


def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, tc.Tensor):
        return data.detach().cpu().numpy()
    return np.array(data)


def totensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = tc.from_numpy(data)
    elif isinstance(data, tc.Tensor):
        tensor = data.detach()
    else:
        tensor = tc.tensor(data)
    
    if cuda:
        tensor = tensor.cuda()
    return tensor


def toscalar(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, tc.Tensor):
        return data.item()
    return data
