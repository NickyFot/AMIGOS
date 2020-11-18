import numpy as np
import torch
from torch import nn


class AnnotatorsAverage(object):
    def __call__(self, x: torch.Tensor):
        return torch.mean(x, dim=1)


class ColumnSelect(object):
    def __init__(self, keys: list):
        self.keys = keys

    def __call__(self, feats: dict):
        feats = np.array([feats.get(key) for key in self.keys])
        return feats


def images_collate(batch):
    x_data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.stack(target, 0)
    x_data = nn.utils.rnn.pad_sequence(x_data, batch_first=True)
    x_data = x_data.permute(0, 2, 1, 3, 4)
    return x_data, target


def series_collate(batch):
    x_data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.stack(target, 0)
    x_data = nn.utils.rnn.pad_sequence(x_data, batch_first=True)
    x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], x_data.shape[2])
    return x_data, target