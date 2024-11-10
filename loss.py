import torch.nn as nn
from typing import Union, Callable


def get_loss_fn(loss_fn: Union[str, Callable] = 'cross_entropy_loss', weights=None, label_smoothing=0.0):
    if isinstance(loss_fn, str):
        loss_fn = loss_fn.lower()
        if loss_fn == 'cross_entropy_loss':
            # return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
            return nn.CrossEntropyLoss()
        elif loss_fn == 'bce_loss':
            return nn.BCELoss()
        elif loss_fn == 'nll_loss':
            return nn.NLLLoss()
    else:
        return loss_fn
