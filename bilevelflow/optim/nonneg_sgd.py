# Adapted from https://openreview.net/attachment?id=l0V53bErniB&name=supplementary_material

import torch
from torch.optim.sgd import SGD
from torch.optim.optimizer import required
import typing as _typing


class NONNEGSGD(SGD):
    '''
        Projected gradient descent with non-negative constraint, based on:
        #From: http://pmelchior.net/blog/proximal-matrix-factorization-in-pytorch.html

    '''
    def __init__(self, params, lr=required, momentum=0, dampening=0, nesterov=False):

        kwargs = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=0, nesterov=nesterov)
        super().__init__(params, **kwargs)

    def step(self, closure=None):
        # perform a gradient step
        # optionally with momentum or nesterov acceleration
        super().step(closure=closure)
        prox = torch.nn.Threshold(0,0) #Non-negative

        for group in self.param_groups:
            # apply the proximal operator to each parameter in a group
            for p in group['params']:
                p.data = prox(p.data)

#Some code borrowed from: 
#https://github.com/facebookresearch/higher/blob/5a4dee462c670b25f999505d7d6a32c4cc873fc7/higher/optim.py

def _add(
    tensor: torch.Tensor,
    a1: _typing.Union[float, int, torch.Tensor],
    a2: _typing.Optional[torch.Tensor] = None
) -> torch.Tensor:
    if a2 is None:
        value: _typing.Union[torch.Tensor, float] = 1.
        other = a1
    else:
        value = a1
        other = a2
    return tensor + (value * other)

_GroupedGradsType = _typing.List[_typing.List[torch.Tensor]]
