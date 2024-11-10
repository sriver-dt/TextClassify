import torch.nn as nn
import torch.optim as optim
from typing import Callable, Union


def get_optimizer(net: nn.Module, lr: float, optim_name: Union[str, Callable] = 'sgd', **kwargs):
    group_weights, group_bias = [], []
    for name, params in net.named_parameters():
        if params.requires_grad is False:
            continue
        elif 'bias' in name:
            group_bias.append(params)
        else:
            group_weights.append(params)

    if isinstance(optim_name, str):
        optim_name = optim_name.lower()
        if optim_name == 'sgd':
            optimizer = optim.SGD(
                params=group_bias,
                lr=lr,
                momentum=kwargs.get('momentum', 0),  # 动量法的累计梯度的系数
                dampening=kwargs.get('dampening', 0),  # 动量法中当前梯度的系数值(1-dampening)
                weight_decay=0.0,  # 针对bias不进行惩罚性限制
                nesterov=kwargs.get('nesterov', False)  # 牛顿动量法
            )
            optimizer.add_param_group(
                param_group={
                    'params': group_weights,
                    'lr': lr * 0.5,
                    'weight_decay': kwargs.get('weight_decay', 0.0)
                }
            )
        elif optim_name == 'adam':
            optimizer = optim.Adam(params=group_bias, lr=lr)
            optimizer.add_param_group(
                param_group={
                    'params': group_weights,
                    'lr': lr * 0.5,
                    'weight_decay': kwargs.get('weight_decay', 0.0)
                }
            )
        elif optim_name == 'adamw':
            optimizer = optim.AdamW(params=group_bias, lr=lr)
            optimizer.add_param_group(
                param_group={
                    'params': group_weights,
                    'lr': lr * 0.5,
                    'weight_decay': kwargs.get('weight_decay', 0.0)
                }
            )
        else:
            raise ValueError(f'当前优化器不支持：{optim_name}')
        return optimizer
    else:
        return optim_name


def get_scheduler(opt, name: Union[str, Callable] = 'linear'):
    if name is None:
        return None
    elif isinstance(name, str):
        name = name.lower()
        if name == 'linear':
            scheduler = optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.1, total_iters=5)
        else:
            raise ValueError(f'当前优化器不支持：{name}')
    else:
        scheduler = optim.lr_scheduler.LambdaLR(opt, name)
    return scheduler
