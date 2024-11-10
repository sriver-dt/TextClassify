import atexit
import logging
import os
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics


class Train:
    def __init__(self, net: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader,
                 loss_fn: Callable, optimizer: Optimizer, lr_scheduler: LRScheduler,
                 best_metric_func: Callable, total_epoch: int, batch_size: int,
                 device: str = 'cpu', save_dir: Union[str, os.PathLike] = None,
                 log_dir: Union[str, os.PathLike] = None, example_inputs: Optional[torch.Tensor] = None,
                 stop_early: bool = False, stop_early_step: int = 5):
        self.net = net
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.total_epoch = total_epoch
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size
        self.best_metric_func = best_metric_func
        self.best_score = 0.0
        self.start_epoch = 0
        self.train_step = 0
        self.stop_early = stop_early
        self.stop_early_step = stop_early_step
        self.device = torch.device('cuda') if device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')

        if save_dir is None:
            os.makedirs('./save', exist_ok=True)
            self.save_dir = './save'
        else:
            os.makedirs(save_dir, exist_ok=True)
            self.save_dir = save_dir

        if log_dir is None:
            os.makedirs('./logs', exist_ok=True)
            self.log_dir = './logs'
        else:
            os.makedirs(log_dir, exist_ok=True)
            self.log_dir = log_dir

        # 日志信息
        self.writer = SummaryWriter(self.log_dir)
        if example_inputs is not None:
            self.writer.add_graph(self.net, example_inputs)

        # 在程序结束时主动执行close方法
        atexit.register(self.close)

        # 模型恢复
        model_names = os.listdir(self.save_dir)
        model_path = ''
        if 'best_model.pkl' in model_names:
            model_path = os.path.join(self.save_dir, 'best_model.pkl')
        elif 'last_model.pkl' in model_names:
            model_path = os.path.join(self.save_dir, 'last_model.pkl')
        if model_path != '' and os.path.exists(model_path):
            logging.info('-' * 50)
            logging.info(f'开始进行模型参数恢复：{model_path}')
            save_data = torch.load(model_path)
            best_params = save_data['net_state_dict']
            self.start_epoch = save_data['epoch'] + 1
            self.total_epoch += self.start_epoch
            self.best_score = save_data['score']

            # best_param: 实际上就是一个dict字典，key就是参数名称字符串，value就是tensor对象值
            missing_keys, unexpected_keys = net.load_state_dict(best_params, strict=False)
            logging.info(f"未进行参数恢复的参数列表为:{missing_keys}")
            logging.info(f"额外给定的参数列表为:{unexpected_keys}")

    def fit(self):
        self.net.to(self.device)
        print(f'device: {self.device}')
        stop_counts = 0
        for epoch in range(self.start_epoch, self.total_epoch):
            logging.info(f'-----epoch: {epoch} 开始-----')
            # 训练
            self.train(epoch)
            # 更新学习率
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            score = self.eval(epoch)
            # 保存最好的模型
            if score >= self.best_score:
                self.best_score = score
                stop_counts = 0
                self.save('best', epoch, self.best_score)
            # 保存最后一个epoch的模型
            self.save('last', epoch, score)

            # 模型训练效果没有提升，提前停止
            if self.stop_early and score <= self.best_score:
                stop_counts += 1
                if stop_counts == self.stop_early_step:
                    logging.info(f'---epoch {epoch}---提前停止---  best_score {self.best_score}')
                    break

    def train(self, epoch):
        self.net.train()

        # 添加学习率日志信息
        for group_idx, param_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'lr_{group_idx}', param_group['lr'], global_step=epoch)

        for batch_idx, (inputs, labels) in enumerate(self.train_dataloader):
            self.train_step += 1
            if isinstance(inputs, tuple) or isinstance(inputs, list):
                if len(inputs) >= 4:
                    x, mask, x_stroke, stroke_mask = inputs
                    x = (x.to(self.device), mask.to(self.device), x_stroke.to(self.device), stroke_mask.to(self.device))
                else:
                    mask = inputs[1]
                    x = inputs[0]
                    x = (x.to(self.device), mask.to(self.device))
            else:
                x = (inputs.to(self.device),)
            y = labels.to(self.device)
            output = self.net(*x)
            self.optimizer.zero_grad()
            loss = self.loss_fn(output, y)
            loss.backward()
            self.optimizer.step()

            # 添加损失日志信息
            self.writer.add_scalar('train_loss', loss.item(), global_step=self.train_step)
            # 打印日志信息
            if batch_idx % 2 == 0:
                bl = (batch_idx + 1) * self.batch_size
                tl = len(self.train_dataloader.dataset)
                logging.info(f'epoch: {epoch + 1}/{self.total_epoch} '
                             f'{bl}/{tl} {(100.0 * bl) / tl:.2f}% loss: {loss.item():.6f}')

    def eval(self, epoch):
        self.net.eval()
        with torch.no_grad():
            test_y_pred, test_y_true = [], []
            for inputs, labels in self.test_dataloader:
                if isinstance(inputs, tuple) or isinstance(inputs, list):
                    if len(inputs) >= 4:
                        x, mask, x_stroke, stroke_mask = inputs
                        x = (
                            x.to(self.device), mask.to(self.device), x_stroke.to(self.device),
                            stroke_mask.to(self.device))
                    else:
                        mask = inputs[1]
                        x = inputs[0]
                        x = (x.to(self.device), mask.to(self.device))
                else:
                    x = (inputs.to(self.device),)
                y = labels.to(self.device)
                output = self.net(*x)
                output_softmax = torch.softmax(output, dim=-1)
                pred = torch.argmax(output_softmax, dim=1)
                test_y_pred.append(pred)
                test_y_true.append(y)

            test_y_true = torch.concatenate(test_y_true, dim=0).cpu().numpy()
            test_y_pred = torch.concatenate(test_y_pred, dim=0).cpu().numpy()

            # 打印评估信息
            confusion_matrix = metrics.confusion_matrix(y_true=test_y_true, y_pred=test_y_pred)
            report = metrics.classification_report(
                y_true=test_y_true, y_pred=test_y_pred, zero_division=0.0, output_dict=True
            )
            logging.info(f'epoch: {epoch} confusion_matrix:\n{confusion_matrix}\n')
            logging.info(f'epoch: {epoch} report:\n{report}\n')

            # 添加日志信息
            for label, value in report.items():
                if isinstance(value, dict):
                    for metric_name in value.keys():
                        self.writer.add_scalar(
                            f'eval_{label}_{metric_name}', value[metric_name], global_step=epoch
                        )
                else:
                    self.writer.add_scalar(
                        f'eval_{label}', value, global_step=epoch
                    )

            if self.best_metric_func is None:
                return sum(np.equal(test_y_true, test_y_pred)) / len(test_y_true)

            else:
                return self.best_metric_func(test_y_true, test_y_pred)

    def save(self, mode: str, epoch, score):
        model_state_dict = {
            'net_state_dict': self.net.state_dict(),
            'epoch': epoch,
            'score': score
        }
        torch.save(model_state_dict, os.path.join(self.save_dir, f'{mode}_model.pkl'))

    def close(self):
        logging.info('close resources...')
        self.writer.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __enter__(self):
        return self
