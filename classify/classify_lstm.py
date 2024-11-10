import logging
import os
from datetime import datetime

import torch
from sklearn import metrics

from task.lstm.lstm import LstmNet
from task.lstm.dataset import get_dataloader
from train import Train
from loss import get_loss_fn
from optimizer import get_optimizer, get_scheduler


FILE_ROOT_DIR = os.path.dirname(__file__)


def training():
    logging.basicConfig(level=logging.INFO)
    now_str = datetime.now().strftime("%Y%m%d%H%M%S")
    data_file_dir = os.path.join(FILE_ROOT_DIR, '../datas')
    save_dir = os.path.join(FILE_ROOT_DIR, '../output/lstm/modules')
    log_dir = os.path.join(FILE_ROOT_DIR, f'../output/lstm/log/{now_str}')

    lr = 0.001
    batch_size = 256
    total_epoch = 50
    hidden_size = 512
    device = 'cuda'
    stop_early = True
    stop_early_step = 5

    train_dataloader, test_dataloader, token2idx, num_classes, weights = get_dataloader(
        data_file_dir=data_file_dir, batch_size=batch_size,
    )

    words_vocab_size = len(token2idx)

    net = LstmNet(vocab_size=words_vocab_size, hidden_size=hidden_size, num_class=num_classes)
    loss_fn = get_loss_fn(weights=weights, label_smoothing=0.1)
    optimizer = get_optimizer(net=net, lr=lr, optim_name='adam')
    scheduler = get_scheduler(opt=optimizer)
    trainer = Train(
        net=net,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        best_metric_func=lambda y_true, y_pred: metrics.f1_score(y_true=y_true, y_pred=y_pred, average='micro'),
        total_epoch=total_epoch,
        batch_size=batch_size,
        device=device,
        save_dir=save_dir,
        log_dir=log_dir,
        example_inputs=torch.randint(words_vocab_size, size=(2, 10), dtype=torch.int64),
        stop_early=stop_early,
        stop_early_step=stop_early_step
    )
    trainer.fit()


if __name__ == '__main__':
    training()
