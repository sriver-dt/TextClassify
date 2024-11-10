import logging
import os
from datetime import datetime

import torch
from sklearn import metrics

from task.stroke_emb.stroke_net import StrokeNet
from task.stroke_emb.dataset import get_dataloader
from train import Train
from loss import get_loss_fn
from optimizer import get_optimizer, get_scheduler

FILE_ROOT_DIR = os.path.dirname(__file__)


def training():
    logging.basicConfig(level=logging.INFO)
    now_str = datetime.now().strftime("%Y%m%d%H%M%S")
    data_file_dir = os.path.join(FILE_ROOT_DIR, '../datas')
    save_dir = os.path.join(FILE_ROOT_DIR, '../output/stroke_emb/modules')
    log_dir = os.path.join(FILE_ROOT_DIR, f'../output/stroke_emb/log/{now_str}')

    lr = 0.0001
    batch_size = 128
    total_epoch = 50
    device = 'cuda'
    stop_early = True
    stop_early_step = 5
    hidden_size = 512

    train_dataloader, test_dataloader, weights, num_classes, words_size, stroke_n_gram_counts = get_dataloader(
        data_file_dir_=data_file_dir, batch_size=batch_size
    )

    net = StrokeNet(vocab_size=words_size, emb_size=hidden_size, hidden_size=hidden_size,
                    stroke_n_gram_counts=stroke_n_gram_counts, num_classes=num_classes
                    )
    loss_fn = get_loss_fn(weights=weights, label_smoothing=0.1)
    optimizer = get_optimizer(net=net, lr=lr, optim_name='adamw')
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
        example_inputs=torch.randint(words_size, size=(2, 10), dtype=torch.int64),
        stop_early=stop_early,
        stop_early_step=stop_early_step
    )
    trainer.fit()


if __name__ == '__main__':
    training()
