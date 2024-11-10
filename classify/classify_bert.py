import logging
import os
from datetime import datetime

import torch
from sklearn import metrics

from task.bert.bert_base_net import BertNet
from task.bert.dataset import get_dataloader
from train import Train
from loss import get_loss_fn
from optimizer import get_optimizer, get_scheduler

FILE_ROOT_DIR = os.path.dirname(__file__)


def training():
    logging.basicConfig(level=logging.INFO)
    now_str = datetime.now().strftime("%Y%m%d%H%M%S")
    data_file_dir = os.path.join(FILE_ROOT_DIR, '../datas')
    save_dir = os.path.join(FILE_ROOT_DIR, '../output/bert/modules')
    log_dir = os.path.join(FILE_ROOT_DIR, f'../output/bert/log/{now_str}')
    bert_base_chinese_path = r"C:\Users\du\.cache\huggingface\hub\hub\bert-base-chinese"

    lr = 0.0001
    batch_size = 128
    total_epoch = 50
    device = 'cuda'
    stop_early = True
    stop_early_step = 5
    freeze = False
    train_dataloader, test_dataloader, num_classes, weights, bert_tokenizer = get_dataloader(
        data_file_dir=data_file_dir,
        bert_base_chinese_path=bert_base_chinese_path,
        batch_size=batch_size,
    )

    net = BertNet(num_classes=num_classes, bert_base_path=bert_base_chinese_path, freeze=freeze)
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
        example_inputs=torch.randint(100, size=(2, 10), dtype=torch.int64),
        stop_early=stop_early,
        stop_early_step=stop_early_step
    )
    trainer.fit()


if __name__ == '__main__':
    training()
