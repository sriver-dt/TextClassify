import logging
import os
from datetime import datetime

import torch
from sklearn import metrics

from task.word2vec.net import Net
from task.word2vec.dataset import get_dataloader
from train import Train
from loss import get_loss_fn
from optimizer import get_optimizer, get_scheduler

FILE_ROOT_DIR = os.path.dirname(__file__)


def training():
    logging.basicConfig(level=logging.INFO)
    now_str = datetime.now().strftime("%Y%m%d%H%M%S")
    data_file_dir = os.path.join(FILE_ROOT_DIR, '../datas')
    save_dir = os.path.join(FILE_ROOT_DIR, '../output/word2vec/modules')
    save_vec_model_dir = os.path.join(FILE_ROOT_DIR, '../output/word2vec/vec_model')
    log_dir = os.path.join(FILE_ROOT_DIR, f'../output/word2vec/log/{now_str}')

    lr = 0.05
    batch_size = 256
    total_epoch = 50
    device = 'cuda'
    retrain_weight = True
    word2vec_epoch = 50
    stop_early = True
    stop_early_step = 5

    train_dataloader, test_dataloader, token2idx, num_classes, weights, word_vec = get_dataloader(
        data_file_dir=data_file_dir, save_vec_model_dir=save_vec_model_dir, batch_size=batch_size,
        retrain_weight=retrain_weight, vec_epoch=word2vec_epoch
    )

    vocab_size, emb_size, = word_vec.shape
    hidden_size = 512

    net = Net(vocab_size=vocab_size, emb_size=emb_size, hidden_size=hidden_size, emb_weight=word_vec,
              num_classes=num_classes)
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
        example_inputs=torch.randint(vocab_size, size=(2, 10), dtype=torch.int64),
        stop_early=stop_early,
        stop_early_step=stop_early_step
    )
    trainer.fit()


if __name__ == '__main__':
    training()
