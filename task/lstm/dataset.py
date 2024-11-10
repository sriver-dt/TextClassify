import copy
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


class TextDataset(Dataset):
    def __init__(self, X_, Y_):
        super(TextDataset, self).__init__()
        self.X_ = X_
        self.Y_ = Y_

    def __len__(self):
        return len(self.X_)

    def __getitem__(self, index):
        x_ = self.X_[index]
        y_ = self.Y_[index]
        return copy.deepcopy(x_), y_, len(x_)

    @staticmethod
    def my_collate_fn(batch):
        batch_x, batch_y, batch_x_len = list(zip(*batch))
        max_len = max(batch_x_len)
        masks = []
        for i in range(len(batch_x)):
            x = batch_x[i]
            mask = np.zeros(max_len)
            mask[: len(x)] = 1
            masks.append(mask)
            if len(x) < max_len:
                x.extend([0] * (max_len - len(x)))
        batch_x = torch.tensor(batch_x, dtype=torch.int64)
        batch_y = torch.tensor(batch_y, dtype=torch.int64)
        masks = torch.tensor(masks, dtype=torch.float32)
        return (batch_x, masks), batch_y


def get_dataloader(data_file_dir, batch_size):
    datas = pd.read_csv(os.path.join(data_file_dir, 'train.csv'), sep='\t')
    x = datas.text.values
    y = datas.label.values

    # 构建词典
    token2idx = {'<PAD>': 0, '<UNK>': 1}
    X = []
    for text in x:
        x = []
        for token in text.strip().split(' '):
            try:
                token_id = token2idx[token]
            except KeyError:
                token_id = len(token2idx)
                token2idx[token] = token_id
            x.append(token_id)
        X.append(x)

    with open(os.path.join(data_file_dir, 'token2idx.json'), 'w', encoding='utf-8') as writer:
        json.dump(token2idx, writer)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with open(os.path.join(data_file_dir, 'label2idx.json'), 'r', encoding='utf-8') as reader:
        label2idx = json.load(reader)

    # 类别不均衡，计算每个类别的权重
    num_classes = len(label2idx)
    class_counts = datas.label.value_counts()
    weights = class_counts.sum() / (class_counts + 1)    # 加1防止类别为0异常
    weights = np.clip(weights, num_classes, num_classes * 1.1)
    weights = torch.softmax(torch.tensor(weights.values), dim=0)
    weights = torch.tensor(weights, dtype=torch.float32).view(-1)

    train_dataset = TextDataset(x_train, list(y_train))
    test_dataset = TextDataset(x_test, list(y_test))

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=TextDataset.my_collate_fn
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=batch_size*5, collate_fn=TextDataset.my_collate_fn
    )

    return train_dataloader, test_dataloader, token2idx, num_classes, weights
