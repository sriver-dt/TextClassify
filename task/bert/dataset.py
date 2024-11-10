import json
import os

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


class TextDataset(Dataset):
    def __init__(self, X, Y):
        super(TextDataset, self).__init__()
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x_ = self.X[index]
        y_ = self.Y[index]
        return x_, y_


def build_collate_fn(bert_tokenizer):
    def collate_fn(batch):
        batch_x, batch_y = list(zip(*batch))
        token_output = bert_tokenizer(
            batch_x,
            padding=True,
            truncation=True,        # 是否截断
            max_length=512,         # 最大长度
            return_tensors="pt"   # 转化为pytorch张量
           )
        batch_x = token_output['input_ids']
        batch_mask = token_output['attention_mask'].to(dtype=torch.float32)
        batch_y = torch.tensor(batch_y, dtype=torch.int64)
        return (batch_x, batch_mask), batch_y

    return collate_fn


def get_dataloader(data_file_dir, bert_base_chinese_path, batch_size):
    datas = pd.read_csv(os.path.join(data_file_dir, 'meta_data/train.csv'), header=None, sep='\t', names=['x', 'y'])
    with open(os.path.join(data_file_dir, 'label2idx.json'), 'r', encoding='utf-8') as reader:
        label2idx = json.load(reader)
        idx2label = {v: k for k, v in label2idx.items()}

    num_classes = len(idx2label)
    cnt = datas.y.value_counts()
    weights = np.array(sum(cnt) / cnt)
    weights = weights.clip(num_classes, num_classes * 1.2)
    weights = torch.softmax(torch.tensor(weights), dim=0)
    weights = torch.tensor(weights, dtype=torch.float32)

    Y = datas.y.apply(lambda label: label2idx[label])

    x_train, x_test, y_train, y_test = train_test_split(datas.x.values, list(Y), test_size=0.2, random_state=15)

    train_dataset = TextDataset(x_train, y_train)
    test_dataset = TextDataset(x_test, y_test)

    bert_tokenizer = BertTokenizer.from_pretrained(bert_base_chinese_path)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=build_collate_fn(bert_tokenizer)
    )

    test_dataloader = DataLoader(test_dataset, batch_size*5, collate_fn=build_collate_fn(bert_tokenizer))

    return train_dataloader, test_dataloader, num_classes, weights, bert_tokenizer


# if __name__ == '__main__':
#     FILE_ROOT_DIR = os.path.dirname(__file__)
#     data_file_dir = os.path.join(FILE_ROOT_DIR, '../../datas')
#     get_dataloader(data_file_dir, 16)
