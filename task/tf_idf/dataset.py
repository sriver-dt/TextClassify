import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


class TextDataset(Dataset):
    def __init__(self, X, Y):
        super(TextDataset, self).__init__()
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)


def get_dataloader(data_file_dir, batch_size):
    datas = pd.read_csv(os.path.join(data_file_dir, 'train.csv'), sep='\t')
    texts = datas.text.values
    labels = datas.label.values
    x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.25, random_state=42)
    
    vectorizer = TfidfVectorizer()
    x_train_vec = vectorizer.fit_transform(x_train).toarray()
    x_test_vec = vectorizer.transform(x_test).toarray()

    train_dataset = TextDataset(x_train_vec, y_train)
    test_dataset = TextDataset(x_test_vec, y_test)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size*5)

    with open(os.path.join(data_file_dir, 'label2idx.json'), 'r', encoding='utf-8') as reader:
        label2idx = json.load(reader)

    # 类别不均衡，计算每个类别的权重
    num_classes = len(label2idx)
    class_counts = datas.label.value_counts()
    weights = class_counts.sum() / (class_counts + 1)    # 加1防止类别为0异常
    weights = np.clip(weights, num_classes, num_classes * 1.1)
    weights = torch.softmax(torch.tensor(weights.values), dim=0)
    weights = torch.tensor(weights, dtype=torch.float32).view(-1)

    return train_dataloader, test_dataloader, x_test_vec.shape[1], num_classes, weights
