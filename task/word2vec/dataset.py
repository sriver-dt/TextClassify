import copy
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec


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


def get_dataloader(data_file_dir, batch_size, save_vec_model_dir, retrain_weight, vec_epoch=10):
    os.makedirs(save_vec_model_dir, exist_ok=True)
    datas = pd.read_csv(os.path.join(data_file_dir, 'train.csv'), sep='\t')
    texts = datas.text.values
    labels = datas.label.values

    # gensim 初步训练词向量
    sentences = [text.strip().split(' ') for text in datas.text.values]
    if retrain_weight:
        word2vec = Word2Vec(
            sentences=sentences, vector_size=1024, epochs=vec_epoch, window=5, sg=1, negative=5, min_count=0
        )
        word2vec.save(os.path.join(save_vec_model_dir, 'word2vec.model'))
    else:
        word2vec = Word2Vec.load(os.path.join(save_vec_model_dir, 'word2vec.model'))

    # 更新训练权重
    # word2vec.train(datas.text.values, epochs=50, total_words=word2vec.corpus_count)
    # 获取词向量
    vectors = word2vec.wv
    # 构建词向量的tensor对象作为后续训练embedding参数
    vocab_size, vector_size = len(vectors.index_to_key), word2vec.vector_size
    words_vec = torch.zeros(vocab_size+2, vector_size)      # 前两个为'<PAD>'和'<UNK>'预留
    for i, word in enumerate(vectors.index_to_key):
        words_vec[i + 2] = torch.tensor(vectors[word], dtype=torch.float32)

    # 获取词典
    token2idx = vectors.key_to_index
    for word, idx in token2idx.items():
        token2idx[word] = idx + 2
    token2idx['<PAD>'] = 0
    token2idx['<UNK>'] = 1
    token2idx = dict(sorted(token2idx.items(), key=lambda item: item[1]))

    # 将文本数据转换为 token id 序列
    X = []
    for text in texts:
        x = []
        for token in text.strip().split(' '):
            try:
                token_id = token2idx[token]
            except KeyError:
                token_id = token2idx['<UNK>']
            x.append(token_id)
        X.append(x)

    with open(os.path.join(data_file_dir, 'token2idx_w2v.json'), 'w', encoding='utf-8') as writer:
        json.dump(token2idx, writer, ensure_ascii=False, indent=4)

    x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

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

    return train_dataloader, test_dataloader, token2idx, num_classes, weights, words_vec
