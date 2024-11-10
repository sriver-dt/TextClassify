import copy
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


class TextDataset(Dataset):
    def __init__(self, X_, Y_, word_stroke):
        super(TextDataset, self).__init__()
        self.X_ = X_
        self.Y_ = Y_
        self.word_stroke = word_stroke

    def __len__(self):
        return len(self.X_)

    def __getitem__(self, index):
        x_ = self.X_[index]
        y_ = self.Y_[index]
        x_stroke = self.word_stroke[index]
        x_stroke_len = []
        for token in x_stroke:
            x_stroke_len.append(len(token))
        return copy.deepcopy(x_), y_, len(x_), copy.deepcopy(x_stroke), max(x_stroke_len)

    @staticmethod
    def my_collate_fn(batch):
        batch_x, batch_y, batch_x_len, batch_x_stroke, batch_x_stroke_len = list(zip(*batch))

        x_max_len = max(batch_x_len)
        x_stroke_max_len = max(batch_x_stroke_len)

        masks = []
        x_stroke_masks = []
        for i in range(len(batch_x)):
            x = batch_x[i]
            mask = np.zeros(x_max_len)
            mask[: len(x)] = 1
            masks.append(list(mask))
            if len(x) < x_max_len:
                x.extend([0] * (x_max_len - len(x)))

            text_stroke = batch_x_stroke[i]
            x_stroke_mask = torch.zeros(x_max_len, x_stroke_max_len)
            for j in range(len(text_stroke)):
                token = text_stroke[j]
                x_stroke_mask[j][:len(token)] = 1
                if len(token) < x_stroke_max_len:
                    token.extend([0] * (x_stroke_max_len - len(token)))
            if len(text_stroke) < x_max_len:
                zero_mask = [0] * x_stroke_max_len
                text_stroke.extend([zero_mask] * (x_max_len - len(text_stroke)))
            x_stroke_masks.append(x_stroke_mask.tolist())

        batch_x = torch.tensor(batch_x, dtype=torch.int64)
        batch_y = torch.tensor(batch_y, dtype=torch.int64)
        masks = torch.tensor(masks, dtype=torch.float32)
        batch_x_stroke = torch.tensor(batch_x_stroke, dtype=torch.int64)
        x_stroke_masks = torch.tensor(x_stroke_masks, dtype=torch.float32)

        return (batch_x, masks, batch_x_stroke, x_stroke_masks), batch_y


def get_char2stroke(stroke_path):
    stroke2id = {'横': '1', '提': '1', '竖': '2', '竖钩': '2', '撇': '3', '捺': '4', '点': '4'}

    char2stroke = {}
    with open(stroke_path, 'r', encoding='utf-8') as reader:
        for line in reader.readlines():
            line = line.strip().split(':')
            if len(line) == 2:
                strokes_lst = line[1].split(',')
                strokes = [stroke2id[stroke] if stroke in stroke2id else '5' for stroke in strokes_lst]
                char2stroke[line[0]] = ''.join(strokes)
    return char2stroke


def get_stroke_idx():
    strokes2idx = {'<UNK>': 0, '<ELSE>': 1}
    for a in range(1, 6):
        for b in range(1, 6):
            for c in range(1, 6):
                strokes2idx[str(a) + str(b) + str(c)] = len(strokes2idx)
                for d in range(1, 6):
                    strokes2idx[str(a) + str(b) + str(c) + str(d)] = len(strokes2idx)
                    for e in range(1, 6):
                        strokes2idx[str(a) + str(b) + str(c) + str(d) + str(e)] = len(strokes2idx)
    return strokes2idx


def get_token_n_gram(token, char2stroke, strokes2idx, smallest_n=3, biggest_n=5):
    token_stroke = ''
    for char in token:
        if char not in char2stroke:
            return [0]
        token_stroke += char2stroke[char]  # 获取字对应的笔画序列

    token_n_gram = []
    for i in range(smallest_n, biggest_n + 1):
        j = i
        while j <= len(token_stroke):
            token_n_gram.append(strokes2idx[token_stroke[j - i: j]])
            j += 1
    if len(token_n_gram) == 0:
        # 表示当前字的笔画小于3
        return [1]

    return token_n_gram


def get_dataloader(data_file_dir_, batch_size):
    datas = pd.read_csv(os.path.join(data_file_dir_, 'train.csv'), sep='\t')

    # 构建词典，并将文本数据转为id序列
    token2idx = {'<PAD>': 0, '<UNK>': 1}
    X = []
    sentences = []
    for text in datas.text.values:
        x = []
        sentence = []
        for token in text.strip().split(' '):
            try:
                token_id = token2idx[token]
            except KeyError:
                token_id = len(token2idx)
                token2idx[token] = token_id
            x.append(token_id)
            sentence.append(token)
        X.append(x)
        sentences.append(sentence)
    with open(os.path.join(data_file_dir_, 'token2idx.json'), 'w', encoding='utf-8') as writer:
        json.dump(token2idx, writer, ensure_ascii=False, indent=4)

    # 获取基于笔画的n-grams映射数据
    # 获取汉字的笔画映射字典
    stroke_path = os.path.join(data_file_dir_, 'stroke/strokes.txt')
    char2stroke = get_char2stroke(stroke_path)
    with open(os.path.join(data_file_dir_, 'stroke/char2stroke.json'), 'w', encoding='utf-8') as writer:
        json.dump(char2stroke, writer, ensure_ascii=False, indent=4)

    # 获取笔画n-gram到id的映射字典
    strokes2idx = get_stroke_idx()
    with open(os.path.join(data_file_dir_, 'stroke/strokes2idx.json'), 'w', encoding='utf-8') as writer:
        json.dump(strokes2idx, writer, ensure_ascii=False, indent=4)

    # 获取文本基于笔画的n-gram id序列
    X_token_stroke = []
    for text in sentences:
        new_text = []
        for token in text:
            token_n_gram = get_token_n_gram(token, char2stroke, strokes2idx, smallest_n=3, biggest_n=5)
            new_text.append(token_n_gram)
        X_token_stroke.append(new_text)

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(X, datas.label.values, test_size=0.2, random_state=42)
    x_stroke_train, x_stroke_test, _, _ = train_test_split(X_token_stroke, datas.label.values, test_size=0.2,
                                                           random_state=42)

    with open(os.path.join(data_file_dir_, 'label2idx.json'), 'r', encoding='utf-8') as reader:
        label2idx = json.load(reader)

    # 类别不均衡，计算每个类别的权重
    num_classes = len(label2idx)
    class_counts = datas.label.value_counts()
    weights = class_counts.sum() / (class_counts + 1)  # 加1防止类别为0异常
    weights = np.clip(weights, num_classes, num_classes * 1.1)
    weights = torch.softmax(torch.tensor(weights.values), dim=0)
    weights = torch.tensor(weights, dtype=torch.float32).view(-1)

    train_dataset = TextDataset(x_train, list(y_train), x_stroke_train)
    test_dataset = TextDataset(x_test, list(y_test), x_stroke_test)

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=TextDataset.my_collate_fn
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=batch_size * 2, collate_fn=TextDataset.my_collate_fn
    )

    return train_dataloader, test_dataloader, weights, num_classes, len(token2idx), len(strokes2idx)


if __name__ == '__main__':
    FILE_ROOT_DIR = os.path.dirname(__file__)
    data_file_dir = os.path.join(FILE_ROOT_DIR, '../../datas')
    test_path = os.path.join(FILE_ROOT_DIR, '../../test')
    os.makedirs(test_path, exist_ok=True)
    train_dataloader_, test_dataloader_, weights_, num_classes_, words_size, stroke_n_gram_counts = get_dataloader(
        data_file_dir_=data_file_dir, batch_size=128
    )
