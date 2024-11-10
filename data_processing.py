import json

import jieba
import pandas as pd

train_datas = pd.read_csv('datas/meta_data/train.csv', sep='\t', header=None)
train_datas.columns = ['text', 'label']

# 构造label序号映射
labels = set()
for label in train_datas.label:
    labels.add(label)
label2idx = {label: i for i, label in enumerate(labels)}
for i, label in enumerate(train_datas.label):
    train_datas.iloc[i, 1] = label2idx[label]
print(label2idx)
with open('./datas/label2idx.json', mode='w', encoding='utf-8') as file:
    json.dump(label2idx, file, ensure_ascii=False, indent=4)

# 构造词表
# jieba分词
tokens = set()
X = []
for text in train_datas.text:
    token_lst = list(jieba.cut(text.strip()))
    # 按照分词结果重新构建数据
    X.append(' '.join(token_lst))
    tokens.update(token_lst)
train_data = pd.DataFrame([X, train_datas.label]).T
train_data.columns = ['text', 'label']
train_data.to_csv('./datas/train.csv', sep='\t', encoding='utf-8', index=False)
