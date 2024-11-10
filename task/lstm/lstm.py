import torch
import torch.nn as nn
from torch.nn.utils import rnn


class LstmNet(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_class):
        super(LstmNet, self).__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=4,  # lstm 层数
            batch_first=True,  # 是否批次在前，默认False，即[T, N, E]
            bidirectional=True  # 是否双向，默认False
        )
        self.layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_class),
        )

    def forward(self, x, mask=None):
        x = self.emb(x)  # [N, T, E]

        # lstm
        # lstm_output: [N,T,2*hidden_size] 对应每个token的输出向量
        # lstm_state：lstm的状态信息
        if mask is not None:
            # 将序列根据mask打包成可以直接输入lstm的数据
            x = rnn.pack_padded_sequence(
                input=x,
                lengths=torch.sum(mask, dim=1).cpu(),  # 每个序列的长度
                batch_first=True,
                enforce_sorted=False,  # 默认为True, 即给定序列长度是降序排列
            )

        lstm_output, lstm_state = self.lstm(x)

        if mask is not None:
            lstm_output, _ = rnn.pad_packed_sequence(lstm_output, batch_first=True)

        x = lstm_output  # [N, T, 2*hidden_size]
        # 合并
        if mask is None:
            x = torch.mean(x, dim=1)
        else:
            mask = mask.to(dtype=x.dtype, device=x.device)
            x = x * mask[..., None]  # [N, T, 2*hidden_size]
            # [N, T, 2*hidden_size] -> [N, 2*hidden_size]
            x = torch.sum(x, dim=1) / torch.sum(mask, dim=1, keepdim=True)
        return self.layers(x)
