import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, emb_weight, num_classes):
        super(Net, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_size, _weight=emb_weight)
        self.layers = nn.Sequential(
            nn.Linear(emb_size, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x, mask=None):
        x = self.emb(x)  # [N, T, E]
        # 合并
        if mask is None:
            x = torch.mean(x, dim=1)
        else:
            mask = mask.to(dtype=x.dtype, device=x.device)
            x = x * mask[..., None]  # [N, T, E]
            x = torch.sum(x, dim=1) / torch.sum(mask, dim=1, keepdim=True)  # [N, T, E] -> [N, E]
        return self.layers(x)
