import torch
import torch.nn as nn


class EmbNet(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_classes):
        super(EmbNet, self).__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, num_classes),
        )

    def forward(self, x, mask=None):
        x = self.emb(x)     # [N, T, E]
        # 合并
        if mask is None:
            x = torch.mean(x, dim=1)
        else:
            mask = mask.to(dtype=x.dtype, device=x.device)
            x = x * mask[..., None]     # [N, T, E]
            x = torch.sum(x, dim=1) / torch.sum(mask, dim=1, keepdim=True)    # [N, T, E] -> [N, E]
        return self.layers(x)
