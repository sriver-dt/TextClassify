import torch
import torch.nn as nn


class StrokeNet(nn.Module):
    def __init__(self, vocab_size, emb_size, stroke_n_gram_counts, hidden_size, num_classes):
        super(StrokeNet, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.stroke_emb = nn.Embedding(stroke_n_gram_counts, emb_size)
        self.merge_linear = nn.Linear(2 * emb_size, emb_size)
        self.output = nn.Sequential(
            nn.Linear(emb_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x, mask=None, x_stroke=None, stroke_mask=None):
        # token处理
        x = self.emb(x)  # [N, T, E]
        # 合并
        if mask is None:
            x = torch.mean(x, dim=1)  # [N, E]
        else:
            mask = mask.to(dtype=x.dtype, device=x.device)
            x = x * mask[..., None]  # [N, T, E]
            x = torch.sum(x, dim=1) / torch.sum(mask, dim=1, keepdim=True)  # [N, T, E] -> [N, E]

        # stroke处理
        if x_stroke is not None:
            x_stroke = self.stroke_emb(x_stroke)  # [N, T, C, E]
            stroke_mask = stroke_mask.to(dtype=x_stroke.dtype, device=x_stroke.device)  # [N, T, C]
            stroke_mask = stroke_mask[..., None]  # # [N, T, C, 1]
            x_stroke = x_stroke * stroke_mask  # [N, T, C, E]
            x_stroke = x_stroke.sum(-2).sum(-2) / stroke_mask.sum(-2).sum(-2)  # [N, E]

            # 合并token和stroke
            x = torch.concatenate((x, x_stroke), dim=1)  # [N, 2*E]
            x = self.merge_linear(x)

        return self.output(x)
