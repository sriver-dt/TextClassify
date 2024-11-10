import torch.nn as nn


class FcNet(nn.Module):
    def __init__(self, in_features, hidden_size, num_classes):
        super(FcNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, num_classes),
        )

    def forward(self, x):
        return self.layers(x)
