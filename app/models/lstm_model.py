import torch.nn as nn


class ForexLSTM(nn.Module):
    def __init__(self, n_features: int = 5):
        super().__init__()
        self.lstm = nn.LSTM(n_features, 32, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(self.drop(last)).sigmoid().squeeze(-1)
