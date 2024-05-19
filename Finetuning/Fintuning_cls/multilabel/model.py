import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(
        self,
        embed_size=128,
        out=2,
    ):
        super(EsmClassifier, self).__init__()

        self.fc1 = nn.Linear(embed_size, embed_size)
        self.fc2 = nn.Linear(embed_size, out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src):
        x = self.fc1(src)
        out = self.fc2(x)
        out = self.sigmoid(out)
        return out



