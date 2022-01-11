# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def predict(self, x):
        self.eval()
        out = self.forward(x)
        predict_labels = torch.max(out, 1)[1]
        self.train(mode=True)
        return predict_labels


class DPCNN(BasicModule):
    def __init__(self, config):
        super(DPCNN, self).__init__()
        self.config = config
        if self.config.is_embedding:
            self.embedding = nn.Embedding(self.config.word_num, self.config.word_embedding_dimension)

        self.conv_region_embedding = nn.Sequential(
            nn.Conv1d(self.config.word_embedding_dimension, self.config.channel_size, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(self.config.channel_size),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.conv = nn.Sequential(
            # nn.BatchNorm1d(self.config.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.config.channel_size, self.config.channel_size, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(self.config.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.config.channel_size, self.config.channel_size, kernel_size=3, stride=1, padding=1)
        )

        self.pooling = nn.Sequential(
            nn.ConstantPad1d(padding=(0, 1), value=0),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv_block = nn.Sequential(
            # nn.BatchNorm1d(self.config.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.config.channel_size, self.config.channel_size, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(self.config.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.config.channel_size, self.config.channel_size, kernel_size=3, stride=1, padding=1)
        )

        self.fc = nn.Linear(2 * self.config.channel_size, self.config.num_classes)

    def forward(self, x):
        if self.config.is_embedding:
            x = self.embedding(x)  # [batch_size, length, embedding_dimension]
        x = x.permute(0, 2, 1)
        x = self.conv_region_embedding(x)  # [batch_size, channel_size, length]
        x = self.conv(x)                   # [batch_size, channel_size, length]

        while x.size()[-1] > 2:
            x = self._block(x)

        x = x.view(-1, 2 * self.config.channel_size)
        x = self.fc(x)

        return x

    def _block(self, x):
        # Pooling
        px = self.pooling(x)

        # Convolution
        x = self.conv_block(px)

        # Short Cut
        x = x + px

        return x
