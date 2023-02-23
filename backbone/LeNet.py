# -*-coding:utf-8-*-
import torch.nn as nn


class lenet(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(lenet, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            # nn.BatchNorm2d(12),
            nn.Sigmoid(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            # nn.BatchNorm2d(12),
            nn.Sigmoid(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            # nn.BatchNorm2d(12),
            nn.Sigmoid(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
