import torch.nn as nn
from backbone.KL_layer_block import Defense_block
class LeNet_kl(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10, key_in_dim=128):
        super(LeNet_kl, self).__init__()
        self.block1 = Defense_block(key_in_dim=key_in_dim,
                                    in_channels=channel,
                                    out_channels=12,
                                    kernel_size=5,
                                    padding=5 // 2,
                                    stride=2)
        self.act1 = nn.Sigmoid()
        self.block2 = nn.Sequential(nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
                                    # nn.BatchNorm2d(12),
                                    nn.Sigmoid())
        self.block3 = nn.Sequential(nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
                                    # nn.BatchNorm2d(12),
                                    nn.Sigmoid())
        self.fc = nn.Linear(hideen, num_classes)

    def forward(self, x, key):
        output, key_g, key_b = self.block1(x, key, key)
        output = self.act1(output)
        output = self.block2(output)
        output = self.block3(output)
        out = output.view(output.size(0), -1)
        out = self.fc(out)
        return out