import torch.nn as nn

class Defense_block(nn.Module):
    def __init__(self,key_in_dim, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', act='relu'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                 padding=padding, dilation=dilation, groups=groups,bias=bias, padding_mode=padding_mode)
        self.norm = nn.BatchNorm2d(num_features=out_channels, affine=False)
        self.scale_fc = nn.Linear(key_in_dim, out_features=out_channels)
        self.shift_fc = nn.Linear(key_in_dim, out_features=out_channels)
        if act=='relu':
            self.scale_act = nn.ReLU(inplace=True)
            self.shift_act = nn.ReLU(inplace=True)
        elif act=='sigmoid':
            self.scale_act = nn.Sigmoid()
            self.shift_act = nn.Sigmoid()

    def forward(self, x, key_g, key_b):
        x = self.conv(x)
        x = self.norm(x)
        gamma = self.scale_act(self.scale_fc(key_g.squeeze())).view(1,-1,1,1)
        beta = self.shift_act(self.shift_fc(key_b.squeeze())).view(1,-1,1,1)
        x = gamma * x + beta
        return x, gamma, beta