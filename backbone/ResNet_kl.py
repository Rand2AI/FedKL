# -*-coding:utf-8-*-
import torch.nn as nn
from backbone.KL_layer_block import Defense_block

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, act='relu'):
        super().__init__()
        self.act=act
        # residual function
        if act=='relu':
            self.residual_function = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        else:
            self.residual_function = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Sigmoid(),
                nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    # def forward(self, inputs):
    #     x, key_g, key_b = inputs
    #     return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x)), key_g, key_b
    def forward(self, inputs):
        x = inputs
        if self.act=='relu':
            return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
        else:
            return nn.Sigmoid()(self.residual_function(x) + self.shortcut(x))

class BasicBlock_with_defence(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, act='relu'):
        super().__init__()
        self.l1 = Defense_block(in_channels, in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        if act=='relu':
            self.act1 = nn.ReLU(inplace=True)
            self.act2 = nn.ReLU(inplace=True)
        elif act=='sigmoid':
            self.act1 = nn.Sigmoid()
            self.act2 = nn.Sigmoid()
        self.l2 = Defense_block(out_channels, out_channels, out_channels, kernel_size=3, padding=1, bias=False)

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels , kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, inputs):
        x, key_g, key_b = inputs
        output, key_g, key_b = self.l1(x, key_g, key_b)
        output = self.act1(output)
        output, key_g, key_b = self.l2(output, key_g, key_b)
        output = output+self.shortcut(x)
        return self.act2(output), key_g, key_b

class ResNet(nn.Module):

    def __init__(self, block, num_block, key_in_dim, act, num_classes=100):
        super().__init__()

        self.in_channels = 64
        self.act = act

        # self.conv1 = nn.Sequential(
            # nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True))
        # self.conv1 = Defense_block(key_in_dim, 3, 64, kernel_size=3, padding=1, bias=False)
        self.conv1 = Defense_block(key_in_dim, 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if act=='relu':
            self.act1 =  nn.ReLU(inplace=True)
        else:
            self.act1 = nn.Sigmoid()

        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, self.act))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, key):
        output, key_g, key_b = self.conv1(x, key, key)
        output = self.act1(output)
        # output, key_g, key_b = self.conv2_x((output, key_g, key_b))
        # output, key_g, key_b = self.conv3_x((output, key_g, key_b))
        # output, key_g, key_b = self.conv4_x((output, key_g, key_b))
        # output, key_g, key_b = self.conv5_x((output, key_g, key_b))
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


def resnet18_kl_all(key_in_dim, num_classes, act='sigmoid'):
    return ResNet(BasicBlock_with_defence, [2, 2, 2, 2], key_in_dim, num_classes=num_classes, act=act)

def resnet18_kl_single(key_in_dim, num_classes, act='sigmoid'):
    return ResNet(BasicBlock, [2, 2, 2, 2], key_in_dim, num_classes=num_classes, act=act)

def resnet34_kl_single(key_in_dim, num_classes, act='sigmoid'):
    return ResNet(BasicBlock, [3, 4, 6, 3], key_in_dim, num_classes=num_classes, act=act)