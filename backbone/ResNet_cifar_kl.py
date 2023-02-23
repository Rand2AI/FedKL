'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch.nn as nn
import torch.nn.functional as F

from backbone.KL_layer_block import Defense_block

__all__ = ['ResNet', 'resnet20_kl', 'resnet32_kl']

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock_with_defence(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_with_defence, self).__init__()
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)

        self.conv1 = Defense_block(in_planes, in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = Defense_block(planes, planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, inputs):
        x, key_g, key_b = inputs
        out, key_g, key_b = self.conv1(x, key_g, key_b)
        out = F.relu(out)
        out, key_g, key_b = self.conv2(out, key_g, key_b)
        out += self.shortcut(x)
        out = F.relu(out)
        return out, key_g, key_b

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, act='relu'):
        super().__init__()
        self.act=act
        # residual function
        if act == 'relu':
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

    def forward(self, inputs):
        x, key_g, key_b = inputs
        if self.act == 'relu':
            return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x)), key_g, key_b
        else:
            return nn.Sigmoid()(self.residual_function(x) + self.shortcut(x)), key_g, key_b

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, key_in_dim, act, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(16)
        self.conv1 = Defense_block(key_in_dim, 3, 16, kernel_size=3, stride=1, padding=1, bias=False, act=act)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, key):
        out, key_g, key_b = self.conv1(x, key, key)
        out = F.relu(out)
        out, key_g, key_b = self.layer1((out, key_g, key_b))
        out, key_g, key_b = self.layer2((out, key_g, key_b))
        out, key_g, key_b = self.layer3((out, key_g, key_b))
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet20_kl(key_in_dim, num_classes, act='relu'):
    return ResNet(BasicBlock, [3, 3, 3], key_in_dim, num_classes=num_classes, act=act)


def resnet32_kl(key_in_dim, num_classes, act='relu'):
    return ResNet(BasicBlock, [5, 5, 5], key_in_dim, num_classes=num_classes, act=act)

def resnet20_kl_all(key_in_dim, num_classes, act='relu'):
    return ResNet(BasicBlock_with_defence, [3, 3, 3], key_in_dim, num_classes=num_classes, act=act)


def resnet32_kl_all(key_in_dim, num_classes, act='relu'):
    return ResNet(BasicBlock_with_defence, [5, 5, 5], key_in_dim, num_classes=num_classes, act=act)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()