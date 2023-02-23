"""vgg in pytorch
[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn
from backbone.KL_layer_block import Defense_block

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):

    def __init__(self, features, act, num_class=100, key_in_dim=128):
        super().__init__()
        self.kl_layer = Defense_block(key_in_dim=key_in_dim,
                                        in_channels=3,
                                        out_channels=64,
                                        kernel_size=3,
                                        padding=1,
                                        stride=1)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        if act == 'relu':
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_class),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.Sigmoid(),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.Sigmoid(),
                nn.Dropout(),
                nn.Linear(4096, num_class),
            )
        self._initialize_weights()

    def forward(self, x, key):
        x  = self.kl_layer(x, key, key)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            try:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
            except:
                pass

def make_layers(cfg, act, batch_norm=False):
    layers = []

    input_channel = cfg.pop(0)
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]
        if act=='relu':
            layers += [nn.ReLU(inplace=True)]
        else:
            layers += [nn.Sigmoid()]
        input_channel = l

    return nn.Sequential(*layers)


def vgg16_kl(num_classes, key_in_dim, act='relu'):
    return VGG(make_layers(cfg['D'], act=act, batch_norm=True), num_class=num_classes, key_in_dim=key_in_dim, act=act)

# def vgg11_bn(num_classes):
#     return VGG(make_layers(cfg['A'], batch_norm=True), num_class=num_classes)
#
#
# def vgg13_bn(num_classes):
#     return VGG(make_layers(cfg['B'], batch_norm=True), num_class=num_classes)
#
#
# def vgg16_bn(num_classes):
#     return VGG(make_layers(cfg['D'], batch_norm=True), num_class=num_classes)
#
#
# def vgg19_bn(num_classes):
#     return VGG(make_layers(cfg['E'], batch_norm=True), num_class=num_classes)
