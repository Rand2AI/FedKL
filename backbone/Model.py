# -*-coding:utf-8-*-

import torch
from backbone import *
from torchvision.models.resnet import *
from torchvision.models.vgg import *

def build_model(num_classes, config, act='relu'):
    if config['DEFENCE_NETWORK']['WITH_DEFENCE']:
        if config['NETWORK']['BACKBONE'] == 'lenet':
            net = LeNet_kl(channel=3, hideen=768, num_classes=num_classes,key_in_dim=config['DEFENCE_NETWORK']['KEY_LENGTH'])
        elif config['NETWORK']['BACKBONE'] == 'resnet':
            if config['NETWORK']['LAYER_NUMBER'] == 18:
                if config['DEFENCE_NETWORK']['ALL_LAYER']:
                    net = resnet18_kl_all(key_in_dim=config['DEFENCE_NETWORK']['KEY_LENGTH'], num_classes=num_classes, act=act)
                else:
                    net = resnet18_kl_single(key_in_dim=config['DEFENCE_NETWORK']['KEY_LENGTH'], num_classes=num_classes, act=act)
            elif config['NETWORK']['LAYER_NUMBER'] == 34:
                net = resnet34_kl_single(key_in_dim=config['DEFENCE_NETWORK']['KEY_LENGTH'], num_classes=num_classes, act=act)
            elif config['NETWORK']['LAYER_NUMBER'] == 20:
                    net = resnet20_kl(key_in_dim=config['DEFENCE_NETWORK']['KEY_LENGTH'], num_classes=num_classes, act=act)
            elif config['NETWORK']['LAYER_NUMBER'] == 32:
                    net = resnet32_kl(key_in_dim=config['DEFENCE_NETWORK']['KEY_LENGTH'], num_classes=num_classes, act=act)
        elif config['NETWORK']['BACKBONE'] == 'vgg16':
            net = vgg16_kl(key_in_dim=config['DEFENCE_NETWORK']['KEY_LENGTH'], num_classes=num_classes, act=act)
    else:
        if config['NETWORK']['BACKBONE'] == 'lenet':
            net = lenet(channel=3, hideen=768, num_classes=num_classes)
        elif config['NETWORK']['BACKBONE'] == 'vgg16':
            net = vgg16(num_classes=num_classes)
        elif config['NETWORK']['BACKBONE'] == 'resnet':
            if config['NETWORK']['LAYER_NUMBER'] == 18:
                net = resnet18(pretrained=False)
                num_ftrs = net.fc.in_features
                net.fc = torch.nn.Linear(num_ftrs, num_classes)
            elif config['NETWORK']['LAYER_NUMBER'] == 34:
                net = resnet34(pretrained=False)
                num_ftrs = net.fc.in_features
                net.fc = torch.nn.Linear(num_ftrs, num_classes)
            elif config['NETWORK']['LAYER_NUMBER'] == 20:
                    net = resnet20(num_classes=num_classes)
            elif config['NETWORK']['LAYER_NUMBER'] == 32:
                    net = resnet32(num_classes=num_classes)
            else:
                raise Exception("Wrong ResNet Layer Number.")
        else:
            raise Exception("Wrong Backbone Name.")
    return net

def build_leakage_model(net_name, key_in_dim, num_classes, with_kl, act='sigmoid'):
    if net_name == 'res18':
        if with_kl:
            net = resnet18_kl_single(key_in_dim=key_in_dim, num_classes=num_classes, act=act)
        else:
            net = resnet18_leak(num_classes=num_classes)
    elif net_name == 'res34':
        if with_kl:
            net = resnet34_kl_single(key_in_dim=key_in_dim, num_classes=num_classes, act=act)
        else:
            net = resnet34_leak(num_classes=num_classes)
    elif net_name == 'res20':
        if with_kl:
            net = resnet20_kl(key_in_dim=key_in_dim, num_classes=num_classes, act=act)
        else:
            net = resnet20(num_classes=num_classes, act=act)
    elif net_name == 'res32':
        if with_kl:
            net = resnet32_kl(key_in_dim=key_in_dim, num_classes=num_classes, act=act)
        else:
            net = resnet32(num_classes=num_classes, act=act)
    elif net_name == 'vgg16':
        if with_kl:
            net = vgg16_kl(key_in_dim=key_in_dim, num_classes=num_classes, act=act)
        else:
            net = vgg16_leak(num_classes=num_classes)
    elif net_name == 'lenet':
        if with_kl:
            net = LeNet_kl(channel=3, hideen=768, num_classes=num_classes,key_in_dim=key_in_dim)
        else:
            net = lenet(channel=3, hideen=768, num_classes=num_classes)
    else:
        net = None
        exit("Wrong network name")
    return net
