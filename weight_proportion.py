import os
from backbone import *
from utils import get_config

config = get_config(os.path.dirname(os.path.realpath(__file__)))

def count_parameters(model):
    model_dict = model.state_dict()
    total = 0
    kl = 0
    for k, v in model_dict.items():
        total+= v.nelement()
        if 'scale' in k or 'shift' in k:
            kl += v.nelement()
    return total, kl, kl/total


def main():
    num_classes=100
    act = 'relu'
    LeNet = LeNet_kl(channel=3, hideen=768, num_classes=num_classes, key_in_dim=config['DEFENCE_NETWORK']['KEY_LENGTH'])
    resnet20 = resnet20_kl(key_in_dim=config['DEFENCE_NETWORK']['KEY_LENGTH'], num_classes=num_classes, act=act)
    resnet32 = resnet32_kl(key_in_dim=config['DEFENCE_NETWORK']['KEY_LENGTH'], num_classes=num_classes, act=act)
    resnet18 = resnet18_kl_single(key_in_dim=config['DEFENCE_NETWORK']['KEY_LENGTH'], num_classes=num_classes, act=act)
    resnet34 = resnet34_kl_single(key_in_dim=config['DEFENCE_NETWORK']['KEY_LENGTH'], num_classes=num_classes, act=act)
    vgg16 = vgg16_kl(key_in_dim=config['DEFENCE_NETWORK']['KEY_LENGTH'], num_classes=num_classes, act=act)

    LeNet_total, LeNet_l, LeNet_proprotion = count_parameters(LeNet)
    resnet20_total, resnet20_l, resnet20_proprotion = count_parameters(resnet20)
    resnet32_total, resnet32_l, resnet32_proprotion = count_parameters(resnet32)
    resnet18_total, resnet18_l, resnet18_proprotion = count_parameters(resnet18)
    resnet34_total, resnet34_l, resnet34_proprotion = count_parameters(resnet34)
    vgg16_total, vgg16_l, vgg16_proprotion = count_parameters(vgg16)
    print('Done')

if __name__ == '__main__':
    main()