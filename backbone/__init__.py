# -*-coding:utf-8-*-
from backbone.LeNet import lenet
from backbone.ResNet_kl import resnet18_kl_all, resnet18_kl_single, resnet34_kl_single
from backbone.ResNet_cifar import resnet20, resnet32
from backbone.ResNet_cifar_kl import resnet20_kl, resnet32_kl
from backbone.LeNet_kl import LeNet_kl
from backbone.VGGNet_kl import vgg16_kl
from backbone.Leakage_ResNet import resnet18_leak, resnet34_leak
from backbone.Leakage_VGG import vgg16_leak
