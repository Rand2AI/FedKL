# -*-coding:utf-8-*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np
import PIL.Image as Image
from utils.WassersteinDistance import *
import os

def split_gradient(net, with_lock_layer, dy_dx):
    idx = 0
    new_dy_dx = []
    for k in net.state_dict().keys():
        if 'weight' in k or 'bias' in k:
            if with_lock_layer:
                new_dy_dx.append(dy_dx[idx])
            else:
                if 'scale_fc' in k or 'shift_fc' in k:
                    pass
                else:
                    new_dy_dx.append(dy_dx[idx])
            idx += 1
    return new_dy_dx

def flatten_gradients(dy_dx):
    flatten_dy_dx = None
    for layer_g in dy_dx:
        if flatten_dy_dx is None:
            flatten_dy_dx = torch.flatten(layer_g)
        else:
            flatten_dy_dx = torch.cat((flatten_dy_dx, torch.flatten(layer_g)))
    return flatten_dy_dx

def GRNN_gen_dataset(dataset, shape_img):
    class Dataset_from_Image(Dataset):
        def __init__(self, imgs, labs, transform=None):
            self.imgs = imgs  # img paths
            self.labs = labs  # labs is ndarray
            self.transform = transform
            del imgs, labs

        def __len__(self):
            return self.labs.shape[0]

        def __getitem__(self, idx):
            lab = self.labs[idx]
            img = Image.open(self.imgs[idx])
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = self.transform(img)
            return img, lab

    def lfw_dataset(lfw_path, shape_img, num_classes):
        images_all = []
        index_all = []
        folders = os.listdir(lfw_path)
        for foldidx, fold in enumerate(folders):
            if foldidx+1==num_classes: break
            if os.path.isdir(os.path.join(lfw_path, fold)):
                files = os.listdir(os.path.join(lfw_path, fold))
                for f in files:
                    if len(f) > 4:
                        images_all.append(os.path.join(lfw_path, fold, f))
                        index_all.append(foldidx)
        transform = transforms.Compose([transforms.Resize(shape_img),
                                 transforms.ToTensor()])
        dst = Dataset_from_Image(images_all, np.asarray(index_all, dtype=int), transform=transform)
        return dst
    data_path = "./Data/"
    if dataset == 'mnist':
        num_classes = 10
        channel = 3
        hidden = 768
        tt = transforms.Compose([transforms.Resize(shape_img),
                                 transforms.Grayscale(num_output_channels=channel),
                                 transforms.ToTensor()
                                 ])
        dst = datasets.MNIST(data_path + "/mnist", download=True, transform=tt)
    elif dataset == 'cifar100':
        num_classes = 100
        channel = 3
        hidden = 768
        tt = transforms.Compose([transforms.Resize(shape_img),
                                 transforms.ToTensor()])
        dst = datasets.CIFAR100(data_path + "/cifar100", download=True, transform=tt)
    elif dataset == 'cifar10':
        num_classes = 10
        channel = 3
        hidden = 768
        tt = transforms.Compose([transforms.Resize(shape_img),
                                 transforms.ToTensor()])
        dst = datasets.CIFAR10(data_path + "/cifar10", download=True, transform=tt)
    elif dataset == 'ilsvrc':
        hidden = 768
        num_classes = 1000
        channel = 3
        dst = lfw_dataset('./Data/Object/ILSVRC/2012/train/', shape_img, num_classes)
    elif dataset == 'imagenet':
        hidden = 768
        num_classes = 1000
        channel = 3
        tt = transforms.Compose([transforms.Resize(shape_img),
                                 transforms.ToTensor()])
        dst = datasets.ImageNet('./Data/Object/ILSVRC/2012/', transform=tt)
    else:
        dst,  num_classes, channel, hidden = None, None, None, None
        exit('unknown dataset')
    return dst, num_classes, channel,hidden

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    @staticmethod
    def _tensor_size(t):
        return t.size()[1]*t.size()[2]*t.size()[3]

def loss_f(loss_name, flatten_fake_g, flatten_true_g, device1, Gout):
    if loss_name == "mse":
        mse_loss = nn.MSELoss()
        grad_diff = mse_loss(flatten_fake_g, flatten_true_g)
    elif loss_name == "l1":
        grad_diff = (abs(flatten_fake_g - flatten_true_g)).sum()
    elif loss_name == 'tv':
        tv_loss = TVLoss()
        grad_diff = 1e-6 * tv_loss(Gout)
    elif loss_name == "l2":
        grad_diff = ((flatten_fake_g - flatten_true_g) ** 2).sum()
        # grad_diff = torch.sqrt(((flatten_fake_g - flatten_true_g) ** 2).sum())
    elif loss_name == "wd":
        grad_diff = wasserstein_distance(flatten_fake_g.view(1, -1), flatten_true_g.view(1, -1), device=f"cuda:{device1}")
    elif loss_name == "swd":
        grad_diff = sliced_wasserstein_distance(flatten_fake_g.view(1, -1), flatten_true_g.view(1, -1),
                                                device=f"cuda:{device1}")
    elif loss_name == "gswd":
        grad_diff = generalized_sliced_wasserstein_distance(flatten_fake_g.view(1, -1), flatten_true_g.view(1, -1),
                                                            device=f"cuda:{device1}")
    elif loss_name == "mswd":
        grad_diff = max_sliced_wasserstein_distance(flatten_fake_g.view(1, -1), flatten_true_g.view(1, -1),
                                                    device=f"cuda:{device1}")
    elif loss_name == "dswd":
        grad_diff = distributional_sliced_wasserstein_distance(flatten_fake_g.view(1, -1), flatten_true_g.view(1, -1),
                                                               device=f"cuda:{device1}")
    elif loss_name == "mgswd":
        grad_diff = max_generalized_sliced_wasserstein_distance(flatten_fake_g.view(1, -1), flatten_true_g.view(1, -1),
                                                                device=f"cuda:{device1}")
    elif loss_name == "dgswd":
        grad_diff = distributional_generalized_sliced_wasserstein_distance(flatten_fake_g.view(1, -1),
                                                                           flatten_true_g.view(1, -1),
                                                                           device=f"cuda:{device1}")
    elif loss_name == "csd":
        grad_diff = cosine_sum_distance_torch(flatten_fake_g.view(1, -1), flatten_true_g.view(1, -1))
    else:
        raise Exception("Wrong loss name.")
    return grad_diff.requires_grad_(True)

def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * torch.log_softmax(pred, dim=-1), 1))