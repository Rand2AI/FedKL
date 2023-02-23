# -*-coding:utf-8-*-
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms
import os
import numpy as np
import PIL.Image as Image
# import webdataset as wds
import torch
# import pytorch_lightning as pl

def gen_dataset(dataset, shape_img, data_path):
    if dataset == 'mnist':
        num_classes = 10
        tt = transforms.Compose([transforms.Resize(shape_img),
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.ToTensor()
                                 ])
        trainset = datasets.MNIST(root=data_path+'mnist', train=True, download=True, transform=tt)
        testset = datasets.MNIST(root=data_path+'mnist', train=False, download=True, transform=tt)
    elif dataset == 'cifar100':
        mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
        std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]
        num_classes = 100
        tt_train = transforms.Compose([transforms.Resize(shape_img),
                                       transforms.RandomCrop(shape_img[0], padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.Resize(shape_img),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=mean, std=std), ])
        tt_tst = transforms.Compose([transforms.Resize(shape_img),
                                     transforms.Resize(shape_img),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std), ])
        trainset = datasets.CIFAR100(root=data_path+'cifar100', train=True, download=True, transform=tt_train)
        testset = datasets.CIFAR100(root=data_path+'cifar100', train=False, download=True, transform=tt_tst)
    elif dataset == 'cifar10':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        num_classes = 10
        norm = transforms.Normalize(mean=mean, std=std)
        tt_train = transforms.Compose([transforms.Resize(shape_img),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(shape_img[0], padding=4),
                                      transforms.Resize(shape_img),
                                      transforms.ToTensor(),
                                      norm, ])
        tt_tst = transforms.Compose([transforms.Resize(shape_img),
                                     transforms.ToTensor(),
                                     norm, ])
        trainset = datasets.CIFAR10(root=data_path+'cifar10', train=True, download=True, transform=tt_train)
        testset = datasets.CIFAR10(root=data_path+'cifar10', train=False, download=True, transform=tt_tst)
    elif dataset == 'ilsvrc2012':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        num_classes = 1000

        root=data_path+f'/Object/ILSVRC/2012/'
        tt_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                       transforms.RandomRotation(15),
                                       # transforms.RandomCrop(shape_img[0]),
                                       transforms.Resize(shape_img),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=mean, std=std)])
        tt_tst = transforms.Compose([transforms.Resize(shape_img),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std)])
        trainset = datasets.ImageNet(root=root, split='train',transform=tt_train, download=False)
        testset = datasets.ImageNet(root=root, split='val', transform=tt_tst, download=False)

        # tt = transforms.Compose([transforms.Resize(shape_img),
        #                          transforms.ToTensor(),
        #                          transforms.Normalize(mean=mean, std=std), ])
        # dst = datasets.ImageNet('/home/hans/WorkSpace/Data/Object/ILSVRC/2012/', transform=tt)
        #
        # train_size = int(0.8 * len(dst))
        # test_size = len(dst) - train_size
        # trainset, testset = random_split(dst, [train_size, test_size])
    else:
        exit('unknown dataset')

    return trainset, testset, shape_img, num_classes

def identity(x):
    return x

def split_iid_data(dataset, num_users):
    """
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def split_noniid_data(dataset, num_users):
    """
    Sample non-I.I.D client data
    :param dataset:
    :param num_users:
    :return:
    """
    num_shard_to_choose = 2
    num_shards = int(num_users * num_shard_to_choose)
    num_imgs = len(dataset) // num_shards
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    try:
        all_labels = np.array(dataset.dataset.targets)
    except AttributeError:
        all_labels = np.array(dataset.dataset.labs)
    labels = all_labels[idxs]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, num_shard_to_choose, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def untar_ILSVRC2012():
    import tarfile
    from tqdm import tqdm
    train_path = "/home/hans/WorkSpace/Data/Object/ILSVRC/2012/Train/"

    path = train_path
    root, _, files = next(os.walk(path))
    process_bar = tqdm(files, total=len(files), ncols=120)
    for file in process_bar:
        file_name = os.path.join(root, file)
        tar = tarfile.open(file_name)
        names = tar.getnames()

        file_name = os.path.basename(file_name)
        extract_dir = os.path.join(train_path, file_name.split('.')[0])
        # create folder if nessessary
        if os.path.isdir(extract_dir):
            pass
        else:
            os.mkdir(extract_dir)
        for idx, name in enumerate(names):
            tar.extract(name, extract_dir)
            process_bar.set_postfix(file=file,
                                    progress=f"{idx + 1}/{len(names)}")
        tar.close()


if __name__ == '__main__':
    untar_ILSVRC2012()
