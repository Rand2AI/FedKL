# -*- coding: utf-8 -*-
import os, datetime, sys
import numpy as np
from backbone.Model import build_model
from utils import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def Normal(config):
    with_defence = config['DEFENCE_NETWORK']['WITH_DEFENCE']
    epochs = config['TRAIN']['ROUNDS']
    batchsize = config['TRAIN']['BATCH_SIZE']
    # load dataset
    train_dataset, val_dataset, img_size, num_classes = gen_dataset(config['DATA']['TRAIN_DATA'],
                                                                    config['DATA']['IMG_SIZE'],
                                                                    config['DATA']['DATA_ROOT'])
    # data = ImagenetData(shards="imagenet2012-train-{000000..000146}.tar",
    #                     valshards="imagenet2012-val-{000000..000006}.tar",
    #                     batch_size=batchsize,
    #                     workers=4 * len(config['DEVICE']['DEVICE_GPUID']),
    #                     bucket="file:/lustrehome/home/scw1907/hans/Data/Object/ILSVRC/2012/shard/")
    # build model
    model = build_model(num_classes, config)
    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters.size())
    if config['DEVICE']['DEVICE_TOUSE'] == 'GPU':
        model.cuda()
        if len(config['DEVICE']['DEVICE_GPUID']) > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(len(config['DEVICE']['DEVICE_GPUID']))))
    if config['TRAIN']['FINETUNE']:
        checkpoint = torch.load(config['TRAIN']['WEIGHT_TOLOAD'])
        model.load_state_dict(checkpoint)
    train_loader = DataLoader(train_dataset,
                              batch_size=batchsize,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=batchsize,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)
    optimizer = set_optimizer(model.parameters(), config)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[config['TRAIN']['ROUNDS']//3, 2 * (config['TRAIN']['ROUNDS']//3)], gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[60, 120, 160],
                                                        gamma=0.2)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                     milestones=[150, 225],
    #                                                     gamma=0.1)
    # set path
    modelID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if config['NETWORK']['BACKBONE'] == 'resnet':
        model_name = f"{config['NETWORK']['BACKBONE']}{config['NETWORK']['LAYER_NUMBER']}"
    else:
        model_name = config['NETWORK']['BACKBONE']
    if 'lenet' in model_name:
        model_name = 'lenet'
    save_path = config['TRAIN']['SAVE_ROOT'] + f"{config['NAME']}/{config['METHODS']}/{config['METHODS']}-{model_name}-{config['DATA']['TRAIN_DATA']}-B{str(batchsize).zfill(3)}-{modelID}"
    if with_defence:
        save_path+='-kl'
        if config['DEFENCE_NETWORK']['ALL_LAYER']:
            save_path+='-all_layer'
        else:
            save_path+='-single_layer'
        save_path+=f"-key_len_{config['DEFENCE_NETWORK']['KEY_LENGTH']}"
        client_key = np.random.randint(0, 10000, config['DEFENCE_NETWORK']['KEY_LENGTH'])/10000
        if config['DEVICE']['DEVICE_TOUSE'] == 'GPU'and len(config['DEVICE']['DEVICE_GPUID']) > 1:
            client_key = torch.from_numpy(client_key)
            client_key = client_key.unsqueeze(0).repeat(len(config['DEVICE']['DEVICE_GPUID']), 1)
    else:
        client_key = None
    if not config['DEBUG']:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # sys.stdout = Logger(f'{save_path}/log.txt', sys.stdout)
        print(f"\n>>>>>>>>>>>>> {save_path}\n")
        if with_defence:
            torch.save(client_key, f'{save_path}/key.bin')
        # test_tfLogger = TFLogger(save_path + "/TFlog/Test/")
        # train_tfLogger = TFLogger(save_path + f"/TFlog/Train/")
        # save arguments to local
        args_json_path = save_path + "/args.json"
        save_args_as_json(config, args_json_path)
    best=0
    for epoch in range(epochs):
        print("-" * 50)
        print(f"[Epoch: {epoch + 1}/{epochs}]")
        # evaluation
        train_epoch_loss_avg, train_epoch_acc_avg = trainer(model, train_loader, optimizer, criterion, batchsize, client_key)
        if with_defence:
            val_epoch_loss_avg, val_epoch_acc_avg = tester(model, val_loader, criterion,batchsize , client_key, "")
        else:
            val_epoch_loss_avg, val_epoch_acc_avg = evaluator(model, val_loader, criterion, batchsize)
        lr_scheduler.step()
        # if not config['DEBUG']:
        #     train_tfLogger.scalar_summary(f'trn_loss', train_epoch_loss_avg, epoch)
        #     train_tfLogger.scalar_summary(f'trn_acc_avg', train_epoch_acc_avg, epoch)
        #     test_tfLogger.scalar_summary('tst_loss', val_epoch_loss_avg, epoch)
        #     test_tfLogger.scalar_summary('tst_acc', val_epoch_acc_avg, epoch)
        if val_epoch_acc_avg>best and not config['DEBUG']:
            best = val_epoch_acc_avg
            torch.save(model.state_dict(),
                       f'{save_path}/{modelID}-{config["METHODS"]}-epoch:{str(epoch).zfill(3)}-tst_loss:{round(val_epoch_loss_avg, 4)}-tst_acc:{round(val_epoch_acc_avg, 4)}-best.pth')
    # test_tfLogger.close()
    # train_tfLogger.close()
