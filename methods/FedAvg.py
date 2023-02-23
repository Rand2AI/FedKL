# -*- coding: utf-8 -*-
import numpy as np
import os, datetime, copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backbone.Model import build_model
from utils import *


def FedAvg(config):
    torch.manual_seed(999)
    rounds = config['TRAIN']['ROUNDS']
    batchsize = config['TRAIN']['BATCH_SIZE']
    config['DEFENCE_NETWORK']['WITH_DEFENCE'] = False

    train_dataset, test_dataset, img_size, num_classes = gen_dataset(config['DATA']['TRAIN_DATA'],
                                                                     config['DATA']['IMG_SIZE'],
                                                                     config['DATA']['DATA_ROOT'])
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False, num_workers=4*len(config['DEVICE']['DEVICE_GPUID']), pin_memory=True)
    # select client
    selected_client_num = max(int(config['FED']['FRACTION'] * config['FED']['CLIENTS_NUM']), 1)
    print(f"{selected_client_num} of {config['FED']['CLIENTS_NUM']} clients are selected.")
    idxs_client = np.random.choice(range(config['FED']['CLIENTS_NUM']), selected_client_num, replace=False)

    # IID or Non-IID
    if config['DATA']['IS_IID']:
        dict_users = split_iid_data(train_dataset, config['FED']['CLIENTS_NUM'])
    else:
        dict_users = split_noniid_data(train_dataset, config['FED']['CLIENTS_NUM'])
    model_global = build_model(num_classes, config)
    if config['DEVICE']['DEVICE_TOUSE'] == 'GPU':
        model_global.cuda()
        if len(config['DEVICE']['DEVICE_GPUID']) > 1:
            model_global = torch.nn.DataParallel(model_global, device_ids=list(range(len(config['DEVICE']['DEVICE_GPUID']))))
    if config['TRAIN']['FINETUNE']:
        checkpoint = torch.load(config['TRAIN']['WEIGHT_TOLOAD'])
        model_global.load_state_dict(checkpoint)

    modelID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if config['NETWORK']['BACKBONE'] == 'resnet':
        model_name = f"{config['NETWORK']['BACKBONE']}{config['NETWORK']['LAYER_NUMBER']}"
    else:
        model_name = config['NETWORK']['BACKBONE']
    if 'lenet' in model_name:
        model_name = 'lenet'
    save_path = f"./Data/Models/{config['NAME']}/{config['METHODS']}/{config['METHODS']}-{model_name}-{config['DATA']['TRAIN_DATA']}-B{str(batchsize).zfill(3)}-{modelID}"
    if not config['DEBUG']:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # sys.stdout = Logger(f'{save_path}/log.txt', sys.stdout)
        print(f"\n>>>>>>>>>>>>> {save_path}\n")
        # test_tfLogger = TFLogger(save_path + "/TFlog/Test/")
        # train_tfLogger = TFLogger(save_path + f"/TFlog/Train/")
        # save arguments to local
        args_json_path = save_path + "/args.json"
        save_args_as_json(config, args_json_path)

        # client model path
        model_path = {}
        for idx in idxs_client:
            model_path[idx] = save_path + f"/Model/Client_{idx}/"
            if not os.path.exists(model_path[idx]):
                os.makedirs(model_path[idx])

    locals = {idx: local_update(config=config,
                                client_idx=idx,
                                dataset=train_dataset,
                                data_idxs=dict_users[idx],
                                model=copy.deepcopy(model_global),
                                test_loader=test_loader,
                                client_keys=None) for idx in idxs_client}
    weight_global = model_global.state_dict()
    best = 0
    for rd in range(rounds):
        print('\n')
        print("-" * 100)
        print(f"[Round: {rd}/{rounds}]")
        # train
        loss_locals = []
        acc_locals = []
        weight_locals = []
        for idx in idxs_client:
            weight_local, loss_local, acc_local = locals[idx].train(weight_global)
            # if not config['DEBUG']:
            #     torch.save(weight_local, model_path[idx] + f"/client:{idx}-epoch:{str(rd).zfill(3)}-trn_loss:{np.round(loss_local, 4)}-trn_acc:{np.round(acc_local, 4)}-{modelID}.pth")
            weight_locals.append(weight_local)
            loss_locals.append(loss_local)
            acc_locals.append(acc_local)
            # log
            # if not config['DEBUG']:
            #     train_tfLogger.scalar_summary(f'trn_loss_{idx}', loss_local, rd)
            #     train_tfLogger.scalar_summary(f'trn_acc_{idx}', acc_local, rd)
        weight_global = fedavg_weight(weight_locals)
        model_global.load_state_dict(weight_global)

        # test
        test_loss_avg, test_acc_avg = evaluator(model_global, test_loader, nn.CrossEntropyLoss(), batchsize)
        # if not config['DEBUG']:
        #     train_tfLogger.scalar_summary(f'trn_loss_avg', np.mean(loss_locals), rd)
        #     train_tfLogger.scalar_summary(f'trn_acc_avg', np.mean(acc_locals), rd)
        #     test_tfLogger.scalar_summary('tst_loss', test_loss_avg, rd)
        #     test_tfLogger.scalar_summary('tst_acc', test_acc_avg, rd)
        print(save_path)
        print(
            f"Round {rd}\nLocal loss: {np.mean(loss_locals)}, Local Acc: {np.mean(acc_locals)}\nTest  Loss: {test_loss_avg}, Test  Acc: {test_acc_avg}")
        if np.mean(test_acc_avg)>best and not config['DEBUG']:
            best = np.mean(test_acc_avg)
            torch.save(model_global.state_dict(),
                       f'{save_path}/{modelID}-{config["METHODS"]}-round:{str(rd).zfill(3)}-tst_loss:{np.round(np.mean(test_loss_avg), 4)}-tst_acc:{np.round(np.mean(test_acc_avg), 4)}-best.pth')
    # test_tfLogger.close()
    # train_tfLogger.close()
