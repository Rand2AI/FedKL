# -*- coding: utf-8 -*-
import numpy as np
import os, datetime, copy, random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backbone.Model import build_model
from utils import *


def FedKL(config):
    rounds = config['TRAIN']['ROUNDS']
    batchsize = config['TRAIN']['BATCH_SIZE']
    config['DEFENCE_NETWORK']['WITH_DEFENCE'] = True

    train_dataset, test_dataset, img_size, num_classes = gen_dataset(config['DATA']['TRAIN_DATA'],
                                                                     config['DATA']['IMG_SIZE'],
                                                                     config['DATA']['DATA_ROOT'])
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)
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
    print(save_path)
    weight_global = model_global.state_dict()
    initial_key_layer = {k:v for k,v in weight_global.items() if 'scale_fc' in k or 'shift_fc' in k}
    # client_keys = {x: np.random.random_sample(config['DEFENCE_NETWORK']['KEY_LENGTH']) for x in idxs_client}
    # client_keys = {0: np.random.random_sample(config['DEFENCE_NETWORK']['KEY_LENGTH']),
    #                1: np.random.randint(0, 1000, config['DEFENCE_NETWORK']['KEY_LENGTH']),
    #                2: np.random.randint(10, 10000, config['DEFENCE_NETWORK']['KEY_LENGTH'])
    #                }
    # client_keys = {0: np.random.randint(0, 10000, config['DEFENCE_NETWORK']['KEY_LENGTH'])/10000,
    #                1: np.random.randint(0, 10000, config['DEFENCE_NETWORK']['KEY_LENGTH'])/10000,
    #                2: np.random.randint(0, 10000, config['DEFENCE_NETWORK']['KEY_LENGTH'])/10000
    #                }
    client_keys = {0: np.array([random.random() for _ in range(config['DEFENCE_NETWORK']['KEY_LENGTH'])]),
                   1: np.array([random.random() for _ in range(config['DEFENCE_NETWORK']['KEY_LENGTH'])]),
                   2: np.array([random.random() for _ in range(config['DEFENCE_NETWORK']['KEY_LENGTH'])])
                   }
    locals = {idx:local_update(config=config,
                               client_idx=idx,
                               dataset=train_dataset,
                               data_idxs=dict_users[idx],
                               client_keys = client_keys,
                               model=copy.deepcopy(model_global),
                               test_loader=test_loader) for idx in idxs_client}
    # malicious_key=np.random.randint(0, 10000, config['DEFENCE_NETWORK']['KEY_LENGTH']) / 10000
    malicious_key = np.array([random.random() for _ in range(config['DEFENCE_NETWORK']['KEY_LENGTH'])])
    if not config['DEBUG']:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print(f"\n>>>>>>>>>>>>> {save_path}\n")
        test_tfLogger = TFLogger(save_path + "/TFlog/Test/")
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
            torch.save(client_keys[idx], f'{model_path[idx]}/key.bin')
    best = 0
    best_local = 0
    for rd in range(rounds):
        print('\n')
        print("-" * 100)
        print(f"[Round: {rd}/{rounds}]")
        # train
        loss_locals = []
        acc_locals = []
        weight_locals = []
        tst_loss_total = []
        tst_acc_total = []
        for idx in idxs_client:
            weight_local, loss_local, acc_local,val_epoch_loss_avg, val_epoch_acc_avg = locals[idx].train(weight_global)
            tst_loss_total.append(val_epoch_loss_avg)
            tst_acc_total.append(val_epoch_acc_avg)
            if np.mean(val_epoch_acc_avg)>best_local and not config['DEBUG']:
                torch.save(weight_local, model_path[idx] + f"/client:{idx}-epoch:{str(rd).zfill(3)}-tst_loss:{np.mean(val_epoch_loss_avg)}-tst_acc:{np.mean(val_epoch_acc_avg)}-{modelID}.pth")
                # torch.save(weight_local, model_path[idx] + f"/client:{idx}-{modelID}-best.pth")
                best_local = np.mean(val_epoch_acc_avg)
                # train_tfLogger.scalar_summary(f'trn_loss_{idx}', loss_local, rd)
                # train_tfLogger.scalar_summary(f'trn_acc_{idx}', acc_local, rd)
            test_tfLogger.scalar_summary(f'tst_loss_{idx}', np.mean(val_epoch_loss_avg), rd)
            test_tfLogger.scalar_summary(f'tst_acc_{idx}', np.mean(val_epoch_acc_avg), rd)
            weight_locals.append(weight_local)
            loss_locals.append(loss_local)
            acc_locals.append(acc_local)
        weight_global = fedavg_weight(weight_locals)
        model_global.load_state_dict(weight_global)
        # if not config['DEBUG']:
        #     train_tfLogger.scalar_summary(f'trn_loss_avg', np.mean(loss_locals), rd)
        #     train_tfLogger.scalar_summary(f'trn_acc_avg', np.mean(acc_locals), rd)

        # test
        # test_loss_avg, test_acc_avg = evaluator(model_global, test_loader, nn.CrossEntropyLoss(), batchsize, client_keys)
        for k, v in initial_key_layer.items():
            model_global.state_dict()[k] = copy.deepcopy(v)
        test_loss_avg, test_acc_avg = tester(model_global, test_loader, nn.CrossEntropyLoss(), batchsize, malicious_key , "malicious")
        if not config['DEBUG']:
            print(save_path)
            test_tfLogger.scalar_summary('tst_loss', test_loss_avg, rd)
            test_tfLogger.scalar_summary('tst_acc', test_acc_avg, rd)
            test_tfLogger.scalar_summary('tst_loss_avg', np.mean(loss_locals), rd)
            test_tfLogger.scalar_summary('tst_acc_avg', np.mean(acc_locals), rd)
            print(f"Round {rd}\nLocal trn loss: {np.mean(loss_locals)}, Local trn Acc: {np.mean(acc_locals)}\nTest  Loss: {test_loss_avg}, Test  Acc: {test_acc_avg}")
            # print(f"Round {rd}\nLocal loss: {np.mean(loss_locals)}, Local Acc: {np.mean(acc_locals)}")
            if np.mean(tst_acc_total)>best:
                best=np.mean(tst_acc_total)
                torch.save(model_global.state_dict(),
                           f'{save_path}/{modelID}-{config["METHODS"]}-round:{str(rd).zfill(3)}-tst_loss:{np.round(np.mean(tst_loss_total), 4)}-avg_tst_acc:{np.round(np.mean(tst_acc_total), 4)}-tst_acc:{test_acc_avg}-best.pth')
    # if not config['DEBUG']:
    test_tfLogger.close()
#     train_tfLogger.close()
