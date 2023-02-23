# -*- coding: utf-8 -*-
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch, copy
import numpy as np
from utils import set_optimizer, trainer, tester


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class local_update(object):
    def __init__(self, config, client_idx, dataset, data_idxs, model, test_loader,client_keys=None):
        self.test_loader = test_loader
        self.client_keys = client_keys
        if self.client_keys is not None:
            self.client_key = torch.tensor(client_keys[client_idx]).float()
        self.model=model
        self.client_idx = client_idx
        self.data_idxs = data_idxs
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = DataLoader(DatasetSplit(dataset, data_idxs), batch_size=self.config['TRAIN']['BATCH_SIZE'],
                                       shuffle=True, num_workers=4*len(self.config['DEVICE']['DEVICE_GPUID']), pin_memory=True)
        self.optimizer = set_optimizer(self.model.parameters(), self.config)
        # self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[config['TRAIN']['ROUNDS']//3, 2 * (config['TRAIN']['ROUNDS']//3)], gamma=0.1)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                 milestones=[60, 120, 160],
                                                                 gamma=0.2)
        self.key_layer = None

    def train(self, weight_global):
        if self.key_layer is not None:
            for k, v in self.key_layer.items():
                weight_global[k] = v.clone().detach()
        self.model.load_state_dict(weight_global)

        epoch_loss = []
        epoch_acc = []
        for epoch in range(self.config['FED']['CLIENT_EPOCH']):
            other_client_idx = self.client_idx+1 if self.client_idx<2 else 0
            print(f"[Client {self.client_idx} Epoch: {epoch + 1}/{self.config['FED']['CLIENT_EPOCH']}]")
            if self.client_keys is not None:
                val_epoch_loss_avg, val_epoch_acc_avg = tester(self.model, self.test_loader, self.criterion,
                                                               self.config['TRAIN']['BATCH_SIZE'], self.client_key,
                                                               self.client_idx)
                # _,_ = tester(self.model, self.test_loader, self.criterion, self.config['TRAIN']['BATCH_SIZE'],
                #              np.random.randint(0, 10000, self.config['DEFENCE_NETWORK']['KEY_LENGTH'])/10000, "malicious")
                # _,_ = tester(self.model, self.test_loader, self.criterion, self.config['TRAIN']['BATCH_SIZE'],
                #              self.client_keys[other_client_idx], other_client_idx)
                train_epoch_loss_avg, train_epoch_acc_avg = trainer(self.model, self.train_loader, self.optimizer, self.criterion,
                                                                    self.config['TRAIN']['BATCH_SIZE'], self.client_key)
            else:
                train_epoch_loss_avg, train_epoch_acc_avg = trainer(self.model, self.train_loader, self.optimizer,
                                                                    self.criterion,
                                                                    self.config['TRAIN']['BATCH_SIZE'])
            epoch_loss.append(train_epoch_loss_avg)
            epoch_acc.append(train_epoch_acc_avg)
            print("-" * 10)
        self.lr_scheduler.step()
        local_weights = self.model.state_dict()
        if self.client_keys is not None:
            self.key_layer = {k:v.clone().detach() for k,v in local_weights.items() if 'scale_fc' in k or 'shift_fc' in k}
            for k,v in self.key_layer.items():
                local_weights[k] = torch.zeros_like(v)
            return local_weights, np.mean(epoch_loss), np.mean(epoch_acc), val_epoch_loss_avg, val_epoch_acc_avg
        else:
            return local_weights, np.mean(epoch_loss), np.mean(epoch_acc)
