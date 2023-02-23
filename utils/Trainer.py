# -*-coding:utf-8-*-
import datetime, torch
from tqdm import tqdm
import numpy as np


def trainer(model, train_loader, optimizer, criterion, batchsize, client_key=None):
    train_epoch_loss = []
    train_epoch_acc = []
    train_bar_obj = tqdm(train_loader,
                         total=len(train_loader),
                         desc=f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Train",
                         ncols=120)
    model.train()
    for gt_data, gt_label in train_bar_obj:
        # for p in model.parameters():
        #     p.requires_grad = True
        gt_data, gt_label = gt_data.cuda(), gt_label.cuda()
        if client_key  is not None:
            try:
                client_key = client_key.float().cuda()
            except:
                client_key = torch.tensor(client_key).float().cuda()
            preds_out = model(gt_data, client_key)
        else:
            preds_out = model(gt_data)
        loss = criterion(preds_out.cpu(), gt_label.cpu())
        _, preds = preds_out.max(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_epoch_loss.append(loss.cpu().detach().numpy())
        n_correct = 0
        for pred, target in zip(preds, gt_label):
            if pred.item() == target.item():
                n_correct += 1
        accuracy = n_correct / float(batchsize)
        train_epoch_acc.append(accuracy)
        train_bar_obj.set_postfix(loss=np.mean(train_epoch_loss),
                                  accuracy=np.mean(train_epoch_acc))
    return np.mean(train_epoch_loss), np.mean(train_epoch_acc)
