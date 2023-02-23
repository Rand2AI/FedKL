# -*-coding:utf-8-*-
import datetime, torch
from tqdm import tqdm
import numpy as np


def evaluator(model, val_loader, criterion, batchsize):
    total_loss = []
    total_acc = []
    val_epoch_loss = []
    val_epoch_acc = []
    val_bar_obj = tqdm(val_loader,
                       total=len(val_loader),
                       desc=f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Val",
                       ncols=120)
    model.eval()
    for gt_data, gt_label in val_bar_obj:
        gt_data, gt_label = gt_data.cuda(), gt_label.cuda()
        with torch.no_grad():
            preds = model(gt_data)
            loss = criterion(preds, gt_label)
        _, preds = preds.max(1)
        n_correct = 0
        for pred, target in zip(preds, gt_label):
            if pred == target:
                n_correct += 1
        accuracy = n_correct / float(batchsize)
        val_epoch_loss.append(loss.cpu().detach().numpy())
        val_epoch_acc.append(accuracy)
        val_bar_obj.set_postfix(loss=np.mean(val_epoch_loss),
                                accuracy=np.mean(val_epoch_acc))
    total_loss.append(np.mean(val_epoch_loss))
    total_acc.append(np.mean(val_epoch_acc))
    return np.mean(total_loss), np.mean(total_acc)

def tester(model, val_loader, criterion, batchsize, key, idx):
    total_loss = []
    total_acc = []
    val_epoch_loss = []
    val_epoch_acc = []
    val_bar_obj = tqdm(val_loader,
                       total=len(val_loader),
                       desc=f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Test on client {idx}",
                       ncols=120)
    model.eval()
    key = torch.tensor(key).float().cuda()
    for gt_data, gt_label in val_bar_obj:
        with torch.no_grad():
            gt_data, gt_label = gt_data.cuda(), gt_label.cuda()
            preds = model(gt_data, key)
            loss = criterion(preds.cpu(), gt_label.cpu())
        _, preds = preds.max(1)
        n_correct = 0
        for pred, target in zip(preds, gt_label):
            if pred.item() == target.item():
                n_correct += 1
        accuracy = n_correct / float(batchsize)
        val_epoch_loss.append(loss.cpu().detach().numpy())
        val_epoch_acc.append(accuracy)
        val_bar_obj.set_postfix(loss=np.mean(val_epoch_loss),
                                accuracy=np.mean(val_epoch_acc))
    total_loss.append(np.mean(val_epoch_loss))
    total_acc.append(np.mean(val_epoch_acc))
    return np.mean(total_loss), np.mean(total_acc)


