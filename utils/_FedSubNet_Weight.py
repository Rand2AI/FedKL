# -*-coding:utf-8-*-
import copy
import torch


def fedsubnet_rdm_weight(w):
    w_avg = copy.deepcopy(w[0])
    for layer_name in w_avg.keys():
        if len(w_avg[layer_name].shape) == 0:
            # set special layer to be the avg over all local models.
            zero_temp_layer_list = []
            for c in w[1:]:
                zero_temp_layer_list.append(c[layer_name])
            w_avg[layer_name] = torch.mean(torch.tensor(zero_temp_layer_list, dtype=torch.float))
            continue
        non_zero_curren_layer = None
        for client_weiths in w:
            if non_zero_curren_layer is None:
                non_zero_curren_layer = client_weiths[layer_name].unsqueeze(0)
            else:
                if not torch.sum(non_zero_curren_layer)==0:
                    non_zero_curren_layer = torch.cat((non_zero_curren_layer, client_weiths[layer_name].unsqueeze(0)), 0)
        w_avg[layer_name] = torch.mean(non_zero_curren_layer.float(), dim=0, keepdim=True).squeeze(0)
    return w_avg
