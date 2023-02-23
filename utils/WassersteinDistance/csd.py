# -*-coding:utf-8-*-
import torch

def cosine_sum_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return -torch.mean(torch.mm(x1, x2.t()) / torch.mm(w1, w2.t()).clamp(min=eps))+1
