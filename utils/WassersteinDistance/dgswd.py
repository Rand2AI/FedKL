# -*-coding:utf-8-*-
import torch
import torch.nn as nn
from torch import  optim
import numpy as np

class TransformNet(nn.Module):
    def __init__(self, size):
        super(TransformNet, self).__init__()
        self.size = size
        self.net = nn.Sequential(nn.Linear(self.size,self.size))
    def forward(self, inputs):
        out =self.net(inputs)
        return out/torch.sqrt(torch.sum(out**2,dim=1,keepdim=True))

def rand_projections(dim, num_projections=1000):
    projections = torch.randn((num_projections, dim))
    projections /= torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections

def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mean(torch.abs(torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)))

def cost_matrix_slow(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)

def distributional_generalized_sliced_wasserstein_distance(first_samples, second_samples, num_projections=1000, f=None,
                                                           f_op=None, r=1,
                                                           p=2, max_iter=10, lam=1, device='cuda'):
    embedding_dim = first_samples.size(1)
    if f is None:
        f = TransformNet(embedding_dim).to(device)
    if f_op is None:
        f_op = optim.Adam(f.parameters())
    pro = rand_projections(embedding_dim, num_projections).to(device)
    for _ in range(max_iter):
        projections = f(pro)
        reg = lam * cosine_distance_torch(projections, projections)
        wasserstein_distance = cost_matrix_slow(first_samples, second_samples, projections, r, p)
        loss = reg - wasserstein_distance
        f_op.zero_grad()
        loss.backward(retain_graph=True)
        f_op.step()
    projections = f(pro)
    wasserstein_distance = cost_matrix_slow(first_samples, second_samples, projections, r, p)
    return wasserstein_distance