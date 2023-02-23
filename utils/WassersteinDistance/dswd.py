# -*-coding:utf-8-*-
import torch
import torch.nn as nn
from torch import  optim

def rand_projections(dim, num_projections=1000):
    projections = torch.randn((num_projections, dim))
    projections /= torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections

def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mean(torch.abs(torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)))

class TransformNet(nn.Module):
    def __init__(self, size):
        super(TransformNet, self).__init__()
        self.size = size
        self.net = nn.Sequential(nn.Linear(self.size,self.size))
    def forward(self, inputs):
        out =self.net(inputs)
        return out/torch.sqrt(torch.sum(out**2,dim=1,keepdim=True))

def distributional_sliced_wasserstein_distance(first_samples, second_samples, num_projections=1000, f=None, f_op=None,
                                               p=2, max_iter=10, lam=1, device='cuda'):
    embedding_dim = first_samples.size(1)
    if f is None:
        f = TransformNet(embedding_dim).to(device)
    if f_op is None:
        f_op = optim.Adam(f.parameters())
    pro = rand_projections(embedding_dim, num_projections).to(device)
    first_samples_detach = first_samples.detach()
    second_samples_detach = second_samples.detach()
    for _ in range(max_iter):
        projections = f(pro)
        cos = cosine_distance_torch(projections, projections)
        reg = lam * cos
        encoded_projections = first_samples_detach.matmul(projections.transpose(0, 1))
        distribution_projections = (second_samples_detach.matmul(projections.transpose(0, 1)))
        wasserstein_distance = torch.abs((torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                                          torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]))
        wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1), 1. / p)
        wasserstein_distance = torch.pow(torch.pow(wasserstein_distance, p).mean(), 1. / p)
        loss = reg - wasserstein_distance
        f_op.zero_grad()
        loss.backward(retain_graph=True)
        f_op.step()

    projections = f(pro)
    encoded_projections = first_samples.matmul(projections.transpose(0, 1))
    distribution_projections = (second_samples.matmul(projections.transpose(0, 1)))
    wasserstein_distance = torch.abs((torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                                      torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]))
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1), 1. / p)
    wasserstein_distance = torch.pow(torch.pow(wasserstein_distance, p).mean(), 1. / p)
    return wasserstein_distance