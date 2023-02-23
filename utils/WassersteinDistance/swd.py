# -*-coding:utf-8-*-
import torch

def rand_projections(dim, num_projections=1000):
    projections = torch.randn((num_projections, dim))
    projections /= torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections

def wasserstein_distance(first_samples,
                         second_samples,
                         p=2,
                         device='cuda'):
    # wasserstein_distance = torch.abs((torch.sort(first_samples, dim=1)[0] - torch.sort(second_samples, dim=1)[0]))
    # wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1), 1. / p)

    wasserstein_distance = torch.abs(first_samples[0] - second_samples[0])
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p)), 1. / p)
    return torch.pow(torch.pow(wasserstein_distance, p).mean(), 1. / p)

def sliced_wasserstein_distance(first_samples,
                                second_samples,
                                num_projections=1000,
                                p=2,
                                device='cuda'):
    dim = second_samples.size(1)
    projections = rand_projections(dim, num_projections).to(device)
    first_projections = first_samples.matmul(projections.transpose(0, 1))
    second_projections = (second_samples.matmul(projections.transpose(0, 1)))
    wasserstein_distance = torch.abs((torch.sort(first_projections.transpose(0, 1), dim=1)[0] -
                                      torch.sort(second_projections.transpose(0, 1), dim=1)[0]))
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1), 1. / p)
    return torch.pow(torch.pow(wasserstein_distance, p).mean(), 1. / p)
