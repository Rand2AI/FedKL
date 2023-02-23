# -*-coding:utf-8-*-
import torch
import numpy as np

def rand_projections(dim, num_projections=1000):
    projections = torch.randn((num_projections, dim))
    projections /= torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections

def circular_function(x1, x2, theta, r, p):
    cost_matrix_1 = torch.sqrt(cost_matrix_slow(x1, theta * r))
    cost_matrix_2 = torch.sqrt(cost_matrix_slow(x2, theta * r))
    wasserstein_distance = torch.abs((torch.sort(cost_matrix_1.transpose(0, 1), dim=1)[0] -
                                      torch.sort(cost_matrix_2.transpose(0, 1), dim=1)[0]))
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1), 1. / p)
    return torch.pow(torch.pow(wasserstein_distance, p).mean(), 1. / p)

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

def generalized_sliced_wasserstein_distance(first_samples,
                                            second_samples,
                                            r=1,
                                            num_projections=1000,
                                            p=2,
                                            device='cuda'):
    embedding_dim = first_samples.size(1)
    projections = rand_projections(embedding_dim, num_projections).to(device)
    return circular_function(first_samples, second_samples, projections, r, p)
