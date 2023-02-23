# -*-coding:utf-8-*-
import torch
import numpy as np

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

def circular_function(x1, x2, theta, r, p):
    cost_matrix_1 = torch.sqrt(cost_matrix_slow(x1, theta * r))
    cost_matrix_2 = torch.sqrt(cost_matrix_slow(x2, theta * r))
    wasserstein_distance = torch.abs((torch.sort(cost_matrix_1.transpose(0, 1), dim=1)[0] -
                                      torch.sort(cost_matrix_2.transpose(0, 1), dim=1)[0]))
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1), 1. / p)
    return torch.pow(torch.pow(wasserstein_distance, p).mean(), 1. / p)

def max_generalized_sliced_wasserstein_distance(first_samples,
                                                second_samples,
                                                r=1,
                                                p=2,
                                                max_iter=100,
                                                device='cuda'):
    theta = torch.randn((1, first_samples.shape[1]), device=device, requires_grad=True)
    theta.data /= torch.sqrt(torch.sum(theta.data ** 2, dim=1))
    opt = torch.optim.Adam([theta], lr=1e-4)
    for _ in range(max_iter):
        wasserstein_distance = circular_function(first_samples, second_samples, theta, r, p)
        l = - wasserstein_distance
        opt.zero_grad()
        l.backward(retain_graph=True)
        opt.step()
        theta.data /= torch.sqrt(torch.sum(theta.data ** 2, dim=1))
    wasserstein_distance = circular_function(first_samples, second_samples, theta, r, p)
    return wasserstein_distance