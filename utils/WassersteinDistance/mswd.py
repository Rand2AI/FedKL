# -*-coding:utf-8-*-
import torch
def max_sliced_wasserstein_distance(first_samples,
                                    second_samples,
                                    p=2,
                                    max_iter=100,
                                    device='cuda'):
    theta = torch.randn((1, first_samples.shape[1]), device=device, requires_grad=True)
    theta.data /= torch.sqrt(torch.sum(theta.data ** 2, dim=1))
    opt = torch.optim.Adam([theta], lr=1e-4)
    for _ in range(max_iter):
        encoded_projections = torch.matmul(first_samples, theta.transpose(0, 1))
        distribution_projections = torch.matmul(second_samples, theta.transpose(0, 1))
        wasserstein_distance = torch.abs((torch.sort(encoded_projections)[0] -
                                          torch.sort(distribution_projections)[0]))
        wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p))
        l = - wasserstein_distance
        opt.zero_grad()
        l.backward(retain_graph=True)
        opt.step()
        theta.data /= torch.sqrt(torch.sum(theta.data ** 2, dim=1))

    return wasserstein_distance
