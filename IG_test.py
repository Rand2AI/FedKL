import torch
import torchvision
import torch.nn as nn

from backbone.Model import build_leakage_model
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import cv2, os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

key_length = 1024
dataset = 'cifar10' # mnist cifar10 cifar100
net_name = 'res18'  # lenet res20 res18
# shape_img = (32, 32)
shape_img = (256, 256)
with_kl = True
share_key = True
gen_key = False
with_lock_layer = False
if share_key:
    gen_key = False # force to False as no need to regress key
save_path = f"./GRNN-{net_name}-{dataset}-{shape_img[0]}-B{str(batchsize).zfill(4)}-{key_length}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
if not with_kl:
    save_path += f"-no_kl"
else:
    save_path += f"-kl"
if share_key:
    save_path += f"-share_key"
if gen_key:
    save_path += f"-gen_key"
if with_lock_layer:
    save_path += f"-with_lock_layer"


# img index
idx = 3
save_img_path = save_path+f"/saved_img/{idx}/"
if not os.path.exists(save_img_path):
    os.makedirs(save_img_path)

from utils import inversefed
setup = inversefed.utils.system_startup()
defs = inversefed.training_strategy('conservative')

loss_fn, trainloader, validloader, num_classes =  inversefed.construct_dataloaders(dataset, defs, shape_img[0],
                                                                      data_path=f'/home/hans/WorkSpace/Data/{dataset}')

# model = torchvision.models.resnet18(pretrained=trained_model)
# model = Net()
model = build_leakage_model(net_name, key_length, num_classes, with_kl)
model.to(**setup)
model.eval()

dm = torch.as_tensor(inversefed.consts.imagenet_mean, **setup)[:, None, None]
ds = torch.as_tensor(inversefed.consts.imagenet_std, **setup)[:, None, None]
def plot(tensor):
    tensor = tensor.clone().detach()
    tensor.mul_(ds).add_(dm).clamp_(0, 1)
    if tensor.shape[0] == 1:
        gray = tensor[0].permute(1, 2, 0).cpu().numpy()
        colored = cv2.merge([gray, gray, gray])
        return plt.imshow(colored)
        # return plt.imshow(tensor[0].permute(1, 2, 0).cpu())
    else:
        fig, axes = plt.subplots(1, tensor.shape[0], figsize=(12, tensor.shape[0]*12))
        for i, im in enumerate(tensor):
            axes[i].imshow(im.permute(1, 2, 0).cpu())

img, label = validloader.dataset[idx]
labels = torch.as_tensor((label,), device=setup['device'])
ground_truth = img.to(**setup).unsqueeze(0)
# plot(ground_truth)
print([trainloader.dataset.classes[l] for l in labels])

ground_truth_denormalized = torch.clamp(ground_truth * ds + dm, 0, 1)
torchvision.utils.save_image(ground_truth_denormalized, f'{save_img_path}/true.png')

# ground_truth_denormalized = torch.clamp(ground_truth * ds + dm, 0, 1).permute(0,2,3,1)
# ground_truth_denormalized = ground_truth_denormalized[0].cpu().numpy()
# ground_truth_denormalized = cv2.resize(ground_truth_denormalized, (224, 224))
# cv2.imwrite(f'{idx}_{arch}_mnist_input.png', ground_truth_denormalized)

model.zero_grad()
target_loss, _, _ = loss_fn(model(ground_truth), labels)
input_gradient = torch.autograd.grad(target_loss, model.parameters())
input_gradient = [grad.detach() for grad in input_gradient]
full_norm = torch.stack([g.norm() for g in input_gradient]).mean()
print(f'Full gradient norm is {full_norm:e}.')

config = dict(signed=True,
              boxed=True,
              cost_fn='sim',
              indices='top10',
              weights='equal',
              lr=0.1,
              optim='adam',
              restarts=1,
              max_iterations=20_000,
              total_variation=1e-6,
              init='randn',
              filter='median',
              lr_decay=True,
              scoring_choice='loss')

rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=1)
output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=(3, shape_img[0], shape_img[0]))

test_mse = (output.detach() - ground_truth).pow(2).mean()
feat_mse = (model(output.detach())- model(ground_truth)).pow(2).mean()
test_psnr = inversefed.metrics.psnr(output, ground_truth, factor=1/ds)

plot(output)
plt.title(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} "
          f"| PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |");

data = inversefed.metrics.activation_errors(model, output, ground_truth)

fig, axes = plt.subplots(2, 3, sharey=False, figsize=(14,8))
axes[0, 0].semilogy(list(data['se'].values())[:-3])
axes[0, 0].set_title('SE')
axes[0, 1].semilogy(list(data['mse'].values())[:-3])
axes[0, 1].set_title('MSE')
axes[0, 2].plot(list(data['sim'].values())[:-3])
axes[0, 2].set_title('Similarity')

convs = [val for key, val in data['mse'].items() if 'conv' in key]
axes[1, 0].semilogy(convs)
axes[1, 0].set_title('MSE - conv layers')
convs = [val for key, val in data['mse'].items() if 'conv1' in key]
axes[1, 1].semilogy(convs)
convs = [val for key, val in data['mse'].items() if 'conv2' in key]
axes[1, 1].semilogy(convs)
axes[1, 1].set_title('MSE - conv1 vs conv2 layers')
bns = [val for key, val in data['mse'].items() if 'bn' in key]
axes[1, 2].plot(bns)
axes[1, 2].set_title('MSE - bn layers')
fig.suptitle('Error between layers')
plt.show()

output_denormalized = torch.clamp(output * ds + dm, 0, 1)
torchvision.utils.save_image(output_denormalized, f'{idx}_{arch}_mnist_untrained_output.png')

# output_denormalized = torch.clamp(output * ds + dm, 0, 1).permute(0,2,3,1)
# output_denormalized = output_denormalized.cpu().numpy()
# output_denormalized = cv2.resize(output_denormalized, (224, 224))
# cv2.imwrite(f'{idx}_{arch}_untrained_mnist_output.png', output_denormalized)
