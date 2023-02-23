from backbone.Model import build_leakage_model
import matplotlib.pyplot as plt
import datetime, random, torchvision
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal, convert_to_images)
from utils.GGL.reconstructor import NGReconstructor, BOReconstructor, AdamReconstructor, Generator
from utils.Defence_utils import *
from utils.GGL import inversefed
import cv2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

Iteration = 1
key_length = 1024
# img index
# idx = random.randint(0, 1000)
# idxes = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
idxes = list(range(99,99999, 1000))
# idxes = [1000, 1050, 1100, 1150]
repeat = 1
dataset = 'imagenet' # cifar10 cifar100 imagenet
net_name = 'res18'  # lenet res20 res18
# shape_img = (32, 32)
shape_img = (256, 256)
with_kl = True
share_key = False
with_lock_layer = False

save_path = f"./GGL-{net_name}-{dataset}-{shape_img[0]}-B{str(1).zfill(4)}-{key_length}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
if not with_kl:
    save_path += f"-no_kl"
else:
    save_path += f"-kl"
if share_key:
    save_path += f"-share_key"
if with_lock_layer:
    save_path += f"-with_lock_layer"
print(save_path)

setup = inversefed.utils.system_startup()
defs = inversefed.training_strategy('conservative')
device = setup['device']

loss_fn, trainloader, validloader, num_classes =  inversefed.construct_dataloaders(dataset, defs, shape_img[0])
model = build_leakage_model(net_name, key_length, num_classes, with_kl)
# model = torchvision.models.resnet18(pretrained=True)
model.to(**setup)
model.eval()

dm = torch.as_tensor([0.5, 0.5, 0.5], **setup)[:, None, None]
ds = torch.as_tensor([0.5, 0.5, 0.5], **setup)[:, None, None]

if shape_img[0] == 256:
    generator= BigGAN.from_pretrained('biggan-deep-256')
elif shape_img[0] == 32:
    generator = Generator()
    checkpoint = torch.load("") # path to the pregrained wgan model
    generator.load_state_dict(checkpoint['state_dict'])
    generator.eval()
else:
    raise ValueError('shape_img should be 32 or 256')
generator.to(device)

for idx in idxes:
    for i in range(repeat):
        img, label = validloader.dataset[idx]
        labels = torch.as_tensor((label,), device=setup['device'])
        ground_truth = img.to(**setup).unsqueeze(0)
        print(f"idx: {idx}, label: {[trainloader.dataset.classes[l] for l in labels]}")

        model.zero_grad()
        if with_kl:
            key = torch.tensor(np.array([random.random() for _ in range(key_length)])).float().to(device)
            target_loss, _, _ = loss_fn(model(ground_truth, key), labels)
        else:
            key = None
            target_loss, _, _ = loss_fn(model(ground_truth), labels)

        if share_key:
            G_key = key.clone().detach()
        else:
            G_key = torch.tensor(np.array([random.random() for _ in range(key_length)])).float().cuda(device).requires_grad_(False)

        input_gradient = torch.autograd.grad(target_loss, model.parameters())
        input_gradient = split_gradient(model, with_lock_layer, input_gradient)
        input_gradient = [grad.detach() for grad in input_gradient]

        defense_setting = None
        res_trials = [None]
        loss_trials = [None]

        ng_rec = NGReconstructor(fl_model=model, generator=generator, loss_fn=loss_fn, with_kl=with_kl, G_key=G_key, with_lock_layer=with_lock_layer, shape_img=shape_img,
                                 num_classes=num_classes, search_dim=(128,), strategy="CMA", budget=Iteration, use_tanh=True, use_weight=False, defense_setting=defense_setting)
        z_res, x_res, img_res, loss_res = ng_rec.reconstruct(input_gradient)

        original_img = ground_truth.mul_(ds).add_(dm).clamp_(0, 1).mul_(255).permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[
            0].cpu().numpy()
        res_img = x_res.mul_(ds).add_(dm).clamp_(0, 1).mul_(255).permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[
            0].cpu().numpy()
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.imsave(os.path.join(save_path, f'{idx}_true.png'), cv2.resize(original_img, shape_img, interpolation=cv2.INTER_CUBIC))
        plt.imsave(os.path.join(save_path, f'{idx}_out_{i}.png'), cv2.resize(res_img, shape_img, interpolation=cv2.INTER_CUBIC))
        # np.save(os.path.join(save_img_path, 'z.npy'), z_res.clone().cpu().numpy())