import time, datetime, random
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
torch.set_default_tensor_type('torch.cuda.FloatTensor')

from utils.WassersteinDistance import wasserstein_distance
from utils.Defence_utils import *
from utils import get_config, save_args_as_json
from backbone.Model import build_leakage_model

config = get_config(os.path.dirname(os.path.realpath(__file__)))
np.random.seed(999)
torch.manual_seed(999)
torch.cuda.manual_seed_all(999)

def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except:
        pass

def main():
    GPU = "1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    dataset = 'mnist'  # mnist cifar10 cifar100
    net_name = 'lenet'  # lenet res20 res18
    shape_img = (32, 32)
    with_kl = True
    share_key = False
    gen_key = True
    with_lock_layer = False

    num_batch = 1
    Iteration = 500
    num_exp = 20
    key_length = 1024

    if share_key:
        gen_key = False  # force to False as no need to regress key
    save_path = f"./DLG-{net_name}-{dataset}-{shape_img[0]}-B{str(num_batch).zfill(4)}-{key_length}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
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
    save_img_path = save_path + "/saved_img/"

    log_path = save_path + "/Log/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    args_json_path = save_path + "/args.json"
    save_args_as_json(config, args_json_path)

    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    tp = transforms.Compose([transforms.ToPILImage()])
    criterion = cross_entropy_for_onehot
    dst, num_classes, channel, hidden = GRNN_gen_dataset(dataset, shape_img)
    print(f'\n>>>>>>> GPU: {GPU}')
    print(f'>>>>>>> {str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))}: {save_path}')
    ''' train DLG and iDLG '''
    dataloader = torch.utils.data.DataLoader(dst, batch_size=num_batch, shuffle=False)
    G_train_loader = iter(dataloader)
    for idx_net in range(num_exp):
        # train_tfLogger = TFLogger(f'{save_path}/tfrecoard-exp-{str(idx_net).zfill(2)}')
        net = build_leakage_model(net_name, key_length, num_classes, with_kl)

        # have to do this, otherwise will lead to a failure.
        net.apply(weights_init)
        print(f'{str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))}: running {idx_net}|{num_exp} experiment')
        net = net.to(device)
        gt_data, gt_label = next(G_train_loader)
        gt_data, gt_label = gt_data.cuda(), gt_label.cuda()
        gt_onehot_label = label_to_onehot(gt_label, num_classes)

        imidx_list = []
        for imidx in range(num_batch):
            imidx_list.append(gt_label[imidx])
        # compute original gradient

        key = torch.tensor(np.array([random.random() for _ in range(key_length)])).float().to(device)
        if with_kl:
            out = net(gt_data, key)
        else:
            out = net(gt_data)

        y = criterion(out, gt_onehot_label)
        dy_dx = torch.autograd.grad(y, net.parameters())
        new_dy_dx = split_gradient(net, with_lock_layer, dy_dx)
        original_dy_dx = list((_.detach().clone() for _ in new_dy_dx))
        # generate dummy data and label
        dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
        dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

        if share_key:
            G_key = key.clone().detach()
        else:
            G_key = torch.tensor(np.array([random.random() for _ in range(key_length)])).float().to(device).requires_grad_(gen_key)
        if gen_key:
            optimizer = torch.optim.LBFGS([dummy_data, dummy_label, G_key], lr=1)
            # optimizer = torch.optim.RMSprop([dummy_data, dummy_label, G_key], lr=0.0001, momentum=0.99)
        else:
            optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=1)
            # optimizer = torch.optim.RMSprop([dummy_data, dummy_label], lr=0.0001, momentum=0.99)
        history = []
        history_iters = []
        losses = []
        mses = []
        train_iters = []
        iter_bar = tqdm(range(Iteration), total=Iteration, desc=f'{str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))}', ncols=150)
        for iters in iter_bar:
            def closure():
                optimizer.zero_grad()
                if with_kl:
                    pred = net(dummy_data, G_key)
                else:
                    pred = net(dummy_data)
                dummy_loss = criterion(pred, gt_onehot_label)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                new_dummy_dy_dx = split_gradient(net, with_lock_layer, dummy_dy_dx)
                grad_diff = 0
                for gx, gy in zip(new_dummy_dy_dx, original_dy_dx):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
                return grad_diff
            optimizer.step(closure)
            current_loss = closure().item()
            train_iters.append(iters)
            losses.append(current_loss)
            mses.append(torch.mean(abs(dummy_data - gt_data)).item())
            iter_bar.set_postfix(loss=round(current_loss,8),
                                 mses=round(mses[-1], 8))

            # train_tfLogger.scalar_summary("g_l2", current_loss, iters)
            # train_tfLogger.scalar_summary("img_mses", mses[-1], iters)
            # train_tfLogger.scalar_summary("img_wd", wasserstein_distance(dummy_data.view(1, -1), gt_data.view(1, -1)).item(),
            #                               iters)

            if iters % int(Iteration / 10) == 0:
                history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_batch)])
                history_iters.append(iters)
                # if current_loss < 0.0000001: # converge
                #     break
        for imidx in range(num_batch):
            plt.figure(figsize=(12, 8))
            plt.subplot(3, 10, 1)
            plt.imshow(tp(gt_data[min(imidx, len(gt_data) - 1)].cpu()))
            for i in range(min(len(history), 29)):
                plt.subplot(3, 10, i + 2)
                plt.imshow(history[i][imidx])
                plt.title('i:%d, l:%d' % (history_iters[i], torch.argmax(dummy_label, dim=-1)[imidx].item()))
                plt.axis('off')
            path = f"{save_path}/"

            true_path = save_img_path + f"true_data/exp-{str(idx_net).zfill(6)}/"
            fake_path = save_img_path + f"fake_data/exp-{str(idx_net).zfill(6)}/"
            if not os.path.exists(true_path) or not os.path.exists(fake_path):
                os.makedirs(true_path)
                os.makedirs(fake_path)
            tp(gt_data[imidx].cpu()).save(true_path + f"/{imidx}_{gt_label[imidx].item()}.png")
            history[i][imidx].save(fake_path + f"/{imidx}_{torch.argmax(dummy_label, dim=-1)[imidx].item()}.png")

            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path + '/exp:%03d-imidx:%02d-tlabel:%s-Glabel:%d.png' % (idx_net,imidx, imidx_list[min(imidx, len(imidx_list) - 1)], torch.argmax(dummy_label, dim=-1)[imidx].item()))
            plt.close()
        iter_bar.close()
        print('----------------------')

if __name__ == '__main__':
    main()