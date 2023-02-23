import time, datetime, random
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from backbone.Model import build_leakage_model
from utils.GRNN_Generator import generator
from utils.Defence_utils import *
from utils import save_args_as_json
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# -*-coding:utf-8-*-
import torch
import torch.nn as nn
from backbone.KL_layer_block import Defense_block
np.random.seed(999)
torch.manual_seed(999)
torch.cuda.manual_seed_all(999)

class LeNet2(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(LeNet2, self).__init__()
        self.body1 = nn.Sequential(
            nn.Conv2d(channel, 6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.body3 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=5),
            nn.Sigmoid(),
        )
        self.fc = nn.Sequential(
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        out = self.body1(x)
        out = self.body2(out)
        out = self.body3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class LeNet2_kl(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10, key_in_dim=1024):
        super(LeNet2_kl, self).__init__()
        self.conv1 = Defense_block(key_in_dim=key_in_dim,
                                      in_channels=channel,
                                      out_channels=6,
                                      kernel_size=5)
        self.act1 = nn.Sigmoid()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv2 = Defense_block(key_in_dim=6,
                                      in_channels=6,
                                      out_channels=16,
                                      kernel_size=5)
        self.act2 = nn.Sigmoid()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.body3 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=5),
            nn.Sigmoid(),
        )
        self.fc = nn.Sequential(
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x, key):
        out, key_g, key_b = self.conv1(x, key, key)
        out = self.act1(out)
        out = self.pool1(out)

        out, key_g, key_b = self.conv2(out, key_g, key_b)
        out = self.act2(out)
        out = self.pool2(out)

        out = self.body3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

from utils import get_config
config = get_config(os.path.dirname(os.path.realpath(__file__)))

def well_performanc_random():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device0 = 0
    device1 = 0
    batchsize = 1
    save_img_flag = True
    Iteration = 20000
    num_exp = 20
    key_length = 1024

    g_in = 1024
    plot_num = 30
    loss_mode = ["l2", 'wd', 'tv']
    loss_set = ["mse", "l1", "l2", "wd", "tv", "swd", "gswd", "mswd", "dswd", "mgswd", "dgswd", "csd"]
    dataset = 'cifar100' # mnist cifar10 cifar100 imagenet
    net_name = 'res20'  # lenet res20 res18
    shape_img = (32, 32)
    # shape_img = (256, 256)
    with_kl = True
    share_key = True
    gen_key = False
    with_lock_layer = True
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
    save_img_path = save_path+"/saved_img/"

    log_path = save_path + "/Log/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    args_json_path = save_path + "/args.json"
    save_args_as_json(config, args_json_path)

    dst, num_classes, channel, hidden = GRNN_gen_dataset(dataset, shape_img)
    tp = transforms.Compose([transforms.ToPILImage()])
    criterion = nn.CrossEntropyLoss().cuda(device1)
    print(f'{str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))}: {save_path}')
    G_train_loader = iter(torch.utils.data.DataLoader(dst, batch_size=batchsize, shuffle=False))
    for idx_net in range(num_exp):
        # train_tfLogger = TFLogger(f'{save_path}/tfrecoard-exp-{str(idx_net).zfill(4)}')
        print(f'{str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))}: Running {idx_net+1}|{num_exp} experiment')
        net = build_leakage_model(net_name, key_length, num_classes, with_kl)
        net = net.cuda(device1)

        Gnet = generator(num_classes=num_classes, channel=channel, shape_img=shape_img[0],
                         batchsize=batchsize, g_in=g_in).cuda(device0)

        gt_data, gt_label = next(G_train_loader)
        gt_data, gt_label = gt_data.cuda(device1), gt_label.cuda(device1)

        key = torch.tensor(np.array([random.random() for _ in range(key_length)])).float().cuda(device1)
        if with_kl:
            pred = net(gt_data, key)
        else:
            pred = net(gt_data)
        y = criterion(pred, gt_label)
        dy_dx = list(torch.autograd.grad(y, net.parameters()))
        new_dy_dx = split_gradient(net, with_lock_layer, dy_dx)
        flatten_true_g = flatten_gradients(new_dy_dx)

        # flatten_true_g = torch.zeros_like(flatten_true_g)           # all zero testing
        G_ran_in = torch.randn(batchsize, g_in).cuda(device0)
        iter_bar = tqdm(range(Iteration),
                        total=Iteration,
                        desc=f'{str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))}',
                        ncols=180)
        history = []
        history_l = []
        tf_his = []
        tf_his_i = []
        if share_key:
            G_key = key.clone().detach()
        else:
            G_key = torch.tensor(np.array([random.random() for _ in range(key_length)])).float().cuda(device1).requires_grad_(gen_key)
            gt_G_key=G_key.clone().detach()
        if gen_key:
            G_optimizer = torch.optim.RMSprop([{'params':Gnet.parameters()}, {'params':G_key}], lr=0.0001, momentum=0.99)
        else:
            G_optimizer = torch.optim.RMSprop(Gnet.parameters(), lr=0.0001, momentum=0.99)
        for iters in iter_bar:
            Gout, Glabel = Gnet(G_ran_in)
            Gout, Glabel = Gout.cuda(device1), Glabel.cuda(device1)
            G_optimizer.zero_grad()
            if with_kl:
                Gpred = net(Gout, G_key)
            else:
                Gpred = net(Gout)
            Gloss = - torch.mean(torch.sum(Glabel * torch.log(torch.softmax(Gpred, 1)), dim=-1))
            G_dy_dx = torch.autograd.grad(Gloss, net.parameters(), create_graph=True)
            new_G_dy_dx = split_gradient(net, with_lock_layer, G_dy_dx)
            flatten_fake_g = flatten_gradients(new_G_dy_dx).cuda(device1)
            loss_list = []
            for loss_name in loss_mode:
                loss_list.append(loss_f(loss_name=loss_name,
                                        flatten_fake_g=flatten_fake_g,
                                        flatten_true_g=flatten_true_g,
                                        device1=device1,
                                        Gout=Gout))
            grad_diff = sum(loss_list)
            grad_diff.backward()
            G_optimizer.step()
            if gen_key:
                iter_bar.set_postfix(total_loss = np.round(grad_diff.item(), 8),
                                     mses_img=round(torch.mean(abs(Gout-gt_data)).item(), 8),
                                     wd_img=round(wasserstein_distance(Gout.view(1,-1), gt_data.view(1,-1)).item(), 8),
                                     mse_key=round(torch.mean(abs(gt_G_key-G_key)).item(), 8))
                # train_tfLogger.scalar_summary("key_l2", torch.mean(abs(gt_G_key-G_key)).item(), iters)
                # train_tfLogger.scalar_summary("key_wd", wasserstein_distance(gt_G_key.view(1, -1), G_key.view(1, -1)).item(), iters)
            else:
                iter_bar.set_postfix(total_loss=np.round(grad_diff.item(), 8),
                                     mses_img=round(torch.mean(abs(Gout - gt_data)).item(), 8),
                                     wd_img=round(wasserstein_distance(Gout.view(1, -1), gt_data.view(1, -1)).item(), 8))

            # train_tfLogger.scalar_summary("g_l2", loss_list[0].item(), iters)
            # train_tfLogger.scalar_summary("g_wd", loss_list[1].item(), iters)
            # train_tfLogger.scalar_summary("g_tv", loss_list[2].item(), iters)
            # train_tfLogger.scalar_summary("img_mses", torch.mean(abs(Gout-gt_data)).item(), iters)
            # train_tfLogger.scalar_summary("img_wd", wasserstein_distance(Gout.view(1,-1), gt_data.view(1,-1)).item(), iters)
            # train_tfLogger.scalar_summary("toal_loss", grad_diff.item(), iters)

            if iters % int(Iteration / plot_num) == 0:
                tf_his.append([tp(Gout[imidx].detach().cpu()) for imidx in range(batchsize)])
                tf_his_i.append(iters)

            if iters % int(Iteration / plot_num) == 0:
                history.append([tp(Gout[imidx].detach().cpu()) for imidx in range(batchsize)])
                history_l.append([Glabel.argmax(dim=1)[imidx].item() for imidx in range(batchsize)])
            del Gloss, G_dy_dx, flatten_fake_g, grad_diff
        for imidx in range(batchsize):
            plt.figure(figsize=(12, 8))
            plt.subplot(plot_num//10, 10, 1)
            plt.imshow(tp(gt_data[imidx].cpu()))
            for i in range(min(len(history), plot_num-1)):
                plt.subplot(plot_num//10, 10, i + 2)
                plt.imshow(history[i][imidx])
                plt.title('l=%d' % (history_l[i][imidx]))
                # plt.title('i=%d,l=%d' % (history_iters[i], history_l[i][imidx]))
                plt.axis('off')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if save_img_flag:
                true_path = save_img_path + f"true_data/exp{str(idx_net).zfill(4)}/"
                fake_path = save_img_path + f"fake_data/exp{str(idx_net).zfill(4)}/"
                if not os.path.exists(true_path) or not os.path.exists(fake_path):
                    os.makedirs(true_path)
                    os.makedirs(fake_path)
                tp(gt_data[imidx].cpu()).save(true_path + f"/{imidx}_{gt_label[imidx].item()}.png")
                history[i][imidx].save(fake_path + f"/{imidx}_{Glabel.argmax(dim=1)[imidx].item()}.png")
            plt.savefig(save_path + '/exp:%04d-imidx:%03d-tlabel:%d-Glabel:%d.png' % (idx_net,imidx , gt_label[imidx].item(),Glabel.argmax(dim=1)[imidx].item()))
            plt.close()

        # train_tfLogger.images_summary([Glabel.argmax(dim=1)[imidx].item() for imidx in range(batchsize)], tf_his, tf_his_i)
        torch.cuda.empty_cache()
        history.clear()
        history_l.clear()
        tf_his.clear()
        tf_his_i.clear()
        iter_bar.close()
        # train_tfLogger.close()
        print('----------------------')

def save_img():
    import pickle
    data_path = "./Data/cifar100/"
    with open(data_path+"/cifar-100-python/train", mode='rb') as file:
        # 数据集在当脚本前文件夹下
        data_dict = pickle.load(file, encoding='bytes')
        data = list(data_dict[b'data'])
        labels = list(data_dict[b'fine_labels'])
    with open(data_path+"/cifar-100-python/meta", mode='rb') as file:
        data_dict = pickle.load(file, encoding='bytes')
        label_name = list(data_dict[b'fine_label_names'])
    img = np.reshape(data, [-1, 3, 32, 32])
    save_path = data_path + "/raw_img/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(100):
        r = img[i][0]
        g = img[i][1]
        b = img[i][2]
        ir = Image.fromarray(r)
        ig = Image.fromarray(g)
        ib = Image.fromarray(b)
        rgb = Image.merge("RGB", (ir, ig, ib))
        name = str(i) + "-" + label_name[labels[i]].decode() + ".png"
        rgb.save(save_path + name, "PNG")

if __name__ == '__main__':
    well_performanc_random()