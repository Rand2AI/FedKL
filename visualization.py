import os
import matplotlib.pyplot as plt

def acc_loss():
    # dataset= 'cifar10' # cifar10 cifar100 imagenet mnist
    # network = 'resnet18' # resnet18 resnet34 resnet20 resnet32 vgg16 lenet
    datasets = ['cifar10', 'cifar100']
    networks = ['resnet18', 'resnet34', 'resnet20', 'resnet32', 'vgg16']
    for dataset in datasets:
        for network in networks:
            with_path = f"./{dataset}/"
            with_defence_acc = []
            with_defence_loss = []
            without_defence_acc = []
            without_defence_loss = []
            for dirs in os.listdir(with_path):
                if network in dirs:
                    print(dirs)
                    if 'with_defence' in dirs or '-kl' in dirs:
                        for model_name in os.listdir(f"{with_path}{dirs}"):
                            if 'acc' in model_name:
                                with_defence_acc.append(float(model_name.split('-best')[0].split('_acc:')[1]))
                                with_defence_loss.append(float(model_name.split('-tst_acc')[0].split('_loss:')[1]))
                    else:
                        for model_name in os.listdir(f"{with_path}{dirs}"):
                            if 'acc' in model_name:
                                without_defence_acc.append(float(model_name.split('-best')[0].split('_acc:')[1]))
                                without_defence_loss.append(float(model_name.split('-tst_acc')[0].split('_loss:')[1]))
            min_num_acc = min(len(with_defence_acc), len(without_defence_acc))
            min_num_loss = min(len(with_defence_loss), len(without_defence_loss))
            with_defence_acc = with_defence_acc[:min_num_acc]
            with_defence_loss = with_defence_loss[:min_num_loss]
            without_defence_acc = without_defence_acc[:min_num_acc]
            without_defence_loss = without_defence_loss[:min_num_loss]

            with_defence_acc.sort()
            without_defence_acc.sort()
            with_defence_loss.sort(reverse=True)
            without_defence_loss.sort(reverse=True)

            save_path = f'./FIgures/acc_loss/{dataset}-{network}'
            x = list(range(0,200, int(200/min_num_acc)))[:min_num_acc]
            plt.plot(x , with_defence_acc, "or-", label=f'with KL')
            plt.plot(x, without_defence_acc, "^b:",label=f'without KL')
            plt.tick_params(labelsize=20)
            # plt.xlabel('Epoch')
            # plt.ylabel('Accuracy')
            plt.legend(fontsize=20)
            plt.savefig(f'{save_path}_acc.png', pad_inches=0.02, bbox_inches='tight')
            plt.show()


            plt.plot(list(range(0, 200, int(200 / min_num_loss)))[:min_num_loss], with_defence_loss, "or-", label=f'with KL')
            plt.plot(list(range(0, 200, int(200 / min_num_loss)))[:min_num_loss], without_defence_loss, "^b:", label=f'without KL')
            plt.tick_params(labelsize=18)
            # plt.xlabel('Epoch')
            # plt.ylabel('Loss')
            plt.legend(fontsize=20)
            plt.savefig(f'{save_path}_loss.png', pad_inches=0.02, bbox_inches='tight')
            plt.show()

if __name__=='__main__':
    acc_loss()