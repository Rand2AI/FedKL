import torch, os
from utils import *
from torch.utils.data import DataLoader
from backbone.Model import build_model



def main():
    config = get_config(os.path.dirname(os.path.realpath(__file__)))
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(x) for x in config['DEVICE']['DEVICE_GPUID']])
    torch.manual_seed(999)
    with_defence = config['DEFENCE_NETWORK']['WITH_DEFENCE']
    batchsize = config['TRAIN']['BATCH_SIZE']
    train_dataset, val_dataset, img_size, num_classes = gen_dataset(config['DATA']['TRAIN_DATA'],
                                                                    config['DATA']['IMG_SIZE'],
                                                                    config['DATA']['DATA_ROOT'])
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)
    model = build_model(num_classes, config)
    checkpoint = torch.load(config['TEST']['ROOT_PATH']+config['TEST']['MODEL_NAME'])
    model.load_state_dict(checkpoint)
    criterion = torch.nn.CrossEntropyLoss()
    if with_defence:
        key = torch.load(config['TEST']['ROOT_PATH'] + 'key.bin')
    if with_defence:
        val_epoch_loss_avg, val_epoch_acc_avg = tester(model, val_loader, criterion, batchsize, key, "")
    else:
        val_epoch_loss_avg, val_epoch_acc_avg = evaluator(model, val_loader, criterion, batchsize)

if __name__=='__main__':
    main()
