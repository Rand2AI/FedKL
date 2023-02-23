# -*-coding:utf-8-*-
import os, torch
from torch.backends import cudnn

from utils import get_config
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from methods import *

if __name__ == '__main__':
    config = get_config(os.path.dirname(os.path.realpath(__file__)))
    if config['DEVICE']['DEVICE_TOUSE'] == 'GPU':
        seed = 0
        torch.manual_seed(seed)  # sets the seed for generating random numbers.
        torch.cuda.manual_seed(
            seed)  # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
        torch.cuda.manual_seed_all(
            seed)  # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.

        if seed == 0:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch.multiprocessing.set_start_method('spawn')
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(x) for x in config['DEVICE']['DEVICE_GPUID']])
    else:
        raise Exception("Current version does not support CPU yet.")
    eval(config['METHODS'])(config)