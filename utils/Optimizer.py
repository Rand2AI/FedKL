# -*-coding:utf-8-*-
import torch


def set_optimizer(params, config):
    lr = config['OPTIMIZER']['LEARNING_RATE']
    opt = config['OPTIMIZER']['OPT_BACKPROP']
    decay = config['OPTIMIZER']['DECAY']
    momentum = config['OPTIMIZER']['MOMENTUM']
    if opt == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=decay,
                                    nesterov=config['OPTIMIZER']['SGD_NESTEROV'])
    elif opt == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=decay)
    elif opt == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=decay)
    elif opt == 'adadelta':
        optimizer = torch.optim.Adadelta(params, lr=lr, weight_decay=decay)
    else:
        raise Exception("Wrong optimizer name.")
    return optimizer
