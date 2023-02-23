# -*- coding: utf-8 -*-
import json
from utils.Configer import get_config, Logger
from utils.Data import gen_dataset, split_iid_data, split_noniid_data
from utils.Optimizer import set_optimizer
from utils.Evaluator import evaluator, tester
from utils.Trainer import trainer
from utils.FedAvg_Weight import fedavg_weight
from utils.Client import local_update
from utils.Logger import TFLogger

def save_args_as_json(FLconfig, path):
    with open(str(path), "w") as f:
        json.dump(FLconfig, f, indent=4)