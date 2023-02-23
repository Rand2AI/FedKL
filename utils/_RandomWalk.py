import copy
from random import shuffle, random, sample


def core_selection(config, total_num, temp_model_layer, backup_model_layer, reached_last_layer, core_num = None):
    selected_core_num = 0
    current_layer_idx = 0
    sub_layers = {idx: [] for idx in range(len(temp_model_layer))}
    step_ratio = config['SUBNET']['RANDOM_WALKER_RATIO']
    plot_xy = [1, [0]]
    initial_model_layer = copy.deepcopy(temp_model_layer)
    if core_num is None:
        core_num = int(total_num * config['SUBNET']['WEIGHTS_DROP_PROPORTION'])
    while selected_core_num < core_num:
        if len(temp_model_layer[-1]) == 0 and not reached_last_layer:
            temp_model_layer[-1] = copy.deepcopy(backup_model_layer[-1])
        if config['SUBNET']['COVER_MODE']:
            # count the kernal number to decide if an initialization is needed or not.
            temp_count = 0
            for v in temp_model_layer:
                temp_count+=len(v)
            if temp_count==0:
                temp_model_layer = copy.deepcopy(backup_model_layer)
                for idx in range(len(temp_model_layer)):
                    # avoid repeat kernal to be selected
                    temp_model_layer[idx] = list(set(temp_model_layer[idx]).difference(set(initial_model_layer[idx])))
                    shuffle(temp_model_layer[idx])
        current_layer = temp_model_layer[current_layer_idx]
        if len(current_layer) > 0:
            sub_layers[current_layer_idx].append(current_layer.pop(0))
            selected_core_num += 1
            plot_xy[0] += 1
            plot_xy[1].append(current_layer_idx)
            if current_layer_idx == 0 or random() > step_ratio:
                current_layer_idx = min(len(temp_model_layer) - 1, current_layer_idx + 1)
            elif current_layer_idx == len(temp_model_layer) - 1 or random() < step_ratio:
                current_layer_idx = max(0, current_layer_idx - 1)
                if not reached_last_layer:
                    reached_last_layer = True
        else:
            # if reached the last layer, random choose a layer index
            if reached_last_layer:
                current_layer_idx = sample(range(0, len(temp_model_layer)), 1)[0]
            # go next layer
            else:
                current_layer_idx = min(len(temp_model_layer) - 1, current_layer_idx + 1)
    return sub_layers, reached_last_layer, plot_xy, temp_model_layer


def preprocess_layer(weight_global):
    model_layer = []
    model_layer_name = []
    for k, v in weight_global.items():
        if 'weight' in k and 'bn' not in k:
            model_layer.append(list(range(v.size(0))))
            model_layer_name.append(k)
    total_core_num = 0
    for v in model_layer:
        total_core_num += len(v)
    return model_layer, model_layer_name, total_core_num
