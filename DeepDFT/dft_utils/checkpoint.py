import os
import pathlib
import json

import mindspore as ms
from mindspore import nn
from mindspore.nn import learning_rate_schedule as lrs


def save_checkpoint(save_path, /, net:nn.Cell, optimizer:nn.Optimizer,**kwargs):
    #assert pathlib.Path(save_path).is_dir(), "save path must be a filefolder like path"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ms.train.serialization.save_checkpoint(net, os.path.join(save_path, "model.ckpt"))
    ms.train.serialization.save_checkpoint(optimizer, os.path.join(save_path, "optimizer.ckpt"))
    extra_info = kwargs
    with open(os.path.join(save_path, "extra_info.json"), 'w') as f:
        json.dump(extra_info, f)
    

def load_checkpoint(load_path, /, net:nn.Cell, optimizer:nn.Optimizer):
    assert os.path.exists(load_path)

    net_params = ms.train.serialization.load_checkpoint(os.path.join(load_path, "model.ckpt"))
    ms.train.serialization.load_param_into_net(net, net_params)

    optimizer_params = ms.train.serialization.load_checkpoint(os.path.join(load_path, "optimizer.ckpt"))
    ms.train.serialization.load_param_into_net(optimizer, optimizer_params)

    with open(os.path.join(load_path, "extra_info.json"), 'r') as f:
        extra_info = json.load(f)

    return {"net": net_params,
            "optimizer": optimizer_params,
            "extra_info": extra_info}