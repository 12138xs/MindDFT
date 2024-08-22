import os
import sys
import json
import argparse
import math
import logging
import itertools
import timeit

import numpy as np
# import torch
# import torch.utils.data
# torch.set_num_threads(1)  # Try to avoid thread overload on cluster
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype 

from dft_utils.parser import *
from dft_utils import debuger
from dft_utils.dataset_utils import *
from dft_utils.train_utils import *

from gm import densitymodel
import dataset
 

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


# calculate mse rmse
def eval_model(model, dataloader, device):
    running_ae = ms.Tensor(0., dtype=ms.float32)
    running_se = ms.Tensor(0., dtype=ms.float32)
    running_count = ms.Tensor(0., dtype=ms.float32)

    for batch in dataloader:
        batch32 = convert_batch_int32(batch[0])
        outputs = model(batch32)
        targets = batch32["probe_target"]

        running_ae += ms.ops.abs(targets - outputs).sum()
        running_se += ms.ops.square(targets - outputs).sum()
        running_count += batch32["num_probes"].sum()

    mae = (running_ae / running_count).asnumpy()
    rmse = (ms.ops.sqrt(running_se / running_count)).asnumpy()

    return mae, rmse


# calculate means and \sqrt Var
def get_normalization(dataset, per_atom=True):
    try:
        num_targets = len(dataset.transformer.targets)
    except AttributeError:
        num_targets = 1
    x_sum = ms.ops.zeros(num_targets)
    x_2 = ms.ops.zeros(num_targets)
    num_objects = 0
    for sample in dataset:
        x = sample["targets"]
        if per_atom:
            x = x / sample["num_nodes"]
        x_sum += x
        x_2 += x ** 2.0
        num_objects += 1
    # Var(X) = E[X^2] - E[X]^2
    x_mean = x_sum / num_objects
    x_var = x_2 / num_objects - x_mean ** 2.0

    return x_mean, ms.ops.sqrt(x_var)


# def count_parameters(model):
#    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters(model):   
    total_params = 0  
    for _, param in model.parameters_and_names():  
        # 如果param.data.size是一个整数，说明是标量，直接加到总数中  
        if isinstance(param.data.size, int):  
            total_params += 1  
        else:  
            # 否则，使用numel()方法来计算张量中的元素个数  
            total_params += param.data.size.numel()  
    return total_params 


def main():
    args = get_arguments()

    # Create folder
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir, "printlog.txt"), mode="w"
            ),
            logging.StreamHandler(),
        ],
    )

    save_cmdargs(args.output_dir, "commandline_args.txt")
    save_parsed_cmdargs(args.output_dir, "arguments.json", args)

    filelist = get_data_files(args.dataset)

# ===============================================================================================
# DataLoading

    logging.info("loading data %s", args.dataset)

    densitydata = concat_data_files(filelist)
    datasplits = split_data(densitydata, args)
    datasplits["train"] = dataset.RotatingPoolData(datasplits["train"], 20)

    if args.ignore_pbc and args.force_pbc:
        raise ValueError("ignore_pbc and force_pbc are mutually exclusive and can't both be set at the same time")
    elif args.ignore_pbc:
        set_pbc = False
    elif args.force_pbc:
        set_pbc = True
    else:
        set_pbc = None

    train_loader = ms.dataset.GeneratorDataset(
        source=datasplits["train"],
        num_parallel_workers=4,
        column_names=["train"],
        sampler = ms.dataset.RandomSampler()
    )
    train_loader = train_loader.map(operations=dataset.CollateFuncRandomSample(args.cutoff, 1000, set_pbc_to=set_pbc),input_columns=["train"]) 
    train_loader = train_loader.batch(batch_size=1, drop_remainder=True)
    train_loader = train_loader.repeat(count=2)

    val_loader = ms.dataset.GeneratorDataset(
        datasplits["validation"],
        num_parallel_workers=4,
        column_names=["validations"],
        sampler=ms.dataset.RandomSampler()
    )
    val_loader = val_loader.map(operations=dataset.CollateFuncRandomSample(args.cutoff, 5000, set_pbc_to=set_pbc), input_columns=["validations"])
    val_loader = val_loader.batch(batch_size=1, drop_remainder=True)

    logging.info("Preloading validation batch")

# ===========================================================================================
# model relate

    ### 这里使用静态图,CPU
    #ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target='CPU')
    # 后面静态图跑不通，这里先用动态图试试
    #ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target=args.device)

    ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target=args.device)
    if args.use_painn_model:
        net = densitymodel.PainnDensityModel(args.num_interactions, args.node_size, args.cutoff)
    else:
        net = densitymodel.DensityModel(args.num_interactions, args.node_size, args.cutoff)

    logging.debug("model has %d parameters", count_parameters(net))

    # Setup optimizer
    schedulelr = nn.learning_rate_schedule.ExponentialDecayLR(learning_rate=1.0, decay_rate=0.96, decay_steps=100000)
    # learning_rate 或可改为0.0001
    optimizer = nn.Adam(net.trainable_params(), learning_rate=schedulelr)
    criterion = nn.MSELoss()

    loss_net = CustomWithLossCell(net, criterion)
    train_net = nn.TrainOneStepCell(loss_net, optimizer)

# ================================================================================================
# train

    log_interval = 5000
    running_loss = ms.tensor(0.0)
    running_loss_count = ms.tensor(0)
    best_val_mae = np.inf
    step = 0

    if args.load_model:
        param_dict = ms.load_checkpoint(args.load_model)
        ms.load_param_into_net(net, param_dict)

    logging.info("start training")

    data_timer = AverageMeter("data_timer")
    transfer_timer = AverageMeter("transfer_timer")
    train_timer = AverageMeter("train_timer")
    eval_timer = AverageMeter("eval_time")

    endtime = timeit.default_timer()

    #train_loader_list = flatten(list(train_loader))

    for _ in itertools.count():
        #for batch_host in train_loader_list:
        for batch in train_loader:
            batch32 = convert_batch_int32(batch[0])
            debuger.debug_devideline("REMARK BATCH")
            #debuger.print_structure(batch32)
            

            # 使用timeit作为计时方法
            data_timer.update(timeit.default_timer() - endtime)
            tstart = timeit.default_timer()

            transfer_timer.update(timeit.default_timer() - tstart)
            tstart = timeit.default_timer()

            #debuger.debug_devideline("REMARK GRADS")
            #debuger.debug_check_var(batch32, "batch32")
            loss = train_net(*list(batch32.values()))
            debuger.debug_check_var(loss, "loss")
            #debuger.debug_devideline("GRADS END")

            #debuger.debug_check_var(loss, "loss", ["shape", "type"])#Tensor [] Float32
            #debuger.debug_check_var(batch32["probe_target"], "batch32[\"probe_target\"]", ["shape", "dtype"])#Tensor [2] Float32
            #debuger.debug_check_var(batch32["probe_target"].shape, "batch32[\"probe_target\"].shape", ["shape", "dtype"])#tuple
            #debuger.debug_check_var(batch32["num_probes"], "batch32[\"num_probes\"]", ["shape", "dtype"])#Tensor [2] Int32

            running_loss += loss * batch32["probe_target"].shape[0] * batch32["probe_target"].shape[1]
            running_loss_count += ops.sum(batch32["num_probes"])

            #debuger.debug_check_var(running_loss, "running_loss", ["shape", "dtype"])#Tensor [] Float32
            #debuger.debug_check_var(running_loss_count, "running_loss_count", ["shape", "dtype"])#Tensor [] Int32


            train_timer.update(timeit.default_timer() - tstart)
            '''
            # Validate and save model
            if (step % log_interval == 0) or ((step + 1) == args.max_steps):
                tstart = timeit.default_timer()
                #                with torch.no_grad():
                #                    train_loss = (running_loss / running_loss_count).item()
                #                    running_loss = running_loss_count = 0
                train_loss = running_loss / running_loss_count
                running_loss = running_loss_count = 0

                debuger.debug_devideline("modle eval", capital=True)
                val_mae, val_rmse = eval_model(net, val_loader, args.device)
                debuger.debug_devideline("modle eval end", capital=True)

                logging.info(
                    "step=%d, val_mae=%g, val_rmse=%g, sqrt(train_loss)=%g",
                    step,
                    val_mae,
                    val_rmse,
                    math.sqrt(train_loss),
                )
            

                # Save checkpoint
                if val_mae < best_val_mae:
                    #debuger.debug_check_var(net.parameters_dict(), "net",repo=False)
                    #debuger.debug_check_var(optimizer.parameters_dict(), "opt",repo=False)
                    #debuger.debug_check_var(scheduler, "sche")
                    #debuger.debug_check_var(step, "step")
                    #debuger.debug_check_var(best_val_mae, "best_val")
                
                    ms.train.save_checkpoint(net, os.path.join(args.output_dir, "best_model.ckpt"))
                eval_timer.update(timeit.default_timer() - tstart)
                logging.debug("%s %s %s %s" % (data_timer, transfer_timer, train_timer, eval_timer))
                '''
            step += 1

            # scheduler.step()


            if step >= args.max_steps:
                logging.info("Max steps reached, exiting")
                sys.exit(0)

            endtime = timeit.default_timer()
            #debuger.debug_devideline("finish 1 batch")
            debuger.debug_devideline("REMARK BATCH END") 


if __name__ == "__main__":
    main()



