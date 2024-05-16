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

import densitymodel
import dataset

# 这个函数是用来调试的,用来查看数据结构
def print_structure(var, indent=0):  
    if isinstance(var, dict):  
        print(' ' * indent + '{')  
        for key in var:  
            print(' ' * (indent + 2) + 'key:' + key, end=' ')  
            print_structure(var[key], indent + 2)  
        print(' ' * indent + '}')  
    elif isinstance(var, list):  
        print(' ' * indent + '[')  
        for item in var:  
            print_structure(item, indent + 2)  
        print(' ' * indent + ']')  
    else:  
        print(' ' * indent + type(var).__name__)

def structure(var, indent=0):  
    output = ""  
    if isinstance(var, dict):  
        output += ' ' * indent + "{\n"  
        for key in var:  
            output += ' ' * (indent + 2) + "key:" + key + " "  
            output += structure(var[key], indent + 2)  
        output += ' ' * indent + "}\n"  
    elif isinstance(var, list):  
        output += ' ' * indent + "[\n"  
        for item in var:  
            output += structure(item, indent + 2)  
        output += ' ' * indent + "]\n"  
    else:  
        output += ' ' * indent + type(var).__name__ + "\n"  
    return output

# 用于将list(train_loader)转化成单层列表
def flatten(lst):  
    result = []  
    for i in lst:  
        if isinstance(i, list):  
            result.extend(flatten(i))  
        else:  
            result.append(i)  
    return result

# 用来实现LambdaLR
def scheduler_fn(step):
    lr = 0.96 ** (step / 100000)
    return lr

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Train graph convolution network", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="Load model parameters from previous run",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=5.0,
        help="Atomic interaction cutoff distance [Å]",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="Train/test/validation split file json",
    )
    parser.add_argument(
        "--num_interactions",
        type=int,
        default=3,
        help="Number of interaction layers used",
    )
    parser.add_argument(
        "--node_size", type=int, default=64, help="Size of hidden node states"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/model_output",
        help="Path to output directory",
    )
    parser.add_argument(
        "--dataset", type=str, default="./data/qm9", help="Path to ASE database",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=int(1e6),
        help="Maximum number of optimisation steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Set which device to use for training e.g. 'cuda' or 'cpu'",
    )

    parser.add_argument(
        "--use_painn_model",
        action="store_true",
        help="Enable equivariant message passing model (PaiNN)"
    )

    parser.add_argument(
        "--ignore_pbc",
        action="store_true",
        help="If flag is given, disable periodic boundary conditions (force to False) in atoms data",
    )

    parser.add_argument(
        "--force_pbc",
        action="store_true",
        help="If flag is given, force periodic boundary conditions to True in atoms data",
    )

    return parser.parse_args(arg_list)


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


def split_data(dataset, args):
    # Load or generate splits
    if args.split_file:
        with open(args.split_file, "r") as fp:
            splits = json.load(fp)
    else:
        datalen = len(dataset)
        num_validation = int(math.ceil(datalen * 0.05))
        indices = np.random.permutation(len(dataset))
        splits = {
            "train": indices[num_validation:].tolist(),
            "validation": indices[:num_validation].tolist(),
        }

        # Save split file
        with open(os.path.join(args.output_dir, "datasplits.json"), "w") as f:
            json.dump(splits, f)

    # Split the dataset
    datasplits = {}
    for key, indices in splits.items():
        datasplits[key] = dataset.take(indices)
        #datasplits[key] = [dataset[idx] for idx in indices]
    return datasplits


# def eval_model(model, dataloader, device):
#    with torch.no_grad():
#        running_ae = torch.tensor(0., device=device)
#        running_se = torch.tensor(0., device=device)
#        running_count = torch.tensor(0., device=device)
#        for batch in dataloader:
#            device_batch = {
#                k: v.to(device=device, non_blocking=True) for k, v in batch.items()
#            }
#            outputs = model(device_batch)
#            targets = device_batch["probe_target"]
#
#            running_ae += torch.sum(torch.abs(targets - outputs))
#            running_se += torch.sum(torch.square(targets - outputs))
#            running_count += torch.sum(device_batch["num_probes"])
#
#        mae = (running_ae / running_count).item()
#        rmse = (torch.sqrt(running_se / running_count)).item()

#    return mae, rmse
def convert_batch_int32(batch:dict):
    batch32 = {}  
    for key, value in batch.items():  
        if isinstance(value, ms.Tensor) and value.dtype == mstype.int64:   
            converted_value = ms.Tensor(value.asnumpy().astype('int32'), mstype.int32)  
                    #key: [nodes, num_nodes, num_atom_edges, num_probe_edges, num_probes]
        else:   
            converted_value = value
        batch32[key] = converted_value
    return batch32

# calculate mse rmse
def eval_model(model, dataloader):
    running_ae = ms.Tensor(0., dtype=ms.float32)
    running_se = ms.Tensor(0., dtype=ms.float32)
    running_count = ms.Tensor(0., dtype=ms.float32)

    for batch in dataloader:
        print("=============REMARK EVALMODEL============")
        print_structure(batch)
        batch32 = {}  
        for key, value in batch.items():  
            if isinstance(value, ms.Tensor) and value.dtype == mstype.int64:   
                converted_value = ms.Tensor(value.asnumpy().astype('int32'), mstype.int32)  
            else:   
                converted_value = value  
            batch32[key] = converted_value
        outputs = model(batch32)
        print(outputs.dtype)
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

    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
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

    # Save command line args
    with open(os.path.join(args.output_dir, "commandline_args.txt"), "w") as f:
        f.write("\n".join(sys.argv[1:]))
    # Save parsed command line arguments
    with open(os.path.join(args.output_dir, "arguments.json"), "w") as f:
        json.dump(vars(args), f)

    # Setup dataset and loader
    if args.dataset.endswith(".txt"):
        # Text file contains list of datafiles
        with open(args.dataset, "r") as datasetfiles:
            filelist = [os.path.join(os.path.dirname(args.dataset), line.strip('\n')) for line in datasetfiles]
    else:
        filelist = [args.dataset]

    logging.info("loading data %s", args.dataset)
    #    densitydata = torch.utils.data.ConcatDataset([dataset.DensityData(path) for path in filelist])
    #densitydata = dataset.DensityData(filelist[0])
    #combined_data = dataset.DensityData(filelist[0])
    #print(combined_data[0]['atoms'])
    #densitydata = []
    #i = 0
    #length_data = len(combined_data)
    #for file in filelist:
    #    len_file = len(filelist)
    #    combined_data = dataset.DensityData(file)
    #    for single_densitydata in combined_data:
    #        densitydata.append(single_densitydata)
    #        print(i, " / ", 20*len_file)
    #        i+=1
    #        if i>20:
    #            break
    
    for i in range(0, len(filelist)):
        if i==0:
            densitydata = dataset.DensityData(filelist[0])
        else:
            densitydata.concat(dataset.DensityData(filelist[i]))

    #print(densitydata[0]['atoms'])
    # densitydata = ms.dataset.ConcatDataset([dataset.DensityData(path) for path in filelist])

    # Split data into train and validation sets
    datasplits = split_data(densitydata, args)

    datasplits["train"] = dataset.RotatingPoolData(datasplits["train"], 20)
    # 已经查明，RotatingDataSet是可以迭代的，继续向下排查

    if args.ignore_pbc and args.force_pbc:
        raise ValueError("ignore_pbc and force_pbc are mutually exclusive and can't both be set at the same time")
    elif args.ignore_pbc:
        set_pbc = False
    elif args.force_pbc:
        set_pbc = True
    else:
        set_pbc = None

    # Setup loaders
    #  train_loader = torch.utils.data.DataLoader(
    #      datasplits["train"],
    #      2,
    #      num_workers=4,
    #      sampler=torch.utils.data.RandomSampler(datasplits["train"]),
    #      collate_fn=dataset.CollateFuncRandomSample(args.cutoff, 1000, pin_memory=False, set_pbc_to=set_pbc),
    #  )
    
    train_loader = ms.dataset.GeneratorDataset(
        datasplits["train"],
        #  collate_fn, batch_size通过mindspore.dataset.batch 操作支持
        num_parallel_workers=4,
        # sampler=ms.dataset.RandomSampler(datasplits["train"])
        column_names=["train"],
        sampler = ms.dataset.RandomSampler(num_samples=len(datasplits["train"]), replacement=False)
    )
    train_loader = train_loader.map(operations=dataset.CollateFuncRandomSample(args.cutoff, 1000, set_pbc_to=set_pbc),
                                    input_columns=["train"])
    train_loader = train_loader.batch(batch_size=1, drop_remainder=True)
    train_loader = train_loader.repeat(count=3)

    # 结论：train_loader就是不可迭代的
    # 4/10 train_loader可以通过list(train_loader)转换成可迭代的。

    val_loader = ms.dataset.GeneratorDataset(
        datasplits["validation"],
        num_parallel_workers=4,
        column_names=["validations"],
        sampler=ms.dataset.RandomSampler(num_samples=len(datasplits["validation"]), replacement=False)
    )
    val_loader = val_loader.map(operations=dataset.CollateFuncRandomSample(args.cutoff, 5000, set_pbc_to=set_pbc),
                                input_columns=["validations"])
    val_loader = val_loader.batch(batch_size=1, drop_remainder=True)
    logging.info("Preloading validation batch")

    # Initialise model
    ### 这里使用静态图，CPU
    #ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target='CPU')
    # 后面静态图跑不通，这里先用动态图试试
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target='CPU')

    # net = net.to(device)
    if args.use_painn_model:
        net = densitymodel.PainnDensityModel(args.num_interactions, args.node_size, args.cutoff)
    else:
        net = densitymodel.DensityModel(args.num_interactions, args.node_size, args.cutoff)
    logging.debug("model has %d parameters", count_parameters(net))

    # Setup optimizer
    schedulelr = nn.learning_rate_schedule.ExponentialDecayLR(learning_rate=1.0, decay_rate=0.96, decay_steps=100000)
    optimizer = nn.Adam(net.trainable_params(), learning_rate=schedulelr)
    criterion = nn.MSELoss()
    loss_net = nn.WithLossCell(net, criterion)
    train_net = nn.TrainOneStepCell(loss_net, optimizer)

    log_interval = 5000
    running_loss = ms.tensor(0.0)
    running_loss_count = ms.tensor(0)
    best_val_mae = np.inf
    step = 0
    # Restore checkpoint
    #    if args.load_model:
    #        state_dict = torch.load(args.load_model)
    #        net.load_state_dict(state_dict["model"])
    #        step = state_dict["step"]
    #        best_val_mae = state_dict["best_val_mae"]
    #        optimizer.load_state_dict(state_dict["optimizer"])
    #        scheduler.load_state_dict(state_dict["scheduler"])
    # 这里使用ms自带的check_point
    if args.load_model:
        param_dict = ms.load_checkpoint(args.load_model)
        ms.load_param_into_net(net, param_dict)

    logging.info("start training")

    data_timer = AverageMeter("data_timer")
    transfer_timer = AverageMeter("transfer_timer")
    train_timer = AverageMeter("train_timer")
    eval_timer = AverageMeter("eval_time")

    endtime = timeit.default_timer()

    train_loader_list = flatten(list(train_loader))
    train_loader_list = [convert_batch_int32(batch) for batch in train_loader_list]
    val_loader_list = flatten(list(val_loader))
    val_loader_list = [convert_batch_int32(batch) for batch in val_loader_list]
    print(len(train_loader_list), len(val_loader_list))

    # mindspore需要定义梯度操作
    grad_op = ops.GradOperation(get_all=True) 

    for _ in itertools.count():
        #for batch_host in train_loader_list:
        for batch32 in train_loader_list:

            # 使用timeit作为计时方法
            data_timer.update(timeit.default_timer() - endtime)
            tstart = timeit.default_timer()

            # Transfer to 'device'
            # ms不需要将batch_host迁移至batch
            #batch = {
            #    k: v.to(device=device, non_blocking=True)
            #    for (k, v) in batch_host.items()
            #}
            transfer_timer.update(timeit.default_timer() - tstart)
            tstart = timeit.default_timer()

            # Reset gradient
            # ms通常不需要手动清零
            # optimizer.zero_grad()

            # Forward, backward and optimize
            loss = train_net(batch32, batch32["probe_target"])

            #with torch.no_grad():
            #    running_loss += loss * batch["probe_target"].shape[0] * batch["probe_target"].shape[1]
            #    running_loss_count += torch.sum(batch["num_probes"])
            running_loss += ops.mul(loss, ops.mul(batch32["probe_target"].shape[0], batch32["probe_target"].shape[1]))
            running_loss_count += ops.sum(batch32["num_probes"])

            train_timer.update(timeit.default_timer() - tstart)
            print("finished training, now validating.")

            # Validate and save model
            if (step % log_interval == 0) or ((step + 1) == args.max_steps):
                tstart = timeit.default_timer()
                #with torch.no_grad():
                #    train_loss = (running_loss / running_loss_count).item()
                #    running_loss = running_loss_count = 0
                train_loss = running_loss / running_loss_count
                running_loss = 0
                running_loss_count = 0
                val_mae, val_rmse = eval_model(net, val_loader_list)

                logging.info(
                    "step=%d, val_mae=%g, val_rmse=%g, sqrt(train_loss)=%g",
                    step,
                    val_mae,
                    val_rmse,
                    math.sqrt(train_loss),
                )

                # Save checkpoint
                if val_mae < best_val_mae:
                    # best_val_mae = val_mae
                    # torch.save(
                    #    {
                    #        "model": net.state_dict(),
                    #         "optimizer": optimizer.state_dict(),
                    #         "scheduler": scheduler.state_dict(),
                    #         "step": step,
                    #         "best_val_mae": best_val_mae,
                    #     },
                    #     os.path.join(args.output_dir, "best_model.pth"),
                    # )
                    ms.train.serialization.save_checkpoint(
                        net,
                        os.path.join(args.output_dir, "best_model.ckpt"),
                    )
                eval_timer.update(timeit.default_timer() - tstart)
                logging.debug(
                    "%s %s %s %s" % (data_timer, transfer_timer, train_timer, eval_timer)
                )
            step += 1

            # scheduler.step()
            # ms会自动更新scheduler,不需要手动更新。

            if step >= args.max_steps:
                logging.info("Max steps reached, exiting")
                sys.exit(0)

            endtime = timeit.default_timer()


if __name__ == "__main__":
    main()



