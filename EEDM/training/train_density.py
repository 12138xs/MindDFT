import sys
import os
import math
import numpy as np
import mindspore as ms
from mindspore import nn, ops, context
from mindspore.dataset import GeneratorDataset
from mindspore.train.callback import LossMonitor
from mindspore import Tensor
import mindspore.dataset as ds
from mindspore.common.initializer import HeUniform
from mindspore.train import Model
from mindspore import save_checkpoint
import argparse
from datetime import date
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_iso_permuted_dataset, get_scalar_density_comparisons

os.environ['GLOG_v'] = '3'
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU") # GRAPH_MODE


def loss_per_channel(y_ml, y_target, Rs=[(12, 0), (5, 1), (4, 2), (2, 3), (1, 4)]):
    err = y_ml - y_target
    pct_dev = ops.abs(err) / y_target
    loss_per_channel_list = np.zeros(len(Rs))
    normalization = ops.sum(err) / ops.mean(err)

    counter = 0
    for mul, l in Rs:
        if l == 0:
            temp_loss = ops.abs(ops.sum(err[:, :mul].pow(2))) / normalization
        else:
            temp_loss = ops.abs(ops.sum(err[:, counter:counter + mul * (2 * l + 1)].pow(2))) / normalization

        loss_per_channel_list[l] += temp_loss.asnumpy()
        #pct_deviation_list[l] += pct_dev.asnumpy()

        counter += mul * (2 * l + 1)

    return loss_per_channel_list


class Network(nn.Cell):
    def __init__(self, irreps_in, irreps_hidden, irreps_out, irreps_node_attr, irreps_edge_attr, layers, max_radius,
                 number_of_basis, radial_layers, radial_neurons, num_neighbors, num_nodes, reduce_output):
        super(Network, self).__init__()
        # Define your model architecture here using MindSpore's layers.
        self.layers = nn.SequentialCell([
            nn.Dense(2, irreps_hidden[0][0]),
            nn.ReLU(),
            nn.Dense(irreps_hidden[0][0], 70),
        ])

    def construct(self, data):
        x = data['x'].view(-1, 2)
        return self.layers(x).view(-1, 70)

def train_epoch(epoch, model, loss_fn, optimizer, data_loader, Rs, b):
    model.set_train()
    loss_cum = 0.0
    loss_perchannel_cum = np.zeros(len(Rs))
    mae_cum = 0.0
    mue_cum = 0.0

    # Define forward function
    def forward_fn(data, label):
        mask = ops.select(label == 0, ops.zeros_like(label), ops.ones_like(label))
        logits = model(data)*mask
        loss = loss_fn(logits, label)
        return loss, logits

    # Get gradient function
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    # Define function of one-step training
    def train_step(data, label):
        (loss, logits), grads = grad_fn(data, label)
        optimizer(grads)
        return loss, logits

    dataset_size = data_loader.get_dataset_size()
    for batch_idx, trainset in enumerate(data_loader.create_dict_iterator()):
        data = trainset['data']
        target = data['y']
        loss, logits = train_step(data, target)
        for mul, l in Rs:
            if l == 0:
                num_ele = ops.sum(logits[:, :mul])

        mue_cum += num_ele
        mae_cum += ops.abs(num_ele)
        loss_cum += loss

    print(f"{epoch} {float(loss_cum) / dataset_size:.10f}")
    print(f"----Train_Loss l=0 {float(loss_cum) / dataset_size}")
    print(f"    Train_MAE {mae_cum / (dataset_size * b)}")
    return loss_cum, mae_cum, mue_cum


def test_epoch(model, data_loader, loss_fn, Rs, epoch, save_interval, mue_cum, ldep_bool=False, density_spacing=0.1):
    model.set_train(False)
    metrics = []

    test_loss_cum = 0.0
    test_mae_cum = 0.0
    test_mue_cum = 0.0
    bigIs_cum = 0.0
    eps_cum = 0.0
    ep_per_l_cum = np.zeros(len(Rs))
    ele_diff_cum = 0.0

    dataset_size = data_loader.get_dataset_size()
    for batch_idx, testset in enumerate(data_loader.create_dict_iterator()):
        data = testset["data"]
        label = data['y']
        mask = ops.select(label == 0, ops.zeros_like(label), ops.ones_like(label))
        logits = model(data) * mask
        err = logits - label

        for mul, l in Rs:
            if l == 0:
                num_ele = ops.mean(logits[:,:mul])
                # num_ele = sum(sum(logits[:, :mul]))

        test_mue_cum += num_ele
        test_mae_cum += abs(num_ele)
        test_loss_cum += err.pow(2).mean().abs()

        if ldep_bool: 
            num_ele_target, num_ele_ml, bigI, ep, ep_per_l = get_scalar_density_comparisons(data, logits, Rs, spacing=density_spacing, buffer=3.0, ldep=ldep_bool)
            ep_per_l_cum += ep_per_l
        else:
            num_ele_target, num_ele_ml, bigI, ep = get_scalar_density_comparisons(data, logits, Rs, spacing=density_spacing, buffer=3.0, ldep=ldep_bool)

        n_ele = np.sum(data['z'].asnumpy())
        ele_diff_cum += np.abs(n_ele-num_ele_target)
        bigIs_cum += bigI
        eps_cum += ep

        metrics.append([test_loss_cum, test_mae_cum, test_mue_cum, ele_diff_cum, bigIs_cum, eps_cum, ep_per_l_cum])

    print("----Test_Loss",float(metrics[0][0])/dataset_size)
    print("    Test_MAE",metrics[0][1].item()/dataset_size)
    print("    Test_MUE",metrics[0][2].item()/dataset_size)
    print("    Test_Electron_Difference",metrics[0][3].item()/dataset_size)
    print("    Test_big_I",metrics[0][4].item()/dataset_size)
    print("    Test_Epsilon",metrics[0][5].item()/dataset_size)
    print("    Test_Epsilon l=0",metrics[0][-1][0].item()/dataset_size)
    print("    Test_Epsilon l=1",metrics[0][-1][1].item()/dataset_size)
    print("    Test_Epsilon l=2",metrics[0][-1][2].item()/dataset_size)  
    print("    Test_Epsilon l=3",metrics[0][-1][3].item()/dataset_size)
    print("    Test_Epsilon l=4",metrics[0][-1][4].item()/dataset_size)
    if epoch % save_interval == 0:
        save_checkpoint(model, os.path.join("./checkpoints/model_weights_epoch_" + str(epoch) + ".ckpt"))



def main():
    parser = argparse.ArgumentParser(description='train electron density')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--testset', type=str)
    parser.add_argument('--split', type=int)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--qm', type=str, default="pbe0")
    parser.add_argument('--ldep', type=bool, default=False)
    args = parser.parse_args()

    if args.qm == 'ccsd':
        hhh = os.path.dirname(
            os.path.realpath(__file__)) + "/../data/ccsd_h_s_only_def2-universal-jfit-decontract_density.out"
        ooo = os.path.dirname(
            os.path.realpath(__file__)) + "/../data/ccsd_o_s_only_def2-universal-jfit-decontract_density.out"
    else:
        hhh = os.path.dirname(
            os.path.realpath(__file__)) + "/../data/h_s_only_def2-universal-jfit-decontract_density.out"
        ooo = os.path.dirname(
            os.path.realpath(__file__)) + "/../data/o_s_only_def2-universal-jfit-decontract_density.out"

    test_dataset = args.testset
    num_epochs = args.epochs
    ldep_bool = args.ldep

    Rs = [(12, 0), (5, 1), (4, 2), (2, 3), (1, 4)]

    test_dataset = get_iso_permuted_dataset(args.testset, o_iso=ooo, h_iso=hhh)

    split = args.split
    data_file = args.dataset
    lr = 1e-2
    density_spacing = 0.1
    save_interval = 5
    model_kwargs = {
        "irreps_in": 2,
        "irreps_hidden": [(mul, (l, p)) for l, mul in enumerate([125, 40, 25, 15]) for p in [-1, 1]],
        "irreps_out": 12,
        "irreps_node_attr": None,
        "irreps_edge_attr": 3,
        "layers": 3,
        "max_radius": 3.5,
        "number_of_basis": 10,
        "radial_layers": 1,
        "radial_neurons": 128,
        "num_neighbors": 12.2298,
        "num_nodes": 24,
        "reduce_output": False,
    }

    dataset = get_iso_permuted_dataset(data_file, o_iso=ooo, h_iso=hhh)
    random.shuffle(dataset)
    if split > len(dataset):
        raise ValueError('Split is too large for the dataset.')

    b = 1

    def data_generator(data):
        for d in data:
            yield d

    train_loader = GeneratorDataset(lambda: data_generator(dataset[:split]), ["data"], shuffle=True)
    test_loader = GeneratorDataset(lambda: data_generator(test_dataset), ["data"], shuffle=True)

    train_loader = train_loader.batch(b)
    test_loader = test_loader.batch(b)

    model = Network(**model_kwargs)
    optim = nn.Adam(model.trainable_params(), learning_rate=lr)
    loss_fn = nn.MSELoss()

    model_kwargs["train_dataset"] = data_file
    model_kwargs["train_dataset_size"] = split
    model_kwargs["lr"] = lr
    model_kwargs["density_spacing"] = density_spacing

    for epoch in range(num_epochs):
        loss_cum, mae_cum, mue_cum = train_epoch(epoch, model, loss_fn, optim, train_loader, Rs, b)
        test_epoch(model, test_loader, loss_fn, Rs, epoch, save_interval, mue_cum, ldep_bool, density_spacing)

if __name__ == '__main__':
    main()
