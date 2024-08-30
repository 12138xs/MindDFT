import sys
import os
import math
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
from mindspore.dataset import GeneratorDataset
from mindspore.train import Model
from mindspore.train.callback import LossMonitor
from mindspore.common.initializer import XavierUniform
import argparse
from datetime import date
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_iso_dataset


# 设置 MindSpore 运行模式和设备
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU") # GRAPH_MODE

def train_epoch(epoch, model, loss_fn, optimizer, data_loader, b):
    model.set_train(True)
    loss_cum = 0.0
    e_mae = 0.0
    e_mue = 0.0
    f_mae = 0.0

    # Define forward function
    def forward_fn(data, label):
        energy_label, forces_label = label
        energy_pred = model(data, data['pos'])

        # get ml force
        forces_pred = ms.grad(model, 1)(data, data['pos'])
        
        # subtract energy of water monomers (PBE0)
        monomer_energy = Tensor(-76.379999960410643, ms.float32) 
        energy_loss = loss_fn(energy_pred, (energy_label - monomer_energy * data['pos'].shape[0]/3))
        force_loss = loss_fn(forces_pred, forces_label)
        loss = energy_loss + force_loss
        return loss, energy_pred # [energy_pred, forces_pred]

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
        target = [data['energy'], data['forces']]
        loss, logits = train_step(data, target)
        
        # energy_pred, forces_pred = logits
        # monomer_energy = Tensor(-76.379999960410643, ms.float32)
        # energy_err = energy_pred - (data['energy'] - monomer_energy * data['pos'].shape[0]/3)
        # forces_err = forces_pred - data['forces']
        # e_mue += ops.ReduceSum()(energy_err)
        # e_mae += ops.ReduceSum()(ops.Abs()(energy_err))
        # f_mae += ops.ReduceMean()(ops.Abs()(forces_err.flatten()))
        loss_cum += loss

    print(f"{epoch} {float(loss_cum) / dataset_size:.10f}")
    print(f"----Train_Loss l=0 {float(loss_cum) / dataset_size}")
    print(f"    Train_Energy_MAE {e_mae / (dataset_size * b)}")
    print(f"    Train_Energy_MUE {e_mue / (dataset_size * b)}")
    print(f"    Train_Forces_MAE {f_mae / (dataset_size * b)}")

def test_epoch(epoch, model, test_loader):
    model.set_train(False)
    test_loss_cum = 0.0
    test_e_mae = 0.0
    test_e_mue = 0.0
    test_f_mae = 0.0
    monomer_energy = Tensor(-76.379999960410643, ms.float32)

    dataset_size = test_loader.get_dataset_size()
    for data in test_loader.create_dict_iterator():
        pos, energy, forces = data['pos'], data['energy'], data['forces']

        y_ml = model(pos, data['pos'])
        forces_pred = ops.GradOperation()(y_ml, pos)

        energy_err = y_ml - (energy - monomer_energy * pos.shape[0] / 3)
        forces_err = forces_pred - forces

        test_e_mue += ops.ReduceSum()(energy_err)
        test_e_mae += ops.ReduceSum()(ops.Abs()(energy_err))
        test_f_mae += ops.ReduceMean()(ops.Abs()(forces_err.flatten()))

        energy_loss = ops.ReduceMean()(energy_err ** 2)
        force_loss = ops.ReduceMean()(forces_err ** 2)

        test_loss_cum += (energy_loss + force_loss).asnumpy()

    if epoch % 1 == 0:
        print("    Test_Loss: ", float(test_loss_cum) / dataset_size)
        print("    Test_Energy_MAE: ", float(test_e_mae) / dataset_size)
        print("    Test_Energy_MUE: ", float(test_e_mue) / dataset_size)
        print("    Test_Forces_MAE: ", float(test_f_mae) / dataset_size)


# 模型定义
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

    def construct(self, data, pos):
        x = data['x'].view(-1, 2)
        return self.layers(x).view(-1, 70)


# 主函数
def main():
    parser = argparse.ArgumentParser(description='Train energy and force model with MindSpore')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--testset', type=str, required=True)
    parser.add_argument('--split', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--gpu', type=str, default="0")
    args = parser.parse_args()

    # 设置设备
    device_id = int(args.gpu)
    context.set_context(device_id=device_id)

    # 数据加载
    hhh = "../data/h_s_only_def2-universal-jfit-decontract_density.out"
    ooo = "../data/o_s_only_def2-universal-jfit-decontract_density.out"

    split = args.split
    train_dataset = get_iso_dataset(args.dataset, o_iso=ooo, h_iso=hhh)
    test_dataset = get_iso_dataset(args.testset, o_iso=ooo, h_iso=hhh)
    random.shuffle(train_dataset)
    if split > len(train_dataset):
        raise ValueError('Split is too large for the dataset.')
    
    def data_generator(data):
        for d in data:
            yield d

    train_loader = GeneratorDataset(lambda: data_generator(train_dataset[:split]), ["data"], shuffle=True)
    test_loader = GeneratorDataset(lambda: data_generator(test_dataset), ["data"], shuffle=True)

    b = 1
    train_loader = train_loader.batch(b)
    test_loader = test_loader.batch(b)

    # 模型参数
    model_kwargs = {
        "irreps_in": 2,  # 例如：输入维度
        "irreps_hidden": [(mul, (l, p)) for l, mul in enumerate([125, 40, 25, 15]) for p in [-1, 1]],
        "irreps_out": 1,  # 例如：输出维度
        "irreps_node_attr": None,
        "irreps_edge_attr": 3,
        "layers": 3,
        "max_radius": 3.5,
        "number_of_basis": 10,
        "radial_layers": 1,
        "radial_neurons": 128,
        "num_neighbors": 12.2298,
        "num_nodes": 24,
        "reduce_output": True,
    }

    model = Network(**model_kwargs)

    # 优化器和损失函数
    optim = nn.Adam(model.trainable_params(), learning_rate=1e-2)
    loss_fn = nn.MSELoss()

    # 训练循环
    for epoch in range(args.epochs):
        train_epoch(epoch, model, loss_fn, optim, train_loader, b)
        test_epoch(epoch, model, loss_fn, test_loader, b)
        # model.set_train(True)
        # loss_cum = 0.0
        # e_mae = 0.0
        # e_mue = 0.0
        # f_mae = 0.0

        # for data in train_loader.create_dict_iterator():
        #     pos, energy, forces = data['pos'], data['energy'], data['forces']

        #     y_ml = model(pos)
        #     forces_pred = ops.GradOperation()(y_ml, pos)

        #     monomer_energy = Tensor(-76.379999960410643, ms.float32)
        #     energy_err = y_ml - (energy - monomer_energy * pos.shape[0] / 3)
        #     forces_err = forces_pred - forces

        #     e_mue += ops.ReduceSum()(energy_err)
        #     e_mae += ops.ReduceSum()(ops.Abs()(energy_err))
        #     f_mae += ops.ReduceMean()(ops.Abs()(forces_err.flatten()))

        #     energy_loss = ops.ReduceMean()(energy_err ** 2)
        #     force_loss = ops.ReduceMean()(forces_err ** 2)

        #     total_loss = energy_loss + force_loss
        #     total_loss.backward()
        #     optim.step()
        #     optim.zero_grad()

        #     loss_cum += total_loss.asnumpy()

        # # 测试循环
        # model.set_train(False)
        # test_loss_cum = 0.0
        # test_e_mae = 0.0
        # test_e_mue = 0.0
        # test_f_mae = 0.0

        # for data in test_loader.create_dict_iterator():
        #     pos, energy, forces = data['pos'], data['energy'], data['forces']

        #     y_ml = model(pos)
        #     forces_pred = ops.GradOperation()(y_ml, pos)

        #     energy_err = y_ml - (energy - monomer_energy * pos.shape[0] / 3)
        #     forces_err = forces_pred - forces

        #     test_e_mue += ops.ReduceSum()(energy_err)
        #     test_e_mae += ops.ReduceSum()(ops.Abs()(energy_err))
        #     test_f_mae += ops.ReduceMean()(ops.Abs()(forces_err.flatten()))

        #     energy_loss = ops.ReduceMean()(energy_err ** 2)
        #     force_loss = ops.ReduceMean()(forces_err ** 2)

        #     test_loss_cum += (energy_loss + force_loss).asnumpy()

        # if epoch % 1 == 0:
        #     print(f"Epoch {epoch} Loss: {float(loss_cum) / len(train_loader):.10f}")


if __name__ == '__main__':
    main()
