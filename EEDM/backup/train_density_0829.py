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


import numpy as np
from mindspore import Tensor, ops
def gau2grid_density_kdtree(x, y, z, data, ml_y, rs, ldepb=False):
    import numpy as np
    import gau2grid as g2g
    from scipy import spatial

    xyz = np.vstack([x, y, z])
    tree = spatial.cKDTree(xyz.T)

    ml_density = np.zeros_like(x)
    target_density = np.zeros_like(x)

    # l-indexed arrays to dump specific contributions to density
    ml_density_per_l = np.array([np.zeros_like(x) for _ in range(len(rs))])
    target_density_per_l = np.array([np.zeros_like(x) for _ in range(len(rs))])

    for coords, full_coeffs, iso_coeffs, ml_coeffs, alpha, norm in zip(
        data.pos_orig.asnumpy(), data.full_c.asnumpy(), data.iso_c.asnumpy(), ml_y.asnumpy(),
        data.exp.asnumpy(), data.norm.asnumpy()
    ):
        center = coords
        counter = 0
        for mul, l in rs:
            for j in range(mul):
                normal = norm[counter]
                if normal != 0:
                    exp = [alpha[counter]]

                    small = 1e-5
                    angstrom2bohr = 1.8897259886
                    bohr2angstrom = 1 / angstrom2bohr

                    target_full_coeffs = full_coeffs[counter:counter + (2 * l + 1)]

                    pop_ml = ml_coeffs[counter:counter + (2 * l + 1)]
                    c_ml = pop_ml * normal / (2 * np.sqrt(2))
                    ml_full_coeffs = c_ml + iso_coeffs[counter:counter + (2 * l + 1)]

                    target_max = np.amax(np.abs(target_full_coeffs))
                    ml_max = np.amax(np.abs(ml_full_coeffs))
                    max_c = np.amax(np.array([target_max, ml_max]))

                    cutoff = np.sqrt((-1 / exp[0]) * np.log(small / np.abs(max_c * normal))) * bohr2angstrom

                    close_indices = tree.query_ball_point(center, cutoff)
                    points = np.require(xyz[:, close_indices], requirements=['C', 'A'])

                    ret_target = g2g.collocation(points * angstrom2bohr, l, [1], exp, center * angstrom2bohr)
                    ret_ml = g2g.collocation(points * angstrom2bohr, l, [1], exp, center * angstrom2bohr)

                    # Permute back to psi4 ordering
                    psi4_2_e3nn = [[0], [2, 0, 1], [4, 2, 0, 1, 3], [6, 4, 2, 0, 1, 3, 5],
                                   [8, 6, 4, 2, 0, 1, 3, 5, 7], [10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9],
                                   [12, 10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9, 11]]
                    e3nn_2_psi4 = [[0], [1, 2, 0], [2, 3, 1, 4, 0], [3, 4, 2, 5, 1, 6, 0],
                                   [4, 5, 3, 6, 2, 7, 1, 8, 0], [5, 6, 4, 7, 3, 8, 2, 9, 1, 10, 0],
                                   [6, 7, 5, 8, 4, 9, 3, 10, 2, 11, 1, 12, 0]]

                    target_full_coeffs = np.array([target_full_coeffs[k] for k in e3nn_2_psi4[l]])
                    ml_full_coeffs = np.array([ml_full_coeffs[k] for k in e3nn_2_psi4[l]])

                    scaled_components = (target_full_coeffs * normal * ret_target["PHI"].T).T
                    target_tot = np.sum(scaled_components, axis=0)

                    ml_scaled_components = (ml_full_coeffs * normal * ret_target["PHI"].T).T
                    ml_tot = np.sum(ml_scaled_components, axis=0)

                    target_density[close_indices] += target_tot
                    ml_density[close_indices] += ml_tot

                    # Dump l-dependent contributions
                    target_density_per_l[l][close_indices] += target_tot
                    ml_density_per_l[l][close_indices] += ml_tot

                counter += 2 * l + 1

    if ldepb:
        return target_density, ml_density, target_density_per_l, ml_density_per_l
    else:
        return target_density, ml_density


import numpy as np


def find_min_max(coords):
    xmin, xmax = coords[0, 0], coords[0, 0]
    ymin, ymax = coords[0, 1], coords[0, 1]
    zmin, zmax = coords[0, 2], coords[0, 2]

    for coord in coords:
        if coord[0] < xmin:
            xmin = coord[0]
        if coord[0] > xmax:
            xmax = coord[0]
        if coord[1] < ymin:
            ymin = coord[1]
        if coord[1] > ymax:
            ymax = coord[1]
        if coord[2] < zmin:
            zmin = coord[2]
        if coord[2] > zmax:
            zmax = coord[2]

    return xmin, xmax, ymin, ymax, zmin, zmax


def generate_grid(data, spacing=0.5, buffer=2.0):
    buf = buffer
    xmin, xmax, ymin, ymax, zmin, zmax = find_min_max(data.pos_orig.asnumpy())

    x_points = int((xmax - xmin + 2 * buf) / spacing) + 1
    y_points = int((ymax - ymin + 2 * buf) / spacing) + 1
    z_points = int((zmax - zmin + 2 * buf) / spacing) + 1
    npoints = int((x_points + y_points + z_points) / 3)

    xlin = np.linspace(xmin - buf, xmax + buf, npoints)
    ylin = np.linspace(ymin - buf, ymax + buf, npoints)
    zlin = np.linspace(zmin - buf, zmax + buf, npoints)

    x_spacing = xlin[1] - xlin[0]
    y_spacing = ylin[1] - ylin[0]
    z_spacing = zlin[1] - zlin[0]
    vol = x_spacing * y_spacing * z_spacing

    # 使用 'ij' 索引模式生成网格
    x, y, z = np.meshgrid(xlin, ylin, zlin, indexing='ij')

    return x, y, z, vol, x_spacing, y_spacing, z_spacing

from copy import deepcopy, copy
import math
import pickle
import numpy as np
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import Tensor


def get_iso_permuted_dataset(picklefile, **atm_iso):
    dataset = []

    for key, value in atm_iso.items():
        if key == 'h_iso':
            h_data = Tensor(np.loadtxt(value, skiprows=2, usecols=1), dtype=ms.float32)
        elif key == 'c_iso':
            c_data = Tensor(np.loadtxt(value, skiprows=2, usecols=1), dtype=ms.float32)
        elif key == 'n_iso':
            n_data = Tensor(np.loadtxt(value, skiprows=2, usecols=1), dtype=ms.float32)
        elif key == 'o_iso':
            o_data = Tensor(np.loadtxt(value, skiprows=2, usecols=1), dtype=ms.float32)
        elif key == 'p_iso':
            p_data = Tensor(np.loadtxt(value, skiprows=2, usecols=1), dtype=ms.float32)
        else:
            raise ValueError("Isolated atom type not found. Use kwargs \"h_iso\", \"c_iso\", etc.")

    with open(picklefile, "rb") as f:
        molecules = pickle.load(f)

    cnt = 0
    for molecule in molecules:
        # Load data
        # pos = Tensor(molecule['pos'], dtype=ms.float32)
        # z = Tensor(molecule['type'].unsqueeze(1), dtype=ms.float32)
        # x = Tensor(molecule['onehot'], dtype=ms.float32)
        # c = Tensor(molecule['coefficients'], dtype=ms.float32)
        # n = Tensor(molecule['norms'], dtype=ms.float32)
        # exp = Tensor(molecule['exponents'], dtype=ms.float32)
        # full_c = copy.deepcopy(c)
        # iso_c = Tensor(np.zeros_like(c.asnumpy()), dtype=ms.float32)
        # Load from numpy arrays
        pos = Tensor(molecule['pos'], dtype=ms.float32)
        z = Tensor(np.expand_dims(molecule['type'], axis=1), dtype=ms.float32)
        x = Tensor(molecule['onehot'], dtype=ms.float32)
        c = Tensor(molecule['coefficients'], dtype=ms.float32)
        n = Tensor(molecule['norms'], dtype=ms.float32)
        exp = Tensor(molecule['exponents'], dtype=ms.float32)
        full_c = ops.deepcopy(c)
        iso_c = Tensor(np.zeros_like(c.asnumpy()), dtype=ms.float32)

        # Subtract the isolated atoms
        for atom, iso, typ in zip(c, iso_c, z):
            typ_value = typ.asnumpy().item()
            if typ_value == 1.0:
                atom[:h_data.shape[0]] -= h_data
                iso[:h_data.shape[0]] += h_data
            elif typ_value == 6.0:
                atom[:c_data.shape[0]] -= c_data
                iso[:c_data.shape[0]] += c_data
            elif typ_value == 7.0:
                atom[:n_data.shape[0]] -= n_data
                iso[:n_data.shape[0]] += n_data
            elif typ_value == 8.0:
                atom[:o_data.shape[0]] -= o_data
                iso[:o_data.shape[0]] += o_data
            elif typ_value == 15.0:
                atom[:p_data.shape[0]] -= p_data
                iso[:p_data.shape[0]] += p_data
            else:
                raise ValueError("Isolated atom type not supported!")

        pop = mnp.where(n != 0, c * 2 * math.sqrt(2) / n, n)

        # Permute positions, yzx -> xyz
        # p_pos = copy.deepcopy(pos)
        p_pos = ops.deepcopy(pos)
        p_pos[:, 0] = pos[:, 1]
        p_pos[:, 1] = pos[:, 2]
        p_pos[:, 2] = pos[:, 0]

        # Create dataset dictionary
        data_dict = {
            'pos': p_pos,
            'pos_orig': pos,
            'z': z,
            'x': x,
            'y': pop,
            'c': c,
            'full_c': full_c,
            'iso_c': iso_c,
            'exp': exp,
            'norm': n
        }

        dataset.append(data_dict)
        cnt += 1
    print(f"Loaded {cnt} molecules from {picklefile}")
    print(f"Loaded {len(dataset)} molecules from {picklefile}")

    return dataset


def get_scalar_density_comparisons(data, y_ml, Rs, spacing=0.5, buffer=2.0, ldep=False):
    # 生成网格
    x, y, z, vol, x_spacing, y_spacing, z_spacing = generate_grid(data, spacing=spacing, buffer=buffer)

    # l-dependent eps
    ep_per_l = np.zeros(len(Rs))

    if ldep:
        target_density, ml_density, target_density_per_l, ml_density_per_l = gau2grid_density_kdtree(
            x.flatten(), y.flatten(), z.flatten(), data, y_ml, Rs, ldepb=ldep
        )
        for l in range(len(Rs)):
            ep_per_l[l] = 100 * np.sum(np.abs(ml_density_per_l[l] - target_density_per_l[l])) / np.sum(target_density)

    else:
        target_density, ml_density = gau2grid_density_kdtree(
            x.flatten(), y.flatten(), z.flatten(), data, y_ml, Rs, ldepb=ldep
        )

    # 将单位转换为 Bohr
    angstrom2bohr = 1.8897259886

    ep = 100 * np.sum(np.abs(ml_density - target_density)) / np.sum(target_density)

    num_ele_target = np.sum(target_density) * vol * (angstrom2bohr ** 3)
    num_ele_ml = np.sum(ml_density) * vol * (angstrom2bohr ** 3)

    numer = np.sum((ml_density - target_density) ** 2)
    denom = np.sum(ml_density ** 2) + np.sum(target_density ** 2)
    bigI = numer / denom

    if ldep:
        return num_ele_target, num_ele_ml, bigI, ep, ep_per_l
    else:
        return num_ele_target, num_ele_ml, bigI, ep


class Network(nn.Cell):
    def __init__(self, irreps_in, irreps_hidden, irreps_out, irreps_node_attr, irreps_edge_attr, layers, max_radius,
                 number_of_basis, radial_layers, radial_neurons, num_neighbors, num_nodes, reduce_output):
        super(Network, self).__init__()
        # Define your model architecture here using MindSpore's layers.
        self.layers = nn.SequentialCell([
            nn.Dense(24*2, irreps_hidden[0][0]),
            nn.ReLU(),
            nn.Dense(irreps_hidden[0][0], 24*70),
        ])

    def construct(self, data):
        x = data['x'].view(1, -1)
        return self.layers(x).view(24, 70)

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
    for batch_idx, data in enumerate(data_loader.create_dict_iterator()):
        target = data['y']
        loss, logits = train_step(data, target)
        for mul, l in Rs:
            if l == 0:
                num_ele = ops.sum(logits[:, :mul])

        mue_cum += num_ele
        mae_cum += ops.abs(num_ele)
        loss_cum += loss.asnumpy()

        if batch_idx % 1 == 0:
            print(f"{epoch} {float(loss_cum) / len(data_loader):.10f}")
            print(f"Train_Loss l=0 {float(loss_cum) / len(data_loader)}")
            print(f"Train_MAE {mae_cum / (len(data_loader) * b)}")
    return loss_cum, mae_cum, mue_cum


def test_epoch(model, data_loader, loss_fn, Rs, epoch, save_interval, mue_cum):
    model.set_train(False)
    metrics = []
    for testset in data_loader.create_dict_iterator():
        data = testset["data"]
        mask = ops.select(data['y'] == 0, ops.zeros_like(data['y']), ops.ones_like(data['y']))
        y_ml = model(data) * mask
        err = y_ml - data['y']

        for mul, l in Rs:
            if l == 0:
                num_ele = ops.mean(y_ml[:, :mul])

        metrics.append([
            loss_fn(err, data['y']).asnumpy(),
            ops.abs(num_ele).asnumpy(),
            mue_cum.asnumpy(),
            ops.abs(num_ele - ops.sum(data['z'])).asnumpy(),
            None,
            None,
            None
        ])

        if epoch % 1 == 0:
            print(f"Test_Loss {float(metrics[0][0]) / len(data_loader)}")
            print(f"Test_MAE {metrics[0][1] / len(data_loader)}")
            print(f"Test_MUE {metrics[0][2] / len(data_loader)}")
            print(f"Test_Electron_Difference {metrics[0][3] / len(data_loader)}")
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

    train_loader = GeneratorDataset(data_generator(dataset[:split]), ["data"], shuffle=True)
    test_loader = GeneratorDataset(data_generator(test_dataset), ["data"], shuffle=True)

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
        test_epoch(model, test_loader, loss_fn, Rs, epoch, save_interval, mue_cum)
        # model.set_train()
        # loss_cum = 0.0
        # loss_perchannel_cum = np.zeros(len(Rs))
        # mae_cum = 0.0
        # mue_cum = 0.0

        # for step, data in enumerate(train_loader.create_dict_iterator()):
        #     data = data["data"]
        #     mask = ops.select(data['y'] == 0, ops.zeros_like(data['y']), ops.ones_like(data['y']))
        #     y_ml = model(data) * mask
        #     err = y_ml - data['y']

        #     for mul, l in Rs:
        #         if l == 0:
        #             num_ele = ops.sum(y_ml[:, :mul])

        #     mue_cum += num_ele
        #     mae_cum += ops.abs(num_ele)

        #     if ldep_bool:
        #         loss_perchannel_cum += loss_per_channel(y_ml, data['y'], Rs)

        #     loss = loss_fn(err, data['y'])
        #     loss_cum += loss.asnumpy()

        #     loss.backward()
        #     optim.step()
        #     optim.zero_grad()

        # with ms.NoGrad():
        #     metrics = []
        #     for testset in test_loader.create_dict_iterator():
        #         data = testset["data"]
        #         mask = ops.select(data['y'] == 0, ops.zeros_like(data['y']), ops.ones_like(data['y']))
        #         y_ml = model(data) * mask
        #         err = y_ml - data['y']

        #         for mul, l in Rs:
        #             if l == 0:
        #                 num_ele = ops.mean(y_ml[:, :mul])

        #         metrics.append([
        #             loss_fn(err, data['y']).asnumpy(),
        #             ops.abs(num_ele).asnumpy(),
        #             mue_cum.asnumpy(),
        #             ops.abs(num_ele - ops.sum(data['z'])).asnumpy(),
        #             None,
        #             None,
        #             None
        #         ])

        #         if epoch % save_interval == 0:
        #             save_checkpoint(model, os.path.join("./checkpoints/model_weights_epoch_" + str(epoch) + ".ckpt"))

        # if epoch % 1 == 0:
        #     print(f"{epoch} {float(loss_cum) / len(train_loader):.10f}")
        #     print(f"Train_Loss l=0 {float(loss_cum) / len(train_loader)}")
        #     print(f"Train_MAE {mae_cum / (len(train_loader) * b)}")
        #     print(f"Test_Loss {float(metrics[0][0]) / len(test_loader)}")
        #     print(f"Test_MAE {metrics[0][1] / len(test_loader)}")
        #     print(f"Test_MUE {metrics[0][2] / len(test_loader)}")
        #     print(f"Test_Electron_Difference {metrics[0][3] / len(test_loader)}")

if __name__ == '__main__':
    main()
