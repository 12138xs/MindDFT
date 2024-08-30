import math
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
import mindspore.ops as ops
from mindspore import nn, context, Tensor
from mindspore.common import initializer as init
from mindspore.train import Model
from mindspore.train.callback import LossMonitor
from mindchemistry.e3nn.nn.models.gate_points_2101 import Network
from mindchemistry.e3nn import o3
import random

# 设置设备
context.set_context(mode=context.GRAPH_MODE, device_target="GPU" if ms.get_context("device_target") == "GPU" else "CPU")

# 设置默认数据类型
ms.set_context(dtype=ms.float32)

# 加载数据集
hhh = "./data/h_s_only_augccpvdz_density.out"
ooo = "./data/o_s_only_augccpvdz_density.out"
ccc = "./data/c_s_only_augccpvdz_density.out"
nnn = "./data/n_s_only_augccpvdz_density.out"
ppp = "./data/p_s_only_augccpvdz_density.out"

train_datasets = ["1at-400.pkl", "2ta-400.pkl", "3aa-400.pkl", "4ca-400.pkl", "5gt-400.pkl",
                  "6ct-400.pkl", "7ga-400.pkl", "8cg-400.pkl", "9gc-400.pkl", "10gg-400.pkl"]

print(train_datasets)



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

    for molecule in molecules:
        pos = Tensor(molecule['pos'], dtype=ms.float32)
        z = Tensor(molecule['type'].unsqueeze(1), dtype=ms.float32)

        x = Tensor(molecule['onehot'], dtype=ms.float32)

        c = Tensor(molecule['coefficients'], dtype=ms.float32)
        n = Tensor(molecule['norms'], dtype=ms.float32)
        exp = Tensor(molecule['exponents'], dtype=ms.float32)

        full_c = copy.deepcopy(c)
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
        p_pos = copy.deepcopy(pos)
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

    return dataset
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

test_datafile = "2mer-test.pkl"
test_dataset = get_iso_permuted_dataset(test_datafile, h_iso=hhh, c_iso=ccc, n_iso=nnn, o_iso=ooo, p_iso=ppp)
random.shuffle(test_dataset)

b = 1
train_split = [100]

test_loader = ds.GeneratorDataset(test_dataset, ["x", "y"], shuffle=True).batch(b)
num_epochs = 251

# 定义模型参数
Rs = [(14, 0), (5, 1), (5, 2), (2, 3), (1, 4)]

for train_size in train_split:
    model_kwargs = {
        "irreps_in": "5x 0e",  # irreps_in
        "irreps_hidden": [(mul, l, p) for l, mul in enumerate([200, 67, 40, 29]) for p in [-1, 1]],  # irreps_hidden
        "irreps_out": "14x0e + 5x1o + 5x2e + 2x3o + 1x4e",  # irreps_out
        "irreps_node_attr": None,  # irreps_node_attr
        "irreps_edge_attr": o3.Irreps.spherical_harmonics(3),  # irreps_edge_attr
        "layers": 5,
        "max_radius": 3.5,
        "num_neighbors": 12.666666,
        "number_of_basis": 10,
        "radial_layers": 1,
        "radial_neurons": 128,
        "num_nodes": 24,
        "reduce_output": False,
    }

    model = Network(**model_kwargs)
    model.to_float(ms.float32)

    optim = nn.Adam(model.trainable_params(), learning_rate=1e-2)

    loss_fn = nn.MSELoss()

    model.set_train()

    for epoch in range(num_epochs):
        loss_cum = 0.0
        mae_cum = 0.0
        mue_cum = 0.0

        train_num_ele = []
        test_num_ele = []

        for data_file in train_datasets:
            print("Data file: ", data_file)
            train_dataset = get_iso_permuted_dataset(data_file, h_iso=hhh, c_iso=ccc, n_iso=nnn, o_iso=ooo, p_iso=ppp)
            random.shuffle(train_dataset)
            train_loader = ds.GeneratorDataset(train_dataset[:train_size], ["x", "y"], shuffle=True).batch(b)

            for step, data in enumerate(train_loader.create_tuple_iterator()):
                x, y = data
                mask = ops.Cast()(ops.Equal()(y, 0), ms.float32)
                y_ml = model(x) * mask
                err = y_ml - y

                for mul, l in Rs:
                    if l == 0:
                        num_ele = ops.ReduceSum()(ops.ReduceSum()(y_ml[:, :mul]))

                train_num_ele.append(num_ele.asnumpy())

                mue_cum += num_ele
                mae_cum += ops.Abs()(num_ele)

                loss_cum += ops.ReduceMean()(err ** 2)
                loss = loss_fn(y_ml, y)
                loss.backward()
                optim.step()
                optim.clear_grad()

        print("Train num ele: ", len(train_num_ele))
        train_tot = len(train_num_ele)
        train_stdev = np.std(train_num_ele)

        # 测试循环
        model.set_train(False)
        with ops.stop_gradient():
            metrics = []
            for testset in [test_loader]:
                test_loss_cum = 0.0
                test_mae_cum = 0.0
                test_mue_cum = 0.0
                bigIs_cum = 0.0
                eps_cum = 0.0
                ele_diff_cum = 0.0

                for step, data in enumerate(testset.create_tuple_iterator()):
                    x, y = data
                    mask = ops.Cast()(ops.Equal()(y, 0), ms.float32)
                    y_ml = model(x) * mask
                    err = y_ml - y

                    for mul, l in Rs:
                        if l == 0:
                            num_ele = ops.ReduceSum()(ops.ReduceSum()(y_ml[:, :mul]))

                    test_num_ele.append(num_ele.asnumpy())

                    test_mue_cum += num_ele
                    test_mae_cum += ops.Abs()(num_ele)
                    test_loss_cum += ops.ReduceMean()(err ** 2)

                    if (epoch != 0 and epoch % 10 == 0):
                        num_ele_target, num_ele_ml, bigI, ep = get_scalar_density_comparisons(data, y_ml, Rs, spacing=0.2, buffer=4.0)
                        n_ele = np.sum(data.z.asnumpy())
                        ele_diff_cum += np.abs(n_ele - num_ele_target)
                        bigIs_cum += bigI
                        eps_cum += ep

                metrics.append([test_loss_cum, test_mae_cum, test_mue_cum])

            test_stdev = np.std(test_num_ele)

        # 日志记录和模型保存 (这里省略了 wandb 日志部分)
        # 可以使用 MindSpore 的日志工具或自定义日志

        # 保存模型
        ms.save_checkpoint(model, "trainerror-100-" + str(epoch) + ".ckpt")
