# +
from mindspore.ops import deepcopy
from mindspore import Tensor
import numpy as np
import mindspore as ms
import mindspore.ops as ops

def collate_list_of_dicts(list_of_dicts, batch_info=None):
    dict_of_lists = {k: [dic[k] for dic in list_of_dicts] for k in list_of_dicts[0]}
    # collated = {k: Tensor(np.vstack(dict_of_lists[k]), dtype=ms.float32) for k in dict_of_lists}
    collated = {k : ops.cat(dict_of_lists[k], axis=0) for k in dict_of_lists}
    return collated

def flatten_list(nested_list):
    """Flatten an arbitrarily nested list, without recursion (to avoid
    stack overflows). Returns a new list, the original list is unchanged.
    >> list(flatten_list([1, 2, 3, [4], [], [[[[[[[[[5]]]]]]]]]]))
    [1, 2, 3, 4, 5]
    >> list(flatten_list([[1, 2], 3]))
    [1, 2, 3]
    """
    nested_list = deepcopy(nested_list)
    while nested_list:
        sublist = nested_list.pop(0)
        if isinstance(sublist, list):
            nested_list = sublist + nested_list
        else:
            yield sublist

def get_iso_permuted_dataset(picklefile, amberFlag=0, **atm_iso):
    import math
    import pickle
    import numpy as np
    import mindspore as ms
    import mindspore.numpy as mnp
    from mindspore import Tensor

    dataset = []

    for key, value in atm_iso.items():
        if key == 'h_iso':
            h_data = np.loadtxt(value, skiprows=2, usecols=1)
        elif key == 'c_iso':
            c_data = np.loadtxt(value, skiprows=2, usecols=1)
        elif key == 'n_iso':
            n_data = np.loadtxt(value, skiprows=2, usecols=1)
        elif key == 'o_iso':
            o_data = np.loadtxt(value, skiprows=2, usecols=1)
        elif key == 'p_iso':
            p_data = np.loadtxt(value, skiprows=2, usecols=1)
        else:
            raise ValueError("Isolated atom type not found. Use kwargs \"h_iso\", \"c_iso\", etc.")

    with open(picklefile, "rb") as f:
        molecules = pickle.load(f)

    for idx, molecule in enumerate(molecules):
        if idx == 50:
            break
        pos    = molecule['pos']
        z      = np.expand_dims(molecule['type'], axis=1)
        x      = molecule['onehot']
        c      = molecule['coefficients']
        n      = molecule['norms']
        exp    = molecule['exponents']
        full_c = c.copy()
        iso_c  = np.zeros_like(c)

        if amberFlag==1:
            amber_chg = molecule['amber_chg']

        # Subtract the isolated atoms
        for atom, iso, typ in zip(c, iso_c, z):
            typ_value = typ.item()
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

        pop = np.where(n != 0, c * 2 * math.sqrt(2) / n, n)

        # Permute positions, yzx -> xyz
        p_pos = pos.copy()
        p_pos[:, 0] = pos[:, 1]
        p_pos[:, 1] = pos[:, 2]
        p_pos[:, 2] = pos[:, 0]

        # Create dataset dictionary
        # if amberFlag==1:
        #     data_dict = {
        #         'pos': p_pos,
        #         'pos_orig': pos,
        #         'z': z,
        #         'x': x,
        #         'y': pop,
        #         'c': c,
        #         'full_c': full_c,
        #         'iso_c': iso_c,
        #         'exp': exp,
        #         'norm': n,
        #         'amber_chg': amber_chg
        #     }
        # else:
        #     data_dict = {
        #         'pos': p_pos,
        #         'pos_orig': pos,
        #         'z': z,
        #         'x': x,
        #         'y': pop,
        #         'c': c,
        #         'full_c': full_c,
        #         'iso_c': iso_c,
        #         'exp': exp,
        #         'norm': n
        #     }

        # Create dataset dictionary with Tensor
        if amberFlag==1:
            data_dict = {
                'pos': Tensor(p_pos, dtype=ms.float32),
                'pos_orig': Tensor(pos, dtype=ms.float32),
                'z': Tensor(z, dtype=ms.float32),
                'x': Tensor(x, dtype=ms.float32),
                'y': Tensor(pop, dtype=ms.float32),
                'c': Tensor(c, dtype=ms.float32),
                'full_c': Tensor(full_c, dtype=ms.float32),
                'iso_c': Tensor(iso_c, dtype=ms.float32),
                'exp': Tensor(exp, dtype=ms.float32),
                'norm': Tensor(n, dtype=ms.float32),
                'amber_chg': Tensor(amber_chg, dtype=ms.float32)
            }
        else:
            data_dict = {
                'pos': Tensor(p_pos, dtype=ms.float32),
                'pos_orig': Tensor(pos, dtype=ms.float32),
                'z': Tensor(z, dtype=ms.float32),
                'x': Tensor(x, dtype=ms.float32),
                'y': Tensor(pop, dtype=ms.float32),
                'c': Tensor(c, dtype=ms.float32),
                'full_c': Tensor(full_c, dtype=ms.float32),
                'iso_c': Tensor(iso_c, dtype=ms.float32),
                'exp': Tensor(exp, dtype=ms.float32),
                'norm': Tensor(n, dtype=ms.float32)
            }

        dataset.append(data_dict)

    print(f"Loaded {len(dataset)} molecules from {picklefile}")

    return dataset


def get_iso_dataset(picklefile, **atm_iso):
    import math
    import pickle
    import numpy as np
    import mindspore as ms
    import mindspore.numpy as mnp
    from mindspore import Tensor

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
        pos = Tensor(molecule['pos'], dtype=ms.float32)
        # z is atomic number- may want to make 1,0
        z = Tensor(np.expand_dims(molecule['type'], axis=1), dtype=ms.float32)
        x = Tensor(molecule['onehot'], dtype=ms.float32)
        c = Tensor(molecule['coefficients'], dtype=ms.float32)
        n = Tensor(molecule['norms'], dtype=ms.float32)
        exp = Tensor(molecule['exponents'], dtype=ms.float32)
        energy = Tensor(molecule['energy'], dtype=ms.float32)
        # this is a gradient, not forces
        # convert from hartree/bohr to kcal/mol/ang
        bohr2ang = 0.529177
        hartree2kcal = 627.5094740631
        forces = molecule['forces']*hartree2kcal/bohr2ang

        full_c = deepcopy(c)

        # Subtract the isolated atoms
        for atom, typ in zip(c, z):
            typ_value = typ.asnumpy().item()
            if typ_value == 1.0:
                atom[:h_data.shape[0]] -= h_data
            elif typ_value == 6.0:
                atom[:c_data.shape[0]] -= c_data
            elif typ_value == 7.0:
                atom[:n_data.shape[0]] -= n_data
            elif typ_value == 8.0:
                atom[:o_data.shape[0]] -= o_data
            elif typ_value == 15.0:
                atom[:p_data.shape[0]] -= p_data
            else:
                raise ValueError("Isolated atom type not supported!")

        pop = mnp.where(n != 0, c * 2 * math.sqrt(2) / n, n)

        # Create dataset dictionary
        data_dict = {
            'pos': pos,
            'pos_orig': pos,
            'z': z,
            'x': x,
            'y': pop,
            'c': c,
            'full_c': full_c,
            'exp': exp,
            'norm': n,
            'energy': energy,
            'forces': forces
        }

        dataset.append(data_dict)
        cnt += 1
        
    print(f"Loaded {cnt} molecules from {picklefile}")
    print(f"Loaded {len(dataset)} molecules from {picklefile}")

    return dataset


# NOTE: The units of x, y, z here are assumed to be angstrom
#       I convert to bohr for gau2grid, but the grid remains in angstroms
def gau2grid_density_kdtree(x, y, z, data, ml_y, rs, isoOnlyFlag=0, small=1e-5):
    import numpy as np
    import gau2grid as g2g
    from scipy import spatial

    xyz = np.vstack([x, y, z])
    tree = spatial.cKDTree(xyz.T)

    ml_density = np.zeros_like(x)
    target_density = np.zeros_like(x)

    for coords, full_coeffs, iso_coeffs, ml_coeffs, alpha, norm in zip(
        data['pos_orig'].asnumpy(), data['full_c'].asnumpy(), data['iso_c'].asnumpy(), ml_y.asnumpy(),
        data['exp'].asnumpy(), data['norm'].asnumpy()
    ):
        center = coords
        counter = 0
        for mul, l in rs:
            for j in range(mul):
                normal = norm[counter]
                if normal != 0:
                    exp = [alpha[counter]]

                    angstrom2bohr = 1.8897259886
                    bohr2angstrom = 1 / angstrom2bohr

                    target_full_coeffs = full_coeffs[counter:counter + (2 * l + 1)]

                    pop_ml = ml_coeffs[counter:counter + (2 * l + 1)]
                    c_ml = pop_ml * normal / (2 * np.sqrt(2))
                    if isoOnlyFlag==1:
                        c_ml = 0.0 # ML coeff set to 0.0; only isolated atom used
                    ml_full_coeffs = c_ml + iso_coeffs[counter:counter + (2 * l + 1)]

                    target_max = np.amax(np.abs(target_full_coeffs))
                    ml_max = np.amax(np.abs(ml_full_coeffs))
                    max_c = np.amax(np.array([target_max, ml_max]))

                    cutoff = np.sqrt((-1 / exp[0]) * np.log(small / np.abs(max_c * normal))) * bohr2angstrom

                    close_indices = tree.query_ball_point(center, cutoff)
                    points = np.require(xyz[:, close_indices], requirements=['C', 'A'])

                    ret_target = g2g.collocation(points * angstrom2bohr, l, [1], exp, center * angstrom2bohr)
                    ret_ml = g2g.collocation(points * angstrom2bohr, l, [1], exp, center * angstrom2bohr)

                    # Now permute back to psi4 ordering
                    ##              s     p         d             f                 g
                    psi4_2_e3nn = [[0],[2,0,1],[4,2,0,1,3],[6,4,2,0,1,3,5],[8,6,4,2,0,1,3,5,7]]
                    e3nn_2_psi4 = [[0],[1,2,0],[2,3,1,4,0],[3,4,2,5,1,6,0],[4,5,3,6,2,7,1,8,0]]
                    target_full_coeffs = np.array([target_full_coeffs[k] for k in e3nn_2_psi4[l]])
                    ml_full_coeffs = np.array([ml_full_coeffs[k] for k in e3nn_2_psi4[l]])

                    scaled_components = (target_full_coeffs * normal * ret_target["PHI"].T).T
                    target_tot = np.sum(scaled_components, axis=0)

                    ml_scaled_components = (ml_full_coeffs * normal * ret_target["PHI"].T).T
                    ml_tot = np.sum(ml_scaled_components, axis=0)

                    target_density[close_indices] += target_tot
                    ml_density[close_indices] += ml_tot

                counter += 2 * l + 1

    return target_density, ml_density


# find min and max of coordinates
# def find_min_max(coords):
#     xmin, xmax = coords[0, 0], coords[0, 0]
#     ymin, ymax = coords[0, 1], coords[0, 1]
#     zmin, zmax = coords[0, 2], coords[0, 2]
#     for coord in coords:
#         if coord[0] < xmin:
#             xmin = coord[0]
#         if coord[0] > xmax:
#             xmax = coord[0]
#         if coord[1] < ymin:
#             ymin = coord[1]
#         if coord[1] > ymax:
#             ymax = coord[1]
#         if coord[2] < zmin:
#             zmin = coord[2]
#         if coord[2] > zmax:
#             zmax = coord[2]
#     return xmin, xmax, ymin, ymax, zmin, zmax

def find_min_max(coords):
    xmin = coords[:, 0].min()
    xmax = coords[:, 0].max()
    ymin = coords[:, 1].min()
    ymax = coords[:, 1].max()
    zmin = coords[:, 2].min()
    zmax = coords[:, 2].max()
    return xmin, xmax, ymin, ymax, zmin, zmax


def generate_grid(data, spacing=0.5, buffer=2.0):
    import numpy as np

    buf = buffer
    xmin, xmax, ymin, ymax, zmin, zmax = find_min_max(data['pos_orig'].asnumpy())

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


def get_scalar_density_comparisons(data, y_ml, Rs, spacing=0.5, buffer=2.0):
    import numpy as np
    # 生成网格
    x, y, z, vol, x_spacing, y_spacing, z_spacing = generate_grid(data, spacing=spacing, buffer=buffer)
    target_density, ml_density = gau2grid_density_kdtree(x.flatten(),y.flatten(),z.flatten(),data,y_ml,Rs)

    # 将单位转换为 Bohr
    angstrom2bohr = 1.8897259886
    bohr2angstrom = 1/angstrom2bohr

    ep = 100 * np.sum(np.abs(ml_density - target_density)) / np.sum(target_density)

    num_ele_target = np.sum(target_density) * vol * (angstrom2bohr ** 3)
    num_ele_ml = np.sum(ml_density) * vol * (angstrom2bohr ** 3)

    numer = np.sum((ml_density - target_density) ** 2)
    denom = np.sum(ml_density ** 2) + np.sum(target_density ** 2)
    bigI = numer / denom

    return num_ele_target, num_ele_ml, bigI, ep