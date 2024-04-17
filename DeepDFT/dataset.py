from typing import List, Optional
import gzip
import tarfile
import tempfile
import multiprocessing
import queue
import time
import threading
import logging
import zlib
import os
import io
import math
import lz4.frame
import numpy as np
import ase
import ase.neighborlist
import ase.io.cube
import ase.units
from ase.calculators.vasp import VaspChargeDensity
import asap3
import mindspore as ms


from layer import pad_and_stack

# 这个函数也是用来调试的,用来查看数据结构
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

def _cell_heights(cell_object):
    volume = cell_object.volume
    crossproducts = np.cross(cell_object[[1, 2, 0]], cell_object[[2, 0, 1]])
    crosslengths = np.sqrt(np.sum(np.square(crossproducts), axis=1))
    heights = volume / crosslengths
    return heights


def rotating_pool_worker(dataset, rng, queue):
    while True:
        for index in rng.permutation(len(dataset)).tolist():
            queue.put(dataset[index])


def transfer_thread(queue: multiprocessing.Queue, datalist: list):
    while True:
        for index in range(len(datalist)):
            datalist[index] = queue.get()


class RotatingPoolData:
    """Wrapper for a dataset that continously loads data into a smaller pool.
    The data loading is performed in a separate process and is assumed to be IO bound."""
    def __init__(self, dataset, pool_size):
        super(RotatingPoolData).__init__()
        self.pool_size = pool_size
        self.parent_data = dataset
        self.rng = np.random.default_rng()
        logging.debug("Filling rotating data pool of size %d" % pool_size)
        self.data_pool = [
            self.parent_data[i]
            for i in self.rng.integers(
                0, high=len(self.parent_data), size=self.pool_size, endpoint=False
            ).tolist()
        ]
        self.loader_queue = multiprocessing.Queue(2)

        # Start loaders
        self.loader_process = multiprocessing.Process(
            target=rotating_pool_worker,
            args=(self.parent_data, self.rng, self.loader_queue),
        )
        self.transfer_thread = threading.Thread(
            target=transfer_thread, args=(self.loader_queue, self.data_pool)
        )
        self.loader_process.start()
        self.transfer_thread.start()

    def __len__(self):
        return self.pool_size

    def __getitem__(self, index):
        return self.data_pool[index]


class BufferData:
    """Wrapper for a dataset. Loads all data into memory."""
    def __init__(self, dataset):
        self.data_objects = [dataset[i] for i in range(len(dataset))]

    def __len__(self):
        return len(self.data_objects)

    def __getitem__(self, index):
        return self.data_objects[index]


# AttributeError: 'DensityData' object has no attribute 'parent'
class DensityData:
    def __init__(self, datapath):
        if os.path.isfile(datapath) and datapath.endswith(".tar"):
            self.data = DensityDataTar(datapath)
        elif os.path.isdir(datapath):
            self.data = DensityDataDir(datapath)
        else:
            raise ValueError("Did not find dataset at path %s", datapath)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    ### 这是我自定义的
    def take(self, indices):
        if not isinstance(indices, (list, tuple)):  
            raise ValueError("indices must be an iterable (e.g., list or tuple)")    
        return [self.data[idx] for idx in indices]

    def concat(self, other: 'DensityData') -> 'DensityData':  
        """  
        Concatenate this DensityData object with another DensityData object.  
  
        :param other: Another DensityData object to concatenate with.  
        :return: A new DensityData object that contains the concatenated data.  
        """  
        # 首先检查两个对象是否都是DensityData的实例  
        if not isinstance(other, DensityData):  
            raise ValueError("The 'other' object must be an instance of DensityData")  
           
        combined_data = []  
          
        # 合并当前对象的数据  
        for i in range(len(self)):  
            combined_data.append(self[i])  
          
        # 合并另一个对象的数据  
        for i in range(len(other)):  
            combined_data.append(other[i])  
            
        concatenated_data = DensityDataList(combined_data)  
          
        return concatenated_data


class DensityDataDir:
    def __init__(self, directory):
        self.directory = directory
        self.member_list = sorted(os.listdir(self.directory))
        self.key_to_idx = {str(k): i for i,k in enumerate(self.member_list)}

    def __len__(self):
        return len(self.member_list)

    def extractfile(self, filename):
        path = os.path.join(self.directory, filename)

        filecontent = _decompress_file(path)
        if path.endswith((".cube", ".cube.gz", ".cube.zz", "cube.lz4")):
            density, atoms, origin = _read_cube(filecontent)
        else:
            density, atoms, origin = _read_vasp(filecontent)

        grid_pos = _calculate_grid_pos(density, origin, atoms.get_cell())

        metadata = {"filename": filename}
        return {
            "density": density,
            "atoms": atoms,
            "origin": origin,
            "grid_position": grid_pos,
            "metadata": metadata, # Meta information
        }

    def __getitem__(self, index):
        if isinstance(index, str):
            index = self.key_to_idx[index]
        return self.extractfile(self.member_list[index])


class DensityDataTar:
    def __init__(self, tarpath):
        self.tarpath = tarpath
        self.member_list = []

        # Index tar file
        with tarfile.open(self.tarpath, "r:") as tar:
            for member in tar.getmembers():
                self.member_list.append(member)
        self.key_to_idx = {str(k): i for i,k in enumerate(self.member_list)}

    def __len__(self):
        return len(self.member_list)

    def extract_member(self, tarinfo):
        with tarfile.open(self.tarpath, "r") as tar:
            filecontent = _decompress_tarmember(tar, tarinfo)
            if tarinfo.name.endswith((".cube", ".cube.gz", "cube.zz", "cube.lz4")):
                density, atoms, origin = _read_cube(filecontent)
            else:
                density, atoms, origin = _read_vasp(filecontent)

        grid_pos = _calculate_grid_pos(density, origin, atoms.get_cell())

        metadata = {"filename": tarinfo.name}
        return {
            "density": density,
            "atoms": atoms,
            "origin": origin,
            "grid_position": grid_pos,
            "metadata": metadata, # Meta information
        }

    def __getitem__(self, index):
        if isinstance(index, str):
            index = self.key_to_idx[index]
        return self.extract_member(self.member_list[index])


class AseNeigborListWrapper:
    """Wrapper around ASE neighborlist to have the same interface as asap3 neighborlist"""
    def __init__(self, cutoff, atoms):
        self.neighborlist = ase.neighborlist.NewPrimitiveNeighborList(
            cutoff, skin=0.0, self_interaction=False, bothways=True
        )
        self.neighborlist.build(
            atoms.get_pbc(), atoms.get_cell(), atoms.get_positions()
        )
        self.cutoff = cutoff
        self.atoms_positions = atoms.get_positions()
        self.atoms_cell = atoms.get_cell()

    def get_neighbors(self, i, cutoff):
        assert (
            cutoff == self.cutoff
        ), "Cutoff must be the same as used to initialise the neighborlist"

        indices, offsets = self.neighborlist.get_neighbors(i)

        rel_positions = (
            self.atoms_positions[indices]
            + offsets @ self.atoms_cell
            - self.atoms_positions[i][None]
        )

        dist2 = np.sum(np.square(rel_positions), axis=1)

        return indices, rel_positions, dist2


def grid_iterator_worker(atoms, meshgrid, probe_count, cutoff, slice_id_queue, result_queue):
    try:
        neighborlist = asap3.FullNeighborList(cutoff, atoms)
    except Exception as e:
        logging.info("Failed to create asap3 neighborlist, this might be very slow. Error: %s", e)
        neighborlist = None
    while True:
        try:
            slice_id = slice_id_queue.get(True, 1)
        except queue.Empty:
            while not result_queue.empty():
                time.sleep(1)
            result_queue.close()
            return 0
        res = DensityGridIterator.static_get_slice(slice_id, atoms, meshgrid, probe_count, cutoff, neighborlist=neighborlist)
        result_queue.put((slice_id, res))


class DensityGridIterator:
    def __init__(self, densitydict, probe_count: int, cutoff: float, set_pbc_to: Optional[bool] = None):
        num_positions = np.prod(densitydict["grid_position"].shape[0:3])
        self.num_slices = int(math.ceil(num_positions / probe_count))
        self.probe_count = probe_count
        self.cutoff = cutoff
        self.set_pbc = set_pbc_to

        if self.set_pbc is not None:
            self.atoms = densitydict["atoms"].copy()
            self.atoms.set_pbc(self.set_pbc)
        else:
            self.atoms = densitydict["atoms"]

        self.meshgrid = densitydict["grid_position"]

    def get_slice(self, slice_index):
        return self.static_get_slice(slice_index, self.atoms, self.meshgrid, self.probe_count, self.cutoff)

    @staticmethod
    def static_get_slice(slice_index, atoms, meshgrid, probe_count, cutoff, neighborlist=None):
        num_positions = np.prod(meshgrid.shape[0:3])
        flat_index = np.arange(slice_index*probe_count, min((slice_index+1)*probe_count, num_positions))
        pos_index = np.unravel_index(flat_index, meshgrid.shape[0:3])
        probe_pos = meshgrid[pos_index]
        probe_edges, probe_edges_displacement = probes_to_graph(atoms, probe_pos, cutoff, neighborlist)

        if not probe_edges:
            probe_edges = [np.zeros((0,2), dtype=np.int)]
            probe_edges_displacement = [np.zeros((0,3), dtype=np.float32)]

        res = {
            "probe_edges": np.concatenate(probe_edges, axis=0),
            "probe_edges_displacement": np.concatenate(probe_edges_displacement, axis=0).astype(np.float32),
        }
        res["num_probe_edges"] = res["probe_edges"].shape[0]
        res["num_probes"] = len(flat_index)
        res["probe_xyz"] = probe_pos.astype(np.float32)
        return res


    def __iter__(self):
        self.current_slice = 0
        slice_id_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue(100)
        self.finished_slices = dict()
        for i in range(self.num_slices):
            slice_id_queue.put(i)
        self.workers = [multiprocessing.Process(
            target=grid_iterator_worker,
            args=(self.atoms, self.meshgrid, self.probe_count, self.cutoff, slice_id_queue, self.result_queue)) for _ in range(6)
            ]
        for w in self.workers:
            w.start()
        return self

    def __next__(self):
        if self.current_slice < self.num_slices:
            this_slice = self.current_slice
            self.current_slice += 1

            # Retrieve finished slices until we get the one we are looking for
            while this_slice not in self.finished_slices:
                i, res = self.result_queue.get()
                res = {k: ms.Tensor(v) for k,v in res.items()} # convert to mindspore tensor
                self.finished_slices[i] = res
            return self.finished_slices.pop(this_slice)
        else:
            for w in self.workers:
                w.join()
            raise StopIteration


def atoms_and_probe_sample_to_graph_dict(density, atoms, grid_pos, cutoff, num_probes):
    # Sample probes on the calculated grid
    probe_choice_max = np.prod(grid_pos.shape[0:3])
    probe_choice = np.random.randint(probe_choice_max, size=num_probes)
    probe_choice = np.unravel_index(probe_choice, grid_pos.shape[0:3])
    probe_pos = grid_pos[probe_choice]
    probe_target = density[probe_choice]

    atom_edges, atom_edges_displacement, neighborlist, inv_cell_T = atoms_to_graph(atoms, cutoff)
    probe_edges, probe_edges_displacement = probes_to_graph(atoms, probe_pos, cutoff, neighborlist=neighborlist, inv_cell_T=inv_cell_T)

    default_type = ms.float32   # MindSpore默认数据类型

    if not probe_edges:
        probe_edges = [np.zeros((0,2), dtype=np.int)]
        probe_edges_displacement = [np.zeros((0,3), dtype=np.int)]
    # pylint: disable=E1102
    res = {
        "nodes": ms.Tensor(atoms.get_atomic_numbers()),
        "atom_edges": ms.Tensor(np.concatenate(atom_edges, axis=0)),
        "atom_edges_displacement": ms.Tensor(
            np.concatenate(atom_edges_displacement, axis=0), dtype=default_type
        ),
        "probe_edges": ms.Tensor(np.concatenate(probe_edges, axis=0)),
        "probe_edges_displacement": ms.Tensor(
            np.concatenate(probe_edges_displacement, axis=0), dtype=default_type
        ),
        "probe_target": ms.Tensor(probe_target, dtype=default_type),
    }
    res["num_nodes"] = ms.Tensor(res["nodes"].shape[0])
    res["num_atom_edges"] = ms.Tensor(res["atom_edges"].shape[0])
    res["num_probe_edges"] = ms.Tensor(res["probe_edges"].shape[0])
    res["num_probes"] = ms.Tensor(res["probe_target"].shape[0])
    res["probe_xyz"] = ms.Tensor(probe_pos, dtype=default_type)
    res["atom_xyz"] = ms.Tensor(atoms.get_positions(), dtype=default_type)
    res["cell"] = ms.Tensor(np.array(atoms.get_cell()), dtype=default_type)

    return res


def atoms_to_graph_dict(atoms, cutoff):
    atom_edges, atom_edges_displacement, _, _ = atoms_to_graph(atoms, cutoff)

    default_type = ms.float32

    # pylint: disable=E1102
    res = {
        "nodes": ms.Tensor(atoms.get_atomic_numbers()),
        "atom_edges": ms.Tensor(np.concatenate(atom_edges, axis=0)),
        "atom_edges_displacement": ms.Tensor(
            np.concatenate(atom_edges_displacement, axis=0), dtype=default_type
        ),
    }
    res["num_nodes"] = ms.Tensor(res["nodes"].shape[0])
    res["num_atom_edges"] = ms.Tensor(res["atom_edges"].shape[0])
    res["atom_xyz"] = ms.Tensor(atoms.get_positions(), dtype=default_type)
    res["cell"] = ms.Tensor(np.array(atoms.get_cell()), dtype=default_type)

    return res


def atoms_to_graph(atoms, cutoff):
    atom_edges = []
    atom_edges_displacement = []

    inv_cell_T = np.linalg.inv(atoms.get_cell().complete().T)

    # Compute neighborlist
    if (
        np.any(atoms.get_cell().lengths() <= 0.0001)
        or (
            np.any(atoms.get_pbc())
            and np.any(_cell_heights(atoms.get_cell()) < cutoff)
        )
    ):
        neighborlist = AseNeigborListWrapper(cutoff, atoms)
    else:
        neighborlist = asap3.FullNeighborList(cutoff, atoms)

    atom_positions = atoms.get_positions()

    for i in range(len(atoms)):
        neigh_idx, neigh_vec, _ = neighborlist.get_neighbors(i, cutoff)

        self_index = np.ones_like(neigh_idx) * i
        edges = np.stack((neigh_idx, self_index), axis=1)

        neigh_pos = atom_positions[neigh_idx]
        this_pos = atom_positions[i]
        neigh_origin = neigh_vec + this_pos - neigh_pos
        neigh_origin_scaled = np.round(inv_cell_T.dot(neigh_origin.T).T)

        atom_edges.append(edges)
        atom_edges_displacement.append(neigh_origin_scaled)

    return atom_edges, atom_edges_displacement, neighborlist, inv_cell_T


def probes_to_graph(atoms, probe_pos, cutoff, neighborlist=None, inv_cell_T=None):
    probe_edges = []
    probe_edges_displacement = []
    if inv_cell_T is None:
        inv_cell_T = np.linalg.inv(atoms.get_cell().complete().T)

    if hasattr(neighborlist, "get_neighbors_querypoint"):
        results = neighborlist.get_neighbors_querypoint(probe_pos, cutoff)
        atomic_numbers = atoms.get_atomic_numbers()
    else:
        # Insert probe atoms
        num_probes = probe_pos.shape[0]
        probe_atoms = ase.Atoms(numbers=[0] * num_probes, positions=probe_pos)
        atoms_with_probes = atoms.copy()
        atoms_with_probes.extend(probe_atoms)
        atomic_numbers = atoms_with_probes.get_atomic_numbers()

        if (
            np.any(atoms.get_cell().lengths() <= 0.0001)
            or (
                np.any(atoms.get_pbc())
                and np.any(_cell_heights(atoms.get_cell()) < cutoff)
            )
        ):
            neighborlist = AseNeigborListWrapper(cutoff, atoms_with_probes)
        else:
            neighborlist = asap3.FullNeighborList(cutoff, atoms_with_probes)

        results = [neighborlist.get_neighbors(i+len(atoms), cutoff) for i in range(num_probes)]

    atom_positions = atoms.get_positions()
    for i, (neigh_idx, neigh_vec, _) in enumerate(results):
        neigh_atomic_species = atomic_numbers[neigh_idx]

        neigh_is_atom = neigh_atomic_species != 0
        neigh_atoms = neigh_idx[neigh_is_atom]
        self_index = np.ones_like(neigh_atoms) * i
        edges = np.stack((neigh_atoms, self_index), axis=1)

        neigh_pos = atom_positions[neigh_atoms]
        this_pos = probe_pos[i]
        neigh_origin = neigh_vec[neigh_is_atom] + this_pos - neigh_pos
        neigh_origin_scaled = np.round(inv_cell_T.dot(neigh_origin.T).T)

        probe_edges.append(edges)
        probe_edges_displacement.append(neigh_origin_scaled)

    return probe_edges, probe_edges_displacement


def collate_list_of_dicts(list_of_dicts):
    # Convert from "list of dicts" to "dict of lists"
    dict_of_lists = {k: [dic[k] for dic in list_of_dicts] for k in list_of_dicts[0]}

    # Convert each list of tensors to single tensor with pad and stack
    collated = {k: pad_and_stack(dict_of_lists[k]) for k in dict_of_lists}
    return collated


class CollateFuncRandomSample:
    def __init__(self, cutoff, num_probes, set_pbc_to=None):
        self.num_probes = num_probes
        self.cutoff = cutoff
        self.set_pbc = set_pbc_to

    def __call__(self, input_dicts: List):
        # runner中有一处暂时取了filelist[0], 可能会导致本来应该是列表的input_dicts变成字典，所以这里把它转换成list 
        if not isinstance(input_dicts, list):    
            input_dicts = [input_dicts]
            # print("input_dicts has been transformed into list!")
        # print_structure(input_dicts)
        graphs = []
        for i in input_dicts:
            # print("--------------debug remark 2--------------")
            # print_structure(i)
            if self.set_pbc is not None:
                atoms = i["atoms"].copy()
                atoms.set_pbc(self.set_pbc)
            else:
                atoms = i["atoms"]

            graphs.append(atoms_and_probe_sample_to_graph_dict(
                i["density"],
                atoms,
                i["grid_position"],
                self.cutoff,
                self.num_probes,
            ))

        return collate_list_of_dicts(graphs)


class CollateFuncAtoms:
    def __init__(self, cutoff, set_pbc_to=None):
        self.cutoff = cutoff
        self.set_pbc = set_pbc_to

    def __call__(self, input_dicts: List):
        graphs = []
        for i in input_dicts:
            if self.set_pbc is not None:
                atoms = i["atoms"].copy()
                atoms.set_pbc(self.set_pbc)
            else:
                atoms = i["atoms"]

            graphs.append(atoms_to_graph_dict(
                atoms,
                self.cutoff,
            ))

        return collate_list_of_dicts(graphs)


def _calculate_grid_pos(density, origin, cell):
    # Calculate grid positions
    ngridpts = np.array(density.shape)  # grid matrix
    grid_pos = np.meshgrid(
        np.arange(ngridpts[0]) / density.shape[0],
        np.arange(ngridpts[1]) / density.shape[1],
        np.arange(ngridpts[2]) / density.shape[2],
        indexing="ij",
    )
    grid_pos = np.stack(grid_pos, 3)
    grid_pos = np.dot(grid_pos, cell)
    grid_pos = grid_pos + origin
    return grid_pos


def _decompress_tarmember(tar, tarinfo):
    """Extract compressed tar file member and return a bytes object with the content"""

    bytesobj = tar.extractfile(tarinfo).read()
    if tarinfo.name.endswith(".zz"):
        filecontent = zlib.decompress(bytesobj)
    elif tarinfo.name.endswith(".lz4"):
        filecontent = lz4.frame.decompress(bytesobj)
    elif tarinfo.name.endswith(".gz"):
        filecontent = gzip.decompress(bytesobj)
    else:
        filecontent = bytesobj

    return filecontent


def _decompress_file(filepath):
    if filepath.endswith(".zz"):
        with open(filepath, "rb") as fp:
            f_bytes = fp.read()
        filecontent = zlib.decompress(f_bytes)
    elif filepath.endswith(".lz4"):
        with lz4.frame.open(filepath, mode="rb") as fp:
            filecontent = fp.read()
    elif filepath.endswith(".gz"):
        with gzip.open(filepath, mode="rb") as fp:
            filecontent = fp.read()
    else:
        with open(filepath, mode="rb") as fp:
            filecontent = fp.read()
    return filecontent


def _read_vasp(filecontent):
    # Write to tmp file and read using ASE
    tmpfd, tmppath = tempfile.mkstemp(prefix="tmpdeepdft")
    tmpfile = os.fdopen(tmpfd, "wb")
    tmpfile.write(filecontent)
    tmpfile.close()
    vasp_charge = VaspChargeDensity(filename=tmppath)
    os.remove(tmppath)
    density = vasp_charge.chg[-1]  # separate density
    atoms = vasp_charge.atoms[-1]  # separate atom positions

    return density, atoms, np.zeros(3) 


def _read_cube(filecontent):
    textbuf = io.StringIO(filecontent.decode())
    cube = ase.io.cube.read_cube(textbuf)
    # sometimes there is an entry at index 3
    # denoting the number of values for each grid position
    origin = cube["origin"][0:3]
    # by convention the cube electron density is given in electrons/Bohr^3,
    # and ase read_cube does not convert to electrons/Å^3, so we do the conversion here
    cube["data"] *= 1.0 / ase.units.Bohr ** 3
    return cube["data"], cube["atoms"], origin
