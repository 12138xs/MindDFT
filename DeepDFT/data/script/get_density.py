import os
import torch
import dataset
from dataset import _decompress_file,_read_vasp,_calculate_grid_pos,probes_to_graph
import numpy as np
import math
from ase.neighborlist import NeighborList,NewPrimitiveNeighborList

#with open("../qm9vasp.txt", "r") as datasetfiles:
#    filelist = [os.path.join(os.path.dirname("../qm9vasp.txt"), line.strip('\n')) for line in datasetfiles]

#densitydata = torch.utils.data.ConcatDataset([dataset.DensityData(path) for path in filelist])
#datasplits=dataset.DensityData("../000xxx.tar")

path = os.path.join("./data","000001.CHGCAR.lz4")
filecontent = _decompress_file(path)

density, atoms, origin = _read_vasp(filecontent)

#print(np.linalg.inv(atoms.get_cell().complete().T))
#exit()
grid_pos = _calculate_grid_pos(density, origin, atoms.get_cell())
print(grid_pos)

num_pos = np.prod(grid_pos.shape[0:3])
print(num_pos)

probe_count=5000

num_slices = int(math.ceil(num_pos / probe_count))
print(num_slices)

meshgrid = grid_pos
num_positions = num_pos
#slice_index = num_slices
slice_index = 0

#print(num_positions)
#print(slice_index)
#print(probe_count)
#print(num_positions)
#print(slice_index*probe_count)
#print(min((slice_index+1)*probe_count, num_positions))
#for slice_index in range(0,num_slices):
slice_index = 0
#print(slice_index)
flat_index = np.arange(slice_index*probe_count, min((slice_index+1)*probe_count, num_positions))
#print(flat_index)
pos_index = np.unravel_index(flat_index, meshgrid.shape[0:3])
probe_pos = meshgrid[pos_index]
#neighbor
cutoff=4
neighborlist=None
probe_edges, probe_edges_displacement = probes_to_graph(atoms, probe_pos, cutoff, neighborlist)
print(probe_edges)
print(probe_edges_displacement)
print(pos_index)
print(probe_pos)
print(meshgrid.shape[0:3])
print(pos_index)
