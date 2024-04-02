import os
import torch
import dataset
from dataset import _decompress_file,_read_vasp,_calculate_grid_pos
import numpy as np
from ase.neighborlist import NeighborList,NewPrimitiveNeighborList

#with open("../qm9vasp.txt", "r") as datasetfiles:
#    filelist = [os.path.join(os.path.dirname("../qm9vasp.txt"), line.strip('\n')) for line in datasetfiles]

#densitydata = torch.utils.data.ConcatDataset([dataset.DensityData(path) for path in filelist])
#datasplits=dataset.DensityData("../000xxx.tar")

path = os.path.join("./data","000101.CHGCAR.lz4")
filecontent = _decompress_file(path)

density, atoms, origin = _read_vasp(filecontent)

print(atoms.get_pbc())
print(atoms.get_cell())
print(atoms.get_positions())

#atoms_positions = atoms.get_positions()
#atoms_cell = atoms.get_cell()
#atoms=build(atoms.get_pbc(), atoms.get_cell(), atoms.get_positions())
nl=NewPrimitiveNeighborList(1)
#atoms=nl.build(atoms.get_pbc(), atoms.get_cell(), atoms.get_positions())
nl.update(atoms.get_pbc(), atoms.get_cell(), atoms.get_positions())
indices, offsets = nl.get_neighbors(1)
atoms_positions = atoms.get_positions()
atoms_cell = atoms.get_cell()

print(indices)
#print(atoms_positions[1])
rel_positions = (atoms_positions[indices] + offsets @ atoms_cell - atoms_positions[1][None])

dist2 = np.sum(np.square(rel_positions), axis=1)

print(rel_positions)
print(dist2)
exit()

#print(indices)
#print(offsets)

#aa=np.array(density.shape)

#print(density.shape[0])

grid_pos = _calculate_grid_pos(density, origin, atoms.get_cell())
print(grid_pos)
print(len(grid_pos))

#print(atoms.get_positions())
#print(grid_pos)


#print(origin)

#print(density)


#for densitydataset in datasplits:
#    print(densitydataset)
#print(data)
#print(densitydata)
#print(filelist)
