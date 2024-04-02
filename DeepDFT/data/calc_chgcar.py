import numpy
import sys
import os
from ase.calculators.vasp import Vasp
from ase.io import trajectory

class cd:                                                                                                                                                      
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)
 
    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)
 
    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def main():
    slice_id = int(sys.argv[1])
    traj = trajectory.TrajectoryReader("3000klang2.traj")
    for i in range(slice_id*500, (slice_id+1)*500):
        dirname = "res/%05d" % i
        os.mkdir(dirname)
        atoms = traj[i]
        with cd(dirname):
            calc1 = Vasp(xc = 'PBE',
                         istart =0,
                         algo = 'VeryFast',
                         nelm = 180,
                         ispin = 1,
                         nelmdl = 6,
                         isym = 0,
                         lcorr = True,
                         potim = 0.1,
                         nelmin = 5,
                         kpar = 4,
                         kpts = [2,2,2],
                         ismear = 0,
                         ediff  = 0.1E-03,
                         ediffg = -0.01,
                         sigma = 0.1,
                         nsw = 0,
                         isif = 2,
                         ibrion = 2,
                         ldiag = True,
                         lreal = 'Auto',
                         lwave = False,
                         lcharg = True,
                         prec = 'Normal')
            try:
                atoms.set_calculator(calc1)
                e = atoms.get_potential_energy()
            except RuntimeError:
                print("ERROR in %s" % dirname)
        print(dirname)

if __name__ == "__main__":
    main()
