import os
import sys
import timeit
from shutil import copyfile
import numpy as np
import ase.db
from ase.calculators.vasp import Vasp

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

chunk_id = int(sys.argv[1])

indices = np.arange(1000) + chunk_id*1000

os.chdir("data")

with ase.db.connect("../qm9.db") as db:
    for row in db.select():
        if row.index in indices:
            atoms = row.toatoms()
            atoms.center(vacuum=2.0)
            atoms.set_pbc(True)

            new_dir = str(row.index)
            os.mkdir(new_dir)
            start = timeit.default_timer()
            with cd(new_dir):
                calc1 = Vasp(xc = 'PBE',
                                    istart =0,
                                    algo = 'Normal',
                                    icharg= 2,
                                    nelm = 180,
                                    ispin = 1,
                                    nelmdl = 6,
                                    isym = 0,
                                    lcorr = True,
                                    potim = 0.1,
                                    nelmin = 5,
                                    kpts = [1,1,1],
                                    ismear = 0,
                                    ediff  = 0.1E-05,
                                    sigma = 0.1,
                                    nsw =0,
                                    ldiag = True,
                                    lreal = 'Auto',
                                    lwave = False,
                                    lcharg = True,
                                    encut = 400)

                atoms.set_calculator(calc1)
                e = atoms.get_potential_energy()
            stop = timeit.default_timer()
            print("index=%d, %.1f" % (row.index, stop-start))
