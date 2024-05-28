import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
warnings.filterwarnings('ignore')
import numpy as np
import shutil
import pandas as pd
import os
import sys
sys.path.append(os.getcwd())
from inp_params import train_e,ert_epochs,ert_batch_size,ert_patience,train_dos,drt_epochs,drt_batch_size,drt_patience,test_chg,test_e,test_dos,plot_dos,write_chg,comp_chg,ref_chg,grid_spacing,batch_size_fp, num_gamma, cut_off_rad, widest_gaussian, narrowest_gaussian,fp_file,tot_chg
from sklearn.metrics import mean_absolute_error

import time
import h5py
from keras.preprocessing.sequence import pad_sequences

import pymatgen
from pymatgen import io
from pymatgen.io.vasp.outputs import Poscar
import os
sys.path.insert(0, '/home/jianhuan/haochen/ML-DFT-TF114/ML-DFT-main')
elec_dict={6:4,1:1,7:5, 8:6}

from MLDFT.src.FPtf import fp_atom,fp_chg_norm,fp_norm
from MLDFT.src.CHG import init_chgmod,chg_predict,chg_ref,chg_pred_data,chg_pts,chg_print,chg_train,chg_dat_prep,coef_predict
from MLDFT.src.Energy import init_Emod,energy_predict,e_train,retrain_emodel
from MLDFT.src.DOS import init_DOSmod,DOS_pred, DOS_plot,retrain_dosmodel
from MLDFT.src.data_io import get_def_data, get_all_data, get_efp_data, dos_mask,get_e_dos_data, get_dos_data, get_dos_e_train_data,pad_efp_data,pad_dos_dat,pad_dat,chg_data


class Input_parameters:
    train_e=train_e
    ert_epochs=ert_epochs
    ert_batch_size=ert_batch_size
    ert_patience=ert_patience
    train_dos=train_dos
    drt_epochs=drt_epochs
    drt_batch_size=drt_batch_size
    drt_patience=drt_patience
    test_chg=test_chg
    tot_chg=tot_chg
    test_e=test_e
    test_dos=test_dos
    plot_dos=plot_dos
    write_chg=write_chg
    ref_chg=ref_chg
    grid_spacing=grid_spacing
    cut_off_rad = cut_off_rad
    batch_size_fp = batch_size_fp
    widest_gaussian = widest_gaussian
    narrowest_gaussian = narrowest_gaussian
    num_gamma = num_gamma
    fp_file=fp_file

inp_args=Input_parameters()

df_test = pd.read_csv("predict.csv")
inp_args.file_loc_test = df_test['file_loc_test'].tolist()
file= [x for x in inp_args.file_loc_test  if str(x) != 'nan']
print(file)
file_loc = file[0]

poscar_file = os.path.join(file_loc,"POSCAR")
poscar_data=Poscar.from_file(poscar_file)
vol = poscar_data.structure.volume
supercell = poscar_data.structure
dim=supercell.lattice.matrix
atoms=supercell.num_sites
elems_list = sorted(list(set(poscar_data.site_symbols)))

electrons_list = [elec_dict[x] for x in list(poscar_data.structure.atomic_numbers)]
inp_args.total_elec = sum(electrons_list)
dset,basis_mat,sites_elem,num_atoms,at_elem=fp_atom(poscar_data,supercell,elems_list)
total_elec=inp_args.total_elec
dataset1 = dset[:]

print('Total number of electrons inside cell:',total_elec)
i1=at_elem[0]
i2=at_elem[1]
i3=at_elem[2]
i4=at_elem[3]
padding_size=max([i1,i2,i3,i4])
num_atoms=np.array(dataset1.shape[0])

X_3D1,X_3D2,X_3D3,X_3D4,basis1,basis2,basis3,basis4,C_m,H_m,N_m,O_m=chg_data(dataset1,basis_mat,i1,i2,i3,i4,padding_size)
modelCHG=init_chgmod(padding_size)
Coef_at1,Coef_at2,Coef_at3, Coef_at4,C_at_charge, H_at_charge, N_at_charge, O_at_charge=chg_predict(X_3D1,X_3D2,X_3D3,X_3D4,i1,i2,i3,i4,sites_elem,modelCHG,at_elem)
print('Atomic charges for the C atoms (same order as in POSCAR):', C_at_charge)
print('Atomic charges for the H atoms (same order as in POSCAR):', H_at_charge)
if i3!= 0:
    print('Atomic charges for the N atoms (same order as in POSCAR):', N_at_charge)
if i4!=0:
    print('Atomic charges for the O atoms (same order as in POSCAR):', O_at_charge)
localfile_loc = file_loc.replace("/", "_")
print("Writing atomic charges to text files...")
np.savetxt("C_charges" + localfile_loc + ".txt",np.c_[C_at_charge])
np.savetxt("H_charges" + localfile_loc + ".txt",np.c_[H_at_charge])
if i3!= 0:
    np.savetxt("N_charges" + localfile_loc + ".txt",np.c_[N_at_charge])
if i4!=0:
    np.savetxt("O_charges" + localfile_loc + ".txt",np.c_[O_at_charge])
if test_e or test_dos:
    X_C,X_H,X_N,X_O=fp_chg_norm(Coef_at1,Coef_at2,Coef_at3,Coef_at4,X_3D1,X_3D2,X_3D3,X_3D4,padding_size)

if test_dos:
    modelD=init_DOSmod(padding_size)
    C_d,H_d,N_d,O_d=dos_mask(C_m,H_m,N_m,O_m,padding_size)

    modelD.load_weights("/home/jianhuan/haochen/ML-DFT-TF114/ML-DFT-main/MLDFT/Trained_models/weights_DOS.hdf5")



    """
    Pred, uncertainty,VB,devVB,CB,devCB,BG,devBG=DOS_pred(X_C,X_H,X_N,X_O,np.array(total_elec).reshape(1,1),C_d,H_d,N_d,O_d,modelD)
  
    DOS=np.squeeze(Pred)
    print('Valence band maximum:', VB, '+-', devVB, ' eV')
    print('Conduction band minimum:', CB, '+-', devCB, ' eV')
    print('Bandgap:', BG, '+-', devBG, ' eV')
    energy_wind=np.arange(-33.0,1.1,0.1)
    print("Writing DOS curve to text file...")
    np.savetxt("DOS" + localfile_loc + ".txt",np.c_[energy_wind,DOS])
    if plot_dos:
        DOS_plot(energy_wind,DOS,VB,CB,uncertainty,localfile_loc)
"""























