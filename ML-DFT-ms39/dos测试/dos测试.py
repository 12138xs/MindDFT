import warnings
import numpy as np
import shutil
import pandas as pd

import os
import sys
#sys.path.append(os.getcwd())
from sklearn.metrics import mean_absolute_error

import time

import pymatgen
from pymatgen import io
from pymatgen.io.vasp.outputs import Poscar
import os

elec_dict={6:4,1:1,7:5, 8:6}
from MLDFT.src.FP import fp_atom,fp_chg_norm,fp_norm
from MLDFT.src.CHG import init_chgmod,chg_predict,chg_ref,chg_pred_data,chg_pts,chg_print,chg_train,chg_dat_prep,coef_predict
from MLDFT.src.Energy import init_Emod,energy_predict,e_train,retrain_emodel
from MLDFT.src.DOS import init_DOSmod,DOS_pred, DOS_plot,retrain_dosmodel
from MLDFT.src.data_io import get_def_data, get_all_data, get_efp_data, dos_mask,get_e_dos_data, get_dos_data, get_dos_e_train_data,pad_efp_data,pad_dos_dat,pad_dat,chg_data



df_train= pd.read_csv("Train.csv")
df_val=pd.read_csv("Val.csv")
train_list=df_train['files']
val_list=df_val['files']





X_pre_list,X_at,X_el,X_elem,Prop_dos,Prop_vbcb=get_dos_data(train_list)
XV_pre_list,X_at_val,X_el_val,X_elem_val,Prop_dos_val,Prop_vbcb_val=get_dos_data(val_list)
padding_size=max(np.amax(X_elem),np.amax(X_elem_val))
padding_size = int(padding_size)
X_1,X_2,X_3,X_4,C_m,H_m,N_m,O_m=pad_dat(X_elem,X_pre_list,padding_size)
X_1V,X_2V,X_3V,X_4V,C_mV,H_mV,N_mV,O_mV=pad_dat(X_elem_val,XV_pre_list,padding_size)
vbcb,C_d,H_d,N_d,O_d=pad_dos_dat(Prop_vbcb,X_1,C_m,H_m,N_m,O_m,padding_size)
vbcb_val,C_dV,H_dV,N_dV,O_dV=pad_dos_dat(Prop_vbcb_val,X_1V,C_mV,H_mV,N_mV,O_mV,padding_size)
modelCHG=init_chgmod(padding_size)
X_C,X_H,X_N,X_O=get_dos_e_train_data(X_1,X_2,X_3,X_4,X_elem,padding_size,modelCHG)
X_val_C,X_val_H,X_val_N,X_val_O=get_dos_e_train_data(X_1V,X_2V,X_3V,X_4V,X_elem_val,padding_size,modelCHG)
X_C,X_H,X_N,X_O=fp_norm(X_C,X_H,X_N,X_O,padding_size)
X_val_C,X_val_H,X_val_N,X_val_O=fp_norm(X_val_C,X_val_H,X_val_N,X_val_O,padding_size)
retrain_dosmodel(X_C,X_H,X_N,X_O,X_el,C_d,H_d,N_d,O_d,Prop_dos,vbcb,X_val_C,X_val_H,X_val_N,X_val_O,X_el_val,C_dV,H_dV,N_dV,O_dV,Prop_dos_val,vbcb_val,drt_epochs,drt_batch_size, drt_patience,padding_size)


