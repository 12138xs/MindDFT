import warnings
import os
import sys

import argparse
import numpy as np
import time

from inp_params import test_chg,write_chg,grid_spacing
import matplotlib
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.train import Model

import pymatgen
from pymatgen import io
from pymatgen.io.vasp.outputs import Poscar
from pymatgen.io.vasp.outputs import Chgcar

from random import Random
from operator import itemgetter
import h5py

import glob
import shutil


class single_atom_modelC_E(nn.Cell):
    def __init__(self):
        super(single_atom_modelC_E, self).__init__()
        self.dense1 = nn.Dense(700,300, activation='tanh')#weight_decay=0.1, kernel_initializer='glorot_uniform'
        self.dense2 = nn.Dense(300,300, activation='tanh')
        self.dense3 = nn.Dense(300,10)

    def construct(self,model_input):
        model_out, basisC = ops.split(model_input,[700,9], axis=-1)
        basisC = ops.reshape(basisC,(-1,3,3))
        TbasisC = ops.transpose(basisC,(0,2,1))

        model_out = self.dense1(model_out)
        model_out = self.dense2(model_out)
        model_out = self.dense2(model_out)
        model_out = self.dense2(model_out)
        model_out = self.dense2(model_out)
        model_out = self.dense3(model_out)
        E, forces, XX, YY, ZZ, XY, YZ, XZ = ops.split(model_out,[1,3,1,1,1,1,1,1],axis=-1)
        Energy = ops.abs(E)
        forces = ops.reshape(forces,(-1,3,1))
        forces_out = ops.matmul(basisC, forces)
        forces_out = ops.reshape(forces_out,(-1,3))
        model_out = ops.concat((XX,XY,XZ,XY,YY,YZ,XZ,YZ,ZZ),axis=-1)
        model_out = ops.reshape(model_out, (-1, 3, 3))
        model_out = ops.matmul(basisC, model_out)
        model_out = ops.matmul(model_out,TbasisC)
        model_out = ops.reshape(model_out,(-1,9))
        XX,XY,XZ,YX,YY,YZ,ZX,ZY,ZZ = ops.split(model_out, [1,1,1,1,1,1,1,1,1], axis=-1)
        model_out = ops.concat([Energy,forces_out,XX,YY,ZZ,XY,YZ,XZ],axis=-1)
        return model_out

class single_atom_modelH_E(nn.Cell):
    def __init__(self):
        super(single_atom_modelH_E, self).__init__()
        self.dense1 = nn.Dense(568,300, activation='tanh')#weight_decay=0.1, kernel_initializer='glorot_uniform'
        self.dense2 = nn.Dense(300,300, activation='tanh')
        self.dense3 = nn.Dense(300,10)

    def construct(self,model_input):
        model_out, basisH = ops.split(model_input,[568,9], axis=-1)
        basisH = ops.reshape(basisH, (-1, 3, 3))
        TbasisH = ops.transpose(basisH,(0,2,1))

        model_out = self.dense1(model_out)
        model_out = self.dense2(model_out)
        model_out = self.dense2(model_out)
        model_out = self.dense2(model_out)
        model_out = self.dense2(model_out)
        model_out = self.dense3(model_out)
        E, forces, XX, YY, ZZ, XY, YZ, XZ = ops.split(model_out,[1,3,1,1,1,1,1,1],axis=-1)
        Energy = ops.abs(E)
        forces = ops.reshape(forces,(-1,3,1))
        forces_out = ops.matmul(basisH, forces)
        forces_out = ops.reshape(forces_out,(-1,3))
        model_out = ops.concat((XX,XY,XZ,XY,YY,YZ,XZ,YZ,ZZ),axis=-1)
        model_out = ops.reshape(model_out, (-1, 3, 3))
        model_out = ops.matmul(basisH, model_out)
        model_out = ops.matmul(model_out,TbasisH)
        model_out = ops.reshape(model_out,(-1,9))
        XX,XY,XZ,YX,YY,YZ,ZX,ZY,ZZ = ops.split(model_out, [1,1,1,1,1,1,1,1,1], axis=-1)
        model_out = ops.concat([Energy,forces_out,XX,YY,ZZ,XY,YZ,XZ],axis=-1)
        return model_out

class single_atom_modelN_E(nn.Cell):
    def __init__(self):
        super(single_atom_modelN_E, self).__init__()
        self.dense1 = nn.Dense(700,300, activation='tanh')#weight_decay=0.1, kernel_initializer='glorot_uniform'
        self.dense2 = nn.Dense(300,300, activation='tanh')
        self.dense3 = nn.Dense(300,10)

    def construct(self,model_input):
        model_out, basisN = ops.split(model_input, [700, 9], axis=-1)
        basisN = ops.reshape(basisN, (-1, 3, 3))
        TbasisN = ops.transpose(basisN, (0, 2, 1))

        model_out = self.dense1(model_out)
        model_out = self.dense2(model_out)
        model_out = self.dense2(model_out)
        model_out = self.dense2(model_out)
        model_out = self.dense2(model_out)
        model_out = self.dense3(model_out)
        E, forces, XX, YY, ZZ, XY, YZ, XZ = ops.split(model_out,[1,3,1,1,1,1,1,1],axis=-1)
        Energy = ops.abs(E)
        forces = ops.reshape(forces,(-1,3,1))
        forces_out = ops.matmul(basisN, forces)
        forces_out = ops.reshape(forces_out,(-1,3))
        model_out = ops.concat((XX,XY,XZ,XY,YY,YZ,XZ,YZ,ZZ),axis=-1)
        model_out = ops.reshape(model_out, (-1, 3, 3))
        model_out = ops.matmul(basisN, model_out)
        model_out = ops.matmul(model_out,TbasisN)
        model_out = ops.reshape(model_out,(-1,9))
        XX,XY,XZ,YX,YY,YZ,ZX,ZY,ZZ = ops.split(model_out, [1,1,1,1,1,1,1,1,1], axis=-1)
        model_out = ops.concat([Energy,forces_out,XX,YY,ZZ,XY,YZ,XZ],axis=-1)
        return model_out

class single_atom_modelO_E(nn.Cell):
    def __init__(self):
        super(single_atom_modelO_E, self).__init__()
        self.dense1 = nn.Dense(700,300, activation='tanh')#weight_decay=0.1, kernel_initializer='glorot_uniform'
        self.dense2 = nn.Dense(300,300, activation='tanh')
        self.dense3 = nn.Dense(300,10)

    def construct(self,model_input):
        model_out, basisE = ops.split(model_input, [700, 9], axis=-1)
        basisE = ops.reshape(basisE, (-1, 3, 3))
        TbasisE = ops.transpose(basisE, (0, 2, 1))

        model_out = self.dense1(model_out)
        model_out = self.dense2(model_out)
        model_out = self.dense2(model_out)
        model_out = self.dense2(model_out)
        model_out = self.dense2(model_out)
        model_out = self.dense3(model_out)
        E, forces, XX, YY, ZZ, XY, YZ, XZ = ops.split(model_out,[1,3,1,1,1,1,1,1],axis=-1)
        Energy = ops.abs(E)
        forces = ops.reshape(forces,(-1,3,1))
        forces_out = ops.matmul(basisE, forces)
        forces_out = ops.reshape(forces_out,(-1,3))
        model_out = ops.concat((XX,XY,XZ,XY,YY,YZ,XZ,YZ,ZZ),axis=-1)
        model_out = ops.reshape(model_out, (-1, 3, 3))
        model_out = ops.matmul(basisE, model_out)
        model_out = ops.matmul(model_out,TbasisE)
        model_out = ops.reshape(model_out,(-1,9))
        XX,XY,XZ,YX,YY,YZ,ZX,ZY,ZZ = ops.split(model_out, [1,1,1,1,1,1,1,1,1], axis=-1)
        model_out = ops.concat([Energy,forces_out,XX,YY,ZZ,XY,YZ,XZ],axis=-1)
        return model_out

class modelEmod(nn.Cell):
    def __init__(self,padding_size):
        super(modelEmod, self).__init__()
        self.time_distributed_1 = nn.TimeDistributed(single_atom_modelC_E(),time_axis=1, reshape_with_axis=0)
        self.time_distributed_2 = nn.TimeDistributed(single_atom_modelH_E(),time_axis=1, reshape_with_axis=0)
        self.time_distributed_3 = nn.TimeDistributed(single_atom_modelN_E(),time_axis=1, reshape_with_axis=0)
        self.time_distributed_4 = nn.TimeDistributed(single_atom_modelO_E(),time_axis=1, reshape_with_axis=0)
        self.padding_size = padding_size

    def construct(self,input1,input2,input3,input4,input5,input6,input7,input8,input9):
        model_out_CP = self.time_distributed_1(input1)
        model_out_HP = self.time_distributed_2(input2)
        model_out_NP = self.time_distributed_3(input3)
        model_out_OP = self.time_distributed_4(input4)
        EC,forcesC,pressC = ops.split(model_out_CP, [1,3,6], axis=-1)
        EH,forcesH,pressH = ops.split(model_out_HP, [1,3,6], axis=-1)
        EN,forcesN,pressN = ops.split(model_out_NP, [1,3,6], axis=-1)
        EO,forcesO,pressO = ops.split(model_out_OP, [1,3,6], axis=-1)
        model_added = ops.addn([pressC, pressH, pressN, pressO])
        EC = ops.multiply(input6, EC)
        EH = ops.multiply(input7, EH)
        EN = ops.multiply(input8, EN)
        EO = ops.multiply(input9, EO)
        E_added = ops.addn([EC, EH, EN, EO])
        E_tot = ops.sum(E_added, dim=1)
        E_tot = E_tot / input5
        model_p = ops.sum(model_added, dim=1)
        model_p = model_p / input5
        return E_tot,forcesC,forcesH,forcesN,forcesO,model_p

def init_Emod(padding_size):
    model_CHG = modelEmod(padding_size)
    optim = nn.Adam(params=model_CHG.trainable_params(), learning_rate=0.0001,beta1=0.9, beta2=0.999)
    loss = nn.MSELoss()
    model = Model(model_CHG, loss_fn=loss, optimizer=optim)
    return model


"""
i1 = ops.ones((7,6,709))
i2 = ops.ones((7,6,577))
i3 = ops.ones((7,6,709))
i4 = ops.ones((7,6,709))
i5 = ops.ones((1,))
i6 = ops.ones((7,6,1))
i7 = ops.ones((7,6,1))
i8 = ops.ones((7,6,1))
i9 = ops.ones((7,6,1))
model = modelEmod(6)
y1,y2,y3,y4,y5,y6 = model(i1,i2,i3,i4,i5,i6,i7,i8,i9)
print(y1.shape)
print(y2.shape)
print(y3.shape)
print(y4.shape)
print(y5.shape)
print(y6.shape)
"""
def model_weights(train_e,new_weights_e,model_E):
    if train_e:
        model_E.load_weights('newEmodel.hdf5')
    elif new_weights_e:
        model_E.load_weights('newEmodel.hdf5')
    else:
        model_E.load_weights(CONFIG_PATH)


def energy_predict(X_C,X_H,X_N,X_O,basis1,basis2,basis3,basis4,C_m,H_m,N_m,O_m,num_atoms,model_E):
    X_C=np.concatenate((X_C,basis1), axis=-1)
    X_H=np.concatenate((X_H,basis2), axis=-1)
    X_N=np.concatenate((X_N,basis3), axis=-1)
    X_O=np.concatenate((X_O,basis4), axis=-1)
    model_weights(train_e,new_weights_e,model_E)
    E,ForC,ForH,ForN,ForO,pred_press=model_E.predict([X_C,X_H,X_N,X_O,num_atoms,C_m,H_m,N_m,O_m], batch_size=1)
    Pred_Energy=(-1)*E
    ForC=np.squeeze(ForC)
    ForH=np.squeeze(ForH)
    ForN=np.squeeze(ForN)
    ForO=np.squeeze(ForO)
    pred_press=np.squeeze(pred_press)
    return Pred_Energy[0][0],ForC,ForH,ForN,ForO,pred_press

def e_train(file_loc,tot_atoms):
    levels_file=os.path.join(file_loc,"energy")
    levels_data=[[float(s) for s in l.split()] for l in open(levels_file).readlines()]
    Energy=abs(np.array(levels_data[0]))/tot_atoms
    forces_file=os.path.join(file_loc,"forces")
    forces_data=np.array([[float(s) for s in l.split()] for l in open(forces_file).readlines()])
    press_file=os.path.join(file_loc,"stress")
    press_data=np.array([[float(s) for s in l.split()] for l in open(press_file).readlines()])
    press_data=np.reshape(press_data,(6))
    press=np.reshape(press_data,(1,6))

    return Energy,forces_data,press

def retrain_emodel(X_C,X_H,X_N,X_O,C_m,H_m,N_m,O_m,basis1,basis2,basis3,basis4,X_at,ener_ref,forces1,forces2,forces3,forces4,press_ref,X_val_C,X_val_H,X_val_N,X_val_O,C_mV,H_mV,N_mV,O_mV,basis1V,basis2V,basis3V,basis4V,X_at_val,ener_val,forces1V,forces2V,forces3V,forces4V,press_val,ert_epochs,ert_batch_size,ert_patience,padding_size):
    X_C=np.concatenate((X_C,basis1), axis=-1)
    X_H=np.concatenate((X_H,basis2), axis=-1)
    X_N=np.concatenate((X_N,basis3), axis=-1)
    X_O=np.concatenate((X_O,basis4), axis=-1)
    X_val_C=np.concatenate((X_val_C,basis1V), axis=-1)
    X_val_H=np.concatenate((X_val_H,basis2V), axis=-1)
    X_val_N=np.concatenate((X_val_N,basis3V), axis=-1)
    X_val_O=np.concatenate((X_val_O,basis4V), axis=-1)
    filepath="newEmodel.hdf5"
    rtmodel = init_Emod(padding_size)
    rtmodel.load_weights(CONFIG_PATH)
    checkpoint=ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True,mode='min')
    early_stopping_cb = EarlyStopping(patience=ert_patience,restore_best_weights=True)
    callbacks_list = [checkpoint,early_stopping_cb]

    history=rtmodel.fit([X_C,X_H,X_N,X_O,X_at,C_m,H_m,N_m,O_m],[ener_ref,forces1,forces2,forces3,forces4,press_ref],epochs=ert_epochs, batch_size=ert_batch_size,shuffle=True,validation_data=([X_val_C,X_val_H,X_val_N,X_val_O,X_at_val,C_mV,H_mV,N_mV,O_mV],[ener_val,forces1V,forces2V,forces3V,forces4V,press_val]),callbacks=callbacks_list)
