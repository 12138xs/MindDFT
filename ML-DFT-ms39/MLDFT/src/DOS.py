import warnings
import os
import numpy as np
import sys
import math
from inp_params import test_chg,write_chg,grid_spacing
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.train import Model

import pymatgen
from pymatgen import io
from pymatgen.io.vasp.outputs import Poscar

import random
import gc
from random import sample
import json
from operator import itemgetter
import h5py


import glob
import shutil

class single_atom_modelC_1(nn.Cell):
    def __init__(self):
        super(single_atom_modelC_1, self).__init__()
        self.dense1 = nn.Dense(700,600, activation='relu')#weight_decay=0.1, kernel_initializer='glorot_uniform'
        self.dense2 = nn.Dense(600,600, activation='relu')
        self.dense3 = nn.Dense(600,343, activation='relu')
        self.dropout = nn.Dropout(p=0.1)
        self.conv1 = nn.Conv1d(1, 3, 3,pad_mode="valid")
        self.relu = nn.ReLU()
    def construct(self,modelC_in):
        modelC_out = self.dense1(modelC_in)
        modelC_out = self.dropout(modelC_out)
        modelC_out = self.dense2(modelC_out)
        modelC_out = self.dropout(modelC_out)
        modelC_out = self.dense2(modelC_out)
        modelC_out = self.dropout(modelC_out)
        modelC_out = self.dense2(modelC_out)
        modelC_out = self.dropout(modelC_out)
        modelC_out = self.dense3(modelC_out)
        modelC_out = ops.reshape(modelC_out,(-1,1,343))
        modelC_out = self.conv1(modelC_out)
        modelC_out = self.relu(modelC_out)
        modelC_out = ops.mean(modelC_out,axis=1)

        return modelC_out

class single_atom_modelH_1(nn.Cell):
    def __init__(self):
        super(single_atom_modelH_1, self).__init__()
        self.dense1 = nn.Dense(568,600, activation='relu')#weight_decay=0.1, kernel_initializer='glorot_uniform'
        self.dense2 = nn.Dense(600,600, activation='relu')
        self.dense3 = nn.Dense(600,343, activation='relu')
        self.dropout = nn.Dropout(p=0.1)
        self.conv1 = nn.Conv1d(1, 3, 3, pad_mode="valid")
        self.relu = nn.ReLU()

    def construct(self,modelH_in):
        modelH_out = self.dense1(modelH_in)
        modelH_out = self.dropout(modelH_out)
        modelH_out = self.dense2(modelH_out)
        modelH_out = self.dropout(modelH_out)
        modelH_out = self.dense2(modelH_out)
        modelH_out = self.dropout(modelH_out)
        modelH_out = self.dense2(modelH_out)
        modelH_out = self.dropout(modelH_out)
        modelH_out = self.dense3(modelH_out)
        modelH_out = ops.reshape(modelH_out, (-1,1,343))
        modelH_out = self.conv1(modelH_out)
        modelH_out = self.relu(modelH_out)
        modelH_out = ops.mean(modelH_out, axis=1)
        return modelH_out

class single_atom_modelN_1(nn.Cell):
    def __init__(self):
        super(single_atom_modelN_1, self).__init__()
        self.dense1 = nn.Dense(700,600, activation='relu')#weight_decay=0.1, kernel_initializer='glorot_uniform'
        self.dense2 = nn.Dense(600,600, activation='relu')
        self.dense3 = nn.Dense(600,343, activation='relu')
        self.dropout = nn.Dropout(p=0.1)
        self.conv1 = nn.Conv1d(1, 3, 3,pad_mode="valid")
        self.relu = nn.ReLU()
    def construct(self,modelN_in):
        modelN_out = self.dense1(modelN_in)
        modelN_out = self.dropout(modelN_out)
        modelN_out = self.dense2(modelN_out)
        modelN_out = self.dropout(modelN_out)
        modelN_out = self.dense2(modelN_out)
        modelN_out = self.dropout(modelN_out)
        modelN_out = self.dense2(modelN_out)
        modelN_out = self.dropout(modelN_out)
        modelN_out = self.dense3(modelN_out)
        modelN_out = ops.reshape(modelN_out, (-1,1,343))
        modelN_out = self.conv1(modelN_out)
        modelN_out = self.relu(modelN_out)
        modelN_out = ops.mean(modelN_out, axis=1)
        return modelN_out

class single_atom_modelO_1(nn.Cell):
    def __init__(self):
        super(single_atom_modelO_1, self).__init__()
        self.dense1 = nn.Dense(700,600, activation='relu')#weight_decay=0.1, kernel_initializer='glorot_uniform'
        self.dense2 = nn.Dense(600,600, activation='relu')
        self.dense3 = nn.Dense(600,343, activation='relu')
        self.dropout = nn.Dropout(p=0.1)
        self.conv1 = nn.Conv1d(1, 3, 3,pad_mode="valid")
        self.relu = nn.ReLU()
    def construct(self,modelO_in):
        modelO_out = self.dense1(modelO_in)
        modelO_out = self.dropout(modelO_out)
        modelO_out = self.dense2(modelO_out)
        modelO_out = self.dropout(modelO_out)
        modelO_out = self.dense2(modelO_out)
        modelO_out = self.dropout(modelO_out)
        modelO_out = self.dense2(modelO_out)
        modelO_out = self.dropout(modelO_out)
        modelO_out = self.dense3(modelO_out)
        modelO_out = ops.reshape(modelO_out, (-1,1,343))
        modelO_out = self.conv1(modelO_out)
        modelO_out = self.relu(modelO_out)
        modelO_out = ops.mean(modelO_out, axis=1)
        return modelO_out


class modelDOS(nn.Cell):
    def __init__(self,padding_size):
        super(modelDOS, self).__init__()
        self.time_distributed_1 = nn.TimeDistributed(single_atom_modelC_1(),time_axis=1, reshape_with_axis=0)
        self.time_distributed_2 = nn.TimeDistributed(single_atom_modelH_1(),time_axis=1, reshape_with_axis=0)
        self.time_distributed_3 = nn.TimeDistributed(single_atom_modelN_1(),time_axis=1, reshape_with_axis=0)
        self.time_distributed_4 = nn.TimeDistributed(single_atom_modelO_1(),time_axis=1, reshape_with_axis=0)
        self.dense1 = nn.Dense(341,100, activation='relu')
        self.dense2 = nn.Dense(100, 100, activation='relu')
        self.dense3 = nn.Dense(100, 2, activation='relu')
        self.padding_size = padding_size

    def construct(self,input1,input2,input3,input4,input5,input6,input7,input8,input9):
        model_out_C1 = self.time_distributed_1(input1)
        model_out_H1 = self.time_distributed_2(input2)
        model_out_N1 = self.time_distributed_3(input3)
        model_out_O1 = self.time_distributed_4(input4)
        D_C = ops.multiply(input6, model_out_C1)
        D_H = ops.multiply(input7, model_out_H1)
        D_N = ops.multiply(input8, model_out_N1)
        D_O = ops.multiply(input9, model_out_O1)
        model_added1 = ops.addn([D_C,D_H,D_N,D_O])

        model_s1 = ops.sum(model_added1,dim=1)
        model_dos = model_s1/input5
        bands = self.dense1(model_dos)
        bands = self.dense2(bands)
        bands = self.dense3(bands)

        return model_s1,bands

"""
i1 = ops.ones((7,6,700))
i2 = ops.ones((7,6,568))
i3 = ops.ones((7,6,700))
i4 = ops.ones((7,6,700))
i5 = ops.ones((1,))
i6 = ops.ones((7,6,341))
i7 = ops.ones((7,6,341))
i8 = ops.ones((7,6,341))
i9 = ops.ones((7,6,341))
model = modelDOS(6)
y1,y2 = model(i1,i2,i3,i4,i5,i6,i7,i8,i9)
print(y1.shape)
print(y2.shape)
print(model)
"""

def init_DOSmod(padding_size):
    model_CHG = modelDOS(padding_size)
    optim = nn.Adam(params=model_CHG.trainable_params(), learning_rate=0.0001,beta1=0.9, beta2=0.999)
    loss = nn.MSELoss()
    model = Model(model_CHG, loss_fn=loss, optimizer=optim)
    return model

def Dmodel_weights(train_dos,new_weights_dos,modelDOS):
    if train_dos:
        param_dict = ms.load_checkpoint("newDOSmodel.ckpt")
        ms.load_param_into_net(modelDOS, param_dict)
    elif new_weights_dos:
        param_dict = ms.load_checkpoint("newDOSmodel.ckpt")
        ms.load_param_into_net(modelDOS, param_dict)
    else:
        param_dict = ms.load_checkpoint(CONFIG_PATH1)
        ms.load_param_into_net(modelDOS, param_dict)

def DOS_pred(X_C, X_H, X_N, X_O, total_elec, C_d, H_d, N_d, O_d, modelDOS):
    resultD = []
    resultvbcb = []
    Dmodel_weights(train_dos, new_weights_dos, modelDOS)
    for i in range(100):
        Pred, vbcb = modelDOS([X_C, X_H, X_N, X_O, total_elec, C_d, H_d, N_d, O_d], batch_size=1)
        resultD.append(Pred * total_elec)
        resultvbcb.append(vbcb)
    resultD = np.array(resultD)
    Pred = resultD.mean(axis=0)
    resultvbcb = np.array(resultvbcb)
    devVB = resultvbcb.std(axis=0)[0][0]
    devCB = resultvbcb.std(axis=0)[0][1]
    Pred_vb = (-1) * resultvbcb.mean(axis=0)[0][0]
    Pred_cb = (-1) * resultvbcb.mean(axis=0)[0][1]
    uncertainty = resultD.std(axis=0)
    uncertainty = np.squeeze(uncertainty)
    resultvbcb = np.vstack(resultvbcb)
    Bandgap = Pred_cb - Pred_vb
    devBG = resultvbcb[:, 0] - resultvbcb[:, 1]
    devBG = devBG.std(axis=0)
    return Pred, uncertainty, Pred_vb, devVB, Pred_cb, devCB, Bandgap, devBG


def DOS_plot(energy_wind, Pred, VB, CB, uncertainty, localfile_loc):
    plt.plot(energy_wind, Pred, "r-", label='DOS', linewidth=1)
    plt.fill_between(energy_wind, Pred - uncertainty, Pred + uncertainty, color='gray', alpha=0.2)
    plt.axvline(VB, color='b', label='Valence band', linestyle=':')
    plt.axvline(CB, color='g', label='Conduction band', linewidth=1)
    plt.axvline(0, color='k', label='Vacuum level', linestyle='dashed', linewidth=2)
    plt.legend(fontsize=16)
    plt.tick_params(labelsize=16)
    plt.xlabel("Energy (eV)", fontsize=18)
    plt.ylabel("DOS", fontsize=18)
    plt.tight_layout()
    plt.savefig("dos_" + localfile_loc + " .png", dpi=500)
    plt.clf()
    print("Made the dos plot ..")


def dos_data(file_loc, total_elec):
    dos_file = os.path.join(file_loc, "dos")
    dos_data = [[float(s) for s in l.split()] for l in open(dos_file).readlines()]
    dos = []
    for i in range(len(dos_data)):
        dos.append(dos_data[i][0])
    levels_file = os.path.join(file_loc, "VB_CB")
    levels_data = [[float(s) for s in l.split()] for l in open(levels_file).readlines()]
    VB = abs(np.array(levels_data[0]))
    CB = abs(np.array(levels_data[1]))
    Prop = np.array(dos) / total_elec
    return Prop, VB, CB


def retrain_dosmodel(X_C, X_H, X_N, X_O, X_el, C_d, H_d, N_d, O_d, Prop_dos, vbcb, X_val_C, X_val_H, X_val_N, X_val_O,
                     X_el_val, C_dV, H_dV, N_dV, O_dV, Prop_dos_val, vbcb_val, drt_epochs, drt_batch_size, drt_patience,
                     padding_size):
    filepath = "newDOSmodel.ckpt"
    rtmodel = init_DOSmod(padding_size)
    param_dict = ms.load_checkpoint(CONFIG_PATH1)
    ms.load_param_into_net(rtmodel, param_dict)
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True,
                                 mode='min')
    early_stopping_cb = EarlyStopping(patience=drt_patience, restore_best_weights=True)
    callbacks_list = [checkpoint, early_stopping_cb]
    history = rtmodel.fit(drt_epochs, train_dataset, valid_dataset)

    history = rtmodel.fit([X_C, X_H, X_N, X_O, X_el, C_d, H_d, N_d, O_d], [Prop_dos, vbcb], epochs=drt_epochs,
                          batch_size=drt_batch_size, shuffle=True, validation_data=(
        [X_val_C, X_val_H, X_val_N, X_val_O, X_el_val, C_dV, H_dV, N_dV, O_dV], [Prop_dos_val, vbcb_val]),
                          callbacks=callbacks_list)





























