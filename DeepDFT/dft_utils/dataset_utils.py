import os
import json
import math
import numpy as np
from dataset import DensityData

def get_data_files(data_path)->list:
    # Setup dataset and loader
    # --Why not use nargs in argparser? 

    if data_path.endswith(".txt"): #"../data/qm9/qm9vasp_test.txt"
        # Text file contains list of datafiles
        with open(data_path, "r") as datasetfiles:
            filelist = [os.path.join(os.path.dirname(data_path), line.strip('\n')) for line in datasetfiles]
    else:
        filelist = [data_path]
    return filelist

def concat_data_files(file_list:list)->DensityData:
    densitydata = DensityData(file_list[0])

    if len(file_list) > 1:
        for file in file_list[1:]:
            densitydata.concat(DensityData(file))
    
    return densitydata

def split_data(dataset:DensityData, args):
    # Load or generate splits
    if args.split_file:
        with open(args.split_file, "r") as fp:
            splits = json.load(fp)
    else:
        datalen = len(dataset)
        num_validation = int(math.ceil(datalen * 0.05))
        indices = np.random.permutation(len(dataset))
        splits = {
            "train": indices[num_validation:].tolist(),
            "validation": indices[:num_validation].tolist(),
        }

        # Save split file
        with open(os.path.join(args.output_dir, "datasplits.json"), "w") as f:
            json.dump(splits, f)

    # Split the dataset
    datasplits = {}
    for key, indices in splits.items():
        datasplits[key] = dataset.take(indices)
        #datasplits[key] = [dataset[idx] for idx in indices]
    return datasplits
'''
def concat_data_files(file_list)->DensityData:
    for i in range(0, len(file_list)):
        if i==0:
            densitydata = DensityData(file_list[0])
        else:
            densitydata.concat(DensityData(file_list[i]))
'''

