#!/bin/sh
#yhrun -N 1 -p GPU_A800 python runner.py --dataset ./data/data/ECMD/ethylenecarbonate.txt --split_file ./data/data/ECMD/splits.json --output_dir ./result/ecmd_0216  --cutoff 4 --num_interactions 3 --use_painn_model --max_steps 100000000 --node_size 128  --batch_size 10 --device cuda:1
#python runner.py --dataset ./data/qm9/qm9vasp_test.txt --split_file ./data/qm9/datasplits_test.json --output_dir ./results/qm9_0412  --cutoff 4 --num_interactions 6 --use_painn_model --max_steps 100000000 --node_size 128  --batch_size 4 --device cuda:0
python runner.py --dataset ./data/qm9/qm9vasp_test.txt --split_file ./data/qm9/datasplits_test.json --output_dir ./results/qm9_0412  --cutoff 4 --num_interactions 6 --use_painn_model --max_steps 100000000 --node_size 128 
