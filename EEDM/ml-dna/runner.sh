#!/bin/bash

python train_dna.py --datapath /data1/cenjianhuan/DNAdatabase/trainset_np.pkl \
                    --testpath /data1/cenjianhuan/DNAdatabase/testset_np.pkl \
                    --ckpt_savepath ./checkpoints \
                    --split 100 \
                    --epochs 251 \
                    --batch 2 \
                    --device GPU \
                    --device_id 5
