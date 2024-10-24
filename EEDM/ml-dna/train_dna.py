import math
import os
import time
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
import mindspore.ops as ops
from mindspore import nn, context, Tensor
from mindspore import save_checkpoint
from gate_points_2101_ms import Network
from mindchemistry.e3 import o3
import random
import timeit
import argparse
from utils import get_iso_permuted_dataset, get_scalar_density_comparisons, collate_list_of_dicts


class ms_train():
    def __init__(self, loss_fn, Rs, b):
        self.loss_fn = loss_fn
        self.Rs = Rs
        self.b = b
        self.minMAE = np.inf
        self.minMUE = np.inf
    
    def train_epoch(self, epoch, model, optimizer, data_loader):
        model.set_train()
        loss_cum = 0.0
        mae_cum = 0.0
        mue_cum = 0.0
        train_num_ele = []

        # Define forward function
        def forward_fn(data, label):
            mask = ops.select(label == 0, ops.zeros_like(label), ops.ones_like(label))
            batch, x, z, edge_src, edge_dst, edge_attr, edge_length_embedded = data
            logits = model(batch, x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)*mask
            loss = self.loss_fn(logits, label)
            return loss, logits

        # Get gradient function
        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

        # Define function of one-step training
        def train_step(data, label):
            (loss, logits), grads = grad_fn(data, label)
            optimizer(grads)
            return loss, logits

        dataset_size = data_loader.get_dataset_size()
        start_time = time.time()
        for batch_idx, trainset in enumerate(data_loader.create_dict_iterator()):
            print(f"  Training -- Batch #{batch_idx} in {dataset_size} batches")
            data = trainset['data']
            target = data['y']
            batch, x, z, edge_src, edge_dst, edge_attr, edge_length_embedded = model.preprocess(data)
            load_time = time.time()-start_time
            start_time = time.time()
            
            loss, logits = train_step((batch, x, z, edge_src, edge_dst, edge_attr, edge_length_embedded), target)
            train_time = time.time()-start_time
            start_time = time.time()

            ms.ms_memory_recycle()

            for mul, l in self.Rs:
                if l == 0:
                    num_ele = ops.sum(logits[:, :mul]).asnumpy().item()

            train_num_ele.append(num_ele)
            mue_cum  += num_ele
            mae_cum  += np.abs(num_ele)
            loss_cum += abs(loss.asnumpy())
            post_time = time.time()-start_time
            start_time = time.time()
            print(f"    Batch #{batch_idx}: Load time: {load_time:.2f}, Train time: {train_time:.2f}, Post time: {post_time:.2f}")
            break

        train_tot = len(train_num_ele)
        train_stdev = np.std(train_num_ele)
        
        print("Epoch: ", epoch, f"{float(loss_cum)/len(data_loader):.10f}")
        print("      MAE", mae_cum/(len(data_loader)*self.b))
        print("      MUE", mue_cum/(len(data_loader)*self.b))
        print("      Train STDEV", train_stdev)
        print("      Train tot", train_tot)
        return loss_cum, mae_cum, mue_cum, train_num_ele


    def test_epoch(self, epoch, model, data_loader):
        model.set_train(False)
        test_loss_cum = 0.0
        test_mae_cum = 0.0
        test_mue_cum = 0.0
        bigIs_cum = 0.0
        eps_cum = 0.0
        ele_diff_cum = 0.0
        metrics = []
        test_num_ele = []

        dataset_size = data_loader.get_dataset_size()
        for batch_idx, testset in enumerate(data_loader.create_dict_iterator()):
            print(f"  Testing -- Batch #{batch_idx} in {dataset_size} batches")
            data = testset["data"]
            label = data['y']
            batch, x, z, edge_src, edge_dst, edge_attr, edge_length_embedded = model.preprocess(data)
            mask = ops.select(label == 0, ops.zeros_like(label), ops.ones_like(label))
            logits = model(batch, x, z, edge_src, edge_dst, edge_attr, edge_length_embedded) * mask
            err = self.loss_fn(logits, label).asnumpy().item()
            ms.ms_memory_recycle()

            for mul, l in self.Rs:
                if l == 0:
                    num_ele = ops.sum(logits[:, :mul]).asnumpy().item()

            test_num_ele.append(num_ele)
            test_mue_cum  += num_ele
            test_mae_cum  += abs(num_ele)
            test_loss_cum += abs(err)
            metrics.append([test_loss_cum, test_mae_cum, test_mue_cum])

            if epoch%10 == 0:
                num_ele_target, num_ele_ml, bigI, ep = get_scalar_density_comparisons(data, logits, self.Rs, spacing=0.2, buffer=4.0)
                n_ele = np.sum(data['z'].asnumpy())
                ele_diff_cum += np.abs(n_ele-num_ele_target)
                bigIs_cum += bigI
                eps_cum += ep
            
            break
 
        test_stdev = np.std(test_num_ele)
        print("      Test Loss", float(metrics[0][0])/len(data_loader))
        print("      Test MAE", metrics[0][1]/len(data_loader))
        print("      Test MUE", metrics[0][2]/len(data_loader))
        print("      Test STDEV", test_stdev)
        if epoch%10 == 0:
            print("      Test electron difference", ele_diff_cum/len(data_loader))
            print("      Test big I", bigIs_cum/len(data_loader))
            print("      Test epsilon", eps_cum/len(data_loader))

        if metrics[0][1]/len(data_loader) < self.minMAE:
            self.minMAE = metrics[0][1]/len(data_loader)
            save_checkpoint(model, os.path.join("./checkpoints/model_weights_best_MAE.ckpt"))
            print("Model saved with min MAE at epoch ", epoch)
        if metrics[0][2]/len(data_loader) < self.minMUE:
            self.minMUE = metrics[0][2]/len(data_loader)
            save_checkpoint(model, os.path.join("./checkpoints/model_weights_best_MUE.ckpt"))
            print("Model saved with min MUE at epoch ", epoch)
    
        return metrics, test_stdev, ele_diff_cum, bigIs_cum, eps_cum
    
def data_generator(data):
    for d in data:
        yield d

def count_parameters(model):
    return sum(p.size for p in model.trainable_params() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description='train electron density')
    parser.add_argument('--datapath', type=str)
    parser.add_argument('--testpath', type=str)
    parser.add_argument('--ckpt_savepath', type=str)
    parser.add_argument('--split', type=int)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--device', type=str, default="GPU")
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()

    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device, device_id=args.device_id)

    # first, get dataset
    hhh = "./data/h_s_only_augccpvdz_density.out"
    ooo = "./data/o_s_only_augccpvdz_density.out"
    ccc = "./data/c_s_only_augccpvdz_density.out"
    nnn = "./data/n_s_only_augccpvdz_density.out"
    ppp = "./data/p_s_only_augccpvdz_density.out"

    # second, check if num_ele is correct
    # def2 basis set max irreps
    Rs = [(14, 0), (5, 1), (5, 2), (2, 3), (1, 4)]

    checkpoint_dir = args.ckpt_savepath
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    b = args.batch
    print("Batch size: ", b)
    train_split = [args.split]
    num_epochs = args.epochs

    train_datafile = args.datapath
    test_datafile = args.testpath

    train_dataset = get_iso_permuted_dataset(train_datafile, h_iso=hhh, c_iso=ccc, n_iso=nnn, o_iso=ooo, p_iso=ppp)
    train_loader = ds.GeneratorDataset(lambda: data_generator(train_dataset), ["data"], shuffle=True)
    train_loader = train_loader.batch(b, per_batch_map=collate_list_of_dicts)

    test_dataset = get_iso_permuted_dataset(test_datafile, h_iso=hhh, c_iso=ccc, n_iso=nnn, o_iso=ooo, p_iso=ppp)
    test_loader = ds.GeneratorDataset(lambda: data_generator(test_dataset), ["data"], shuffle=True)
    test_loader = test_loader.batch(b, per_batch_map=collate_list_of_dicts)
    
    model_kwargs = {
        "irreps_in": "5x 0e", 
        "irreps_hidden": [(mul, (l, p)) for l, mul in enumerate([200,67,40,29]) for p in [-1, 1]], 
        "irreps_out": "14x0e + 5x1o + 5x2e + 2x3o + 1x4e", 
        "irreps_node_attr": None,
        "irreps_edge_attr": o3.Irreps.spherical_harmonics(3), 
        "layers": 3, #5,
        "max_radius": 3.5,
        "num_neighbors": 12.666666,
        "number_of_basis": 10,
        "radial_layers": 1,
        "radial_neurons": 128,
        "num_nodes": 24,
        "reduce_output": False,
    }

    model = Network(**model_kwargs)
    print("Number of parameters: ", count_parameters(model))

    # for batch_idx, testset in enumerate(test_loader.create_dict_iterator()):
    #     model.set_train(False)
    #     batch, x, z, edge_src, edge_dst, edge_attr, edge_length_embedded = model.preprocess(testset['data'])
    #     _ = model(batch, x, z, edge_src, edge_dst, edge_attr, edge_length_embedded).asnumpy()
    #     print("Testing forward pass", batch_idx)
    #     ms.ms_memory_recycle()

    # return 
    optim = nn.Adam(model.trainable_params(), learning_rate=1e-2)
    loss_fn = nn.MSELoss()
    trainer = ms_train(loss_fn, Rs, b)

    for epoch in range(num_epochs):
        _,_,_,_ = trainer.train_epoch(epoch, model, optim, train_loader)
        _,_,_,_,_ = trainer.test_epoch(epoch, model, test_loader)
        
    save_checkpoint(model, os.path.join("./checkpoints/trainall.ckpt"))
            

if __name__ == '__main__':
    main()