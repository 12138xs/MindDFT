import argparse
import os
import tarfile
import logging

import numpy as np
import mindspore as ms
from mindspore import ops

import dataset
import densitymodel
from runner import split_data
from utils import write_cube_to_tar

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Evaluate density model", fromfile_prefix_chars="@"
    )
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--num_interactions", type=int, default=3)
    parser.add_argument("--node_size", type=int, default=64)
    parser.add_argument("--split_file", type=str, default=None)
    parser.add_argument("--split", nargs='*', type=str)
    parser.add_argument("--probe_count", type=int, default=1000)
    parser.add_argument("--write_error_cubes", action="store_true")
    parser.add_argument(
        "--device_target",
        type=str,
        default="GPU",
        help="Specify the type of device to be used for inference e.g. 'Ascend', 'GPU', or 'CPU'",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="Specify the device number to be used for inference e.g. '0' or '1'",
    )
    parser.add_argument(
        "--ignore_pbc",
        action="store_true",
        help="If flag is given, ignore periodic boundary conditions in atoms data",
    )
    parser.add_argument(
        "--use_painn_model",
        action="store_true",
        help="Use painn model as backend",
    )

    return parser.parse_args(arg_list)

def batch_map_fn(data):
    batch_data = {}
    for key, value in data[0].items():
        if isinstance(value, ms.Tensor):
            batch_data[key] = value.asnumpy()
        else:
            batch_data[key] = value
    return batch_data

def main():
    args = get_arguments()

    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    handlers = [
        logging.FileHandler(
            os.path.join(args.output_dir, "printlog.txt"), mode="w"
        ),
        logging.StreamHandler(),
    ]

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=handlers,
    )

    device_target = args.device_target
    device_id = args.device_id
    ms.set_context(device_target=device_target, device_id=device_id)

    # Initialise model and load model
    if args.use_painn_model:
        net = densitymodel.PainnDensityModel(args.num_interactions, args.node_size, args.cutoff,)
    else:
        net = densitymodel.DensityModel(args.num_interactions, args.node_size, args.cutoff,)
    logging.info("loading model from %s", args.load_model)

    state_dict = ms.load_checkpoint(args.load_model)
    param_not_load, _ = ms.load_param_into_net(net, state_dict)

    # Load dataset
    if args.dataset.endswith(".txt"):
        # Text file contains list of datafiles
        with open(args.dataset, "r") as datasetfiles:
            filelist = [os.path.join(os.path.dirname(args.dataset), line.strip('\n')) for line in datasetfiles]
    else:
        filelist = [args.dataset]
    logging.info("loading data %s", args.dataset)
    # densitydata = ms.dataset.ConcatDataset([ms.dataset.GeneratorDataset(dataset.DensityData(path), column_names=["densitydata"]) for path in filelist])
    densitydata = ms.dataset.GeneratorDataset(dataset.DensityData(filelist[0]), column_names=["densitydata"], shuffle=False)
    # densitydata = dataset.DensityData(filelist[0])

    # Split data into train and validation sets
    if args.split_file:
        datasplits = split_data(densitydata, args)
    else:
        datasplits = {"all": densitydata}

    for split_name, densitydataset in datasplits.items():
        if args.split and split_name not in args.split:
            continue

        dataloader = densitydataset.batch(batch_size=1, num_parallel_workers=4)

        if args.write_error_cubes:
            outname = os.path.join(args.output_dir, "eval_" + split_name + ".tar")
            tar = tarfile.open(outname, "w")

        for density_dict in dataloader:
            density = []
            density_dict = batch_map_fn(density_dict)

            # Loop over all slices
            density_iter = dataset.DensityGridIterator(density_dict, args.probe_count, args.cutoff, args.ignore_pbc)

            # Make graph with no probes
            collate_fn = dataset.CollateFuncAtoms(
                cutoff=args.cutoff,
                set_pbc_to=args.ignore_pbc,
            )
            device_batch = collate_fn([density_dict])

            if args.use_painn_model:
                atom_representation_scalar, atom_representation_vector = net.atom_model(device_batch)
            else:
                atom_representation = net.atom_model(device_batch)

            num_positions = np.prod(density_dict["grid_position"].shape[0:3])
            sum_abs_error = ms.Tensor(0, dtype=ms.float32)
            sum_squared_error = ms.Tensor(0, dtype=ms.float32)
            sum_target = ms.Tensor(0, dtype=ms.float32)

            for slice_id, probe_graph_dict in enumerate(density_iter):
                # Transfer target to device
                flat_index = np.arange(slice_id*args.probe_count, min((slice_id+1)*args.probe_count, num_positions))
                pos_index = np.unravel_index(flat_index, density_dict["density"].shape[0:3])
                probe_target = ms.Tensor(density_dict["density"][pos_index], dtype=ms.float32)

                # Transfer model input to device
                probe_dict = dataset.collate_list_of_dicts([probe_graph_dict])
                device_batch["probe_edges"] = probe_dict["probe_edges"]
                device_batch["probe_edges_displacement"] = probe_dict["probe_edges_displacement"]
                device_batch["probe_xyz"] = probe_dict["probe_xyz"]
                device_batch["num_probe_edges"] = probe_dict["num_probe_edges"]
                device_batch["num_probes"] = probe_dict["num_probes"]

                if args.use_painn_model:
                    # res = net.probe_model(device_batch["probe_xyz"], atom_representation_scalar, atom_representation_vector, **device_batch)
                    res = net.probe_model.construct_and_gradients(device_batch, atom_representation_scalar, atom_representation_vector)
                else:
                    # res = net.probe_model(device_batch["probe_xyz"], atom_representation, **device_batch)
                    res = net.probe_model.construct_and_gradients(device_batch, atom_representation)

                # Compare result with target
                error = probe_target - res
                sum_abs_error += ops.sum(ops.abs(error))
                sum_squared_error += ops.sum(ops.square(error))
                sum_target += ops.sum(probe_target)

                if args.write_error_cubes:
                    density.append(res.detach().cpu().numpy())

            voxel_volume = density_dict["atoms"].get_volume()/np.prod(density_dict["density"].shape)
            rmse = ops.sqrt((sum_squared_error/num_positions))
            mae = sum_abs_error/num_positions
            abserror_integral = sum_abs_error*voxel_volume
            total_integral = sum_target*voxel_volume
            percentage_error = 100*abserror_integral/total_integral


            if args.write_error_cubes:
                pred_density = np.concatenate(density, axis=1)
                target_density = density_dict["density"]
                pred_density = pred_density.reshape(target_density.shape)
                errors = pred_density-target_density

                fname_stripped = density_dict["metadata"]["filename"]
                while fname_stripped.endswith(".zz"):
                    fname_stripped = fname_stripped[:-3]
                name, _ = os.path.splitext(fname_stripped)
                write_cube_to_tar(
                    tar,
                    density_dict["atoms"],
                    pred_density,
                    density_dict["grid_position"][0, 0, 0],
                    name + "_prediction" + ".cube" + ".zz",
                    )
                write_cube_to_tar(
                    tar,
                    density_dict["atoms"],
                    errors,
                    density_dict["grid_position"][0, 0, 0],
                    name + "_error" + ".cube" + ".zz",
                    )
                write_cube_to_tar(
                    tar,
                    density_dict["atoms"],
                    target_density,
                    density_dict["grid_position"][0, 0, 0],
                    name + "_target" + ".cube" + ".zz",
                    )

            logging.info("split=%s, filename=%s, mae=%f, rmse=%f, abs_relative_error=%f%%", split_name, density_dict["metadata"]["filename"], mae, rmse, percentage_error)


if __name__ == "__main__":
    main()
