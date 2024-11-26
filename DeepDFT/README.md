# DeepDFT Model Implementation

This is the MindSpore implementation of the DeepDFT model for charge density prediction.

## Setup

Create and activate a virtual environment and install the requirements:

	$ pip install -r requirements.txt

## Data

Training data is expected to be a tar file containing `.cube` (Gaussian) or `.CHGCAR` (VASP) density files.
For best performance the tar files should not be compressed, but the individual files inside the tar
can use `zlib` compression (add `.zz` extension) or lz4 compression (add `.lz4` extension).
The data can be split up in several tar files. In that case create a text (.txt) file
in the same directory as the tar files. The text file must contain the file names of the tar files, one on each line.
Then the text file can then be used as a dataset.

* QM9 Charge Densities and Energies Calculated with VASP [[link]](https://data.dtu.dk/articles/dataset/QM9_Charge_Densities_and_Energies_Calculated_with_VASP/16794500)
* NMC Li-ion Battery Cathode Energies and Charge Densities [[link]](https://data.dtu.dk/articles/dataset/NMC_Li-ion_Battery_Cathode_Energies_and_Charge_Densities/16837721)
* Ethylene Carbonate Molecular  [[link]](https://data.dtu.dk/articles/dataset/Ethylene_Carbonate_Molecular_Dynamics/16691825)

## Running

### train
> python train.py --config config.yaml

### evaluate
> python evaluate_model.py --config config_eval.yaml

### predict
> python predict_with_model.py --config config_pred.yaml

## Reference

[Official Implementation](https://github.com/peterbjorgensen/DeepDFT/tree/main)