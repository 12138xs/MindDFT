# MindDFT

MindSpore implementation of integrated AI models for the DFT (Density Functional Theory) field.

### Code Integration

* **DeepDFT** 【[Source Code](https://github.com/peterbjorgensen/DeepDFT) | [Paper](https://www.nature.com/articles/s41524-022-00863-y)】
  * Model Introduction: This model is based on equivariant message passing on graphs and calculates density using special probe nodes inserted into the graph. Compared to the OrbNet-Equi model, which uses features calculated with the GFN-xTB semi-empirical electronic structure method, this method is purely data-driven, as the only inputs required for prediction are atomic numbers and atomic coordinates (including cell parameters for periodic structures).
  * **Network Architecture**: PaiNN network as the backbone.
  * **Training Module**: `train.py`
  * **Inference Modules**: `evaluate_model.py`, `predict_with_model.py`
  
* **ML-DFT** 【[Source Code](https://github.com/Ramprasad-Group/ML-DFT/tree/main) | [Paper](https://www.nature.com/articles/s41524-023-01115-3)】
  * Introduction: ML-DFT is a combination of various deep learning models that can predict various properties of the electronic structure of molecules and polymers at the DFT level: electron density, density of states (DOS), and total potential energy (with forces and stress tensors). The only required input is structural information in POSCAR format.
  * **Network Architecture**: Contains three sub-models: CHG, DOS, and Energy.
  * **Training Modules**: DOS_retrain, Energy_retrain
  * **Inference Modules**: CHG_predict, DOS_predict, Energy_predict

* **equivariant_electron_density** 【[Source Code](https://github.com/JoshRackers/equivariant_electron_density/tree/main) | [Paper](https://www.cell.com/biophysj/pdf/S0006-3495(22)00727-5.pdf)】
  * Predicting accurate ab initio DNA electron densities with equivariant neural networks.
  * A machine learning model based on an equivariant Euclidean neural network framework to obtain accurate ab initio electron densities for arbitrary DNA structures that are too large for traditional quantum methods.
  * `EEDM`
  
* **ML-DFTXC potential model** 【[Source Code](https://github.com/zhouyyc6782/oep-wy-xcnn/tree/master) | [Paper](https://pubs.acs.org/doi/10.1021/acs.jpclett.9b02838) | [Tutorial](https://www.sciencedirect.com/science/article/abs/pii/B978032390049200010X)】
  * Model Introduction: A deep learning method based on three-dimensional convolutional neural networks (3D-CNN) that determines the precise exchange-correlation (xc) potential of density functional theory (DFT) by mapping quasi-local electron density to the local exchange-correlation (xc) potential. The model is trained using high-precision electron density functions and corresponding xc potentials of small molecules and ions. It demonstrates transferability from small to large molecular systems, effectively captures van der Waals interactions, and outperforms the traditional B3LYP method in terms of accuracy and transferability.
