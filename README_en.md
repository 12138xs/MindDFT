# MindDFT

MindSpore implementation of AI-integrated models for Density Functional Theory (DFT).

### Code Integration

This repository integrates several state-of-the-art machine learning models for DFT-related tasks. Below is an overview of the integrated models:

---

### **DeepDFT**  
**[Source Code](https://github.com/peterbjorgensen/DeepDFT) | [Paper](https://www.nature.com/articles/s41524-022-00863-y)**

- **Model Description**:  
  DeepDFT is based on equivariant message passing on graphs. The model employs special probe nodes inserted into the graph to compute density. Unlike the OrbNet-Equi model, which uses features derived from semi-empirical electronic structure methods like GFN-xTB, DeepDFT is purely data-driven. The only required inputs are atomic numbers and atomic coordinates (including cell parameters for periodic structures).
  
- **Branches**:  
  - **Training**: Available on the `master` branch (note: there is an issue with loss values becoming NaN).  
  - **Inference**: Available on the `evoluate-model` branch.

---

### **ML-DFT**  
**[Source Code](https://github.com/Ramprasad-Group/ML-DFT/tree/main) | [Paper](https://www.nature.com/articles/s41524-023-01115-3)**

- **Model Description**:  
  ML-DFT is a combination of deep learning models designed to predict various properties of molecular and polymeric electronic structures at the DFT level. These properties include electronic density, density of states (DOS), and total potential energy (including forces and stress tensors). The only required input is structural information in the POSCAR format.

- **Branches and Modules**:  
  - The `master` branch includes three sub-models: **CHG**, **DOS**, and **Energy**.  
  - **Training Modules**:  
    - `DOS_retrain`  
    - `Energy_retrain`  
  - **Inference Modules**:  
    - `CHG_predict`  
    - `DOS_predict`  
    - `Energy_predict`

---

### **Equivariant Electron Density**  
**[Source Code](https://github.com/JoshRackers/equivariant_electron_density/tree/main) | [Paper](https://www.cell.com/biophysj/pdf/S0006-3495(22)00727-5.pdf)**

- **Model Description**:  
  This model uses equivariant neural networks to predict accurate ab initio DNA electron densities. It is built on an equivariant Euclidean neural network framework to obtain accurate electron densities for arbitrary DNA structures that are too large for traditional quantum methods.

- **Branch**:  
  - `EEDM`

---

### **ML-DFTXC Potential Model**  
**[Source Code](https://github.com/zhouyyc6782/oep-wy-xcnn/tree/master) | [Paper](https://pubs.acs.org/doi/10.1021/acs.jpclett.9b02838) | [Tutorial](https://www.sciencedirect.com/science/article/abs/pii/B978032390049200010X)**

- **Model Description**:  
  This model is a deep learning approach based on 3D Convolutional Neural Networks (3D-CNN). It determines the accurate exchange-correlation (xc) potential of DFT by mapping quasi-local electronic densities to local xc potentials. Trained using high-precision electronic density functionals and their corresponding xc potentials for small molecules and ions, the model demonstrates transferability to larger molecular systems. It effectively captures van der Waals interactions and outperforms traditional methods like B3LYP in both accuracy and transferability.

---

### How to Use

1. **Environment Setup**:  
   Ensure that MindSpore is properly installed. Refer to the [official documentation](https://www.mindspore.cn/install) for installation instructions.

2. **Training and Inference**:  
   Each model has its own specific training and inference modules, as described above. Refer to the respective branches and modules for more details.

3. **Dependencies**:  
   Install the required dependencies for each model. Detailed dependency lists can be found in the respective repositories linked above.

### Contribution

Contributions are welcome! Please feel free to submit issues or pull requests to improve the repository.

### License

This project is licensed under the MIT License. Please refer to the LICENSE file for details.
