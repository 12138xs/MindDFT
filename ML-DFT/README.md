# ML-DFT (MindSpore Implementation)

This repository contains the MindSpore implementation of the **ML-DFT** model, originally developed by the [Ramprasad Group](https://github.com/Ramprasad-Group/ML-DFT/tree/main). ML-DFT is a collection of deep learning models designed to predict electronic structure properties at the Density Functional Theory (DFT) level.

---

## Features

- **Supported Predictions**:
  - **Electronic Density** (`CHG`)
  - **Density of States** (`DOS`)
  - **Total Potential Energy** (including forces and stress tensors)
  
- **Input Format**: Structural information in **POSCAR** format.

- **MindSpore Integration**:  
  This implementation leverages the MindSpore framework for efficient training and inference on CPU, GPU and Ascend platforms.

---

## Repository Structure

The repository includes the following modules:

- **Training Modules**:
  - `DOS_retrain`: Retraining for Density of States prediction.
  - `Energy_retrain`: Retraining for Total Energy prediction.

- **Inference Modules**:
  - `CHG_predict`: Predicts electronic density.
  - `DOS_predict`: Predicts density of states.
  - `Energy_predict`: Predicts total potential energy.

---

## Getting Started

### 1. Environment Setup

Ensure you have the following dependencies installed:
- **MindSpore**: Follow the [official installation guide](https://www.mindspore.cn/install).
- Additional dependencies: Install other required Python libraries using:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Training and Inference

For training or inference, use the corresponding scripts. For example:
```bash
vim inp_params.py   # set the configs
python ML-DFT.py
```

---

## References

- Original ML-DFT Repository: [GitHub Link](https://github.com/Ramprasad-Group/ML-DFT/tree/main)
- Paper: [Nature Article](https://www.nature.com/articles/s41524-023-01115-3)

---

## Acknowledgments

This implementation is based on the original work by the Ramprasad Group and has been adapted for the MindSpore framework.
