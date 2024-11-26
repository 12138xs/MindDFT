# MindDFT

集成AI for DFT领域模型的MindSpore实现版本。

### 代码集成

* **DeepDFT** 【[源码](https://github.com/peterbjorgensen/DeepDFT) | [文章](https://www.nature.com/articles/s41524-022-00863-y)】
  * 模型简介：该模型基于图上的等变消息传递，并使用插入到图中的特殊探测节点，计算其密度。与使用GFN-xTB半经验电子结构方法计算的特征的OrbNet-Equi模型46相比，该方法纯粹是数据驱动的，因为预测所需的唯一输入是原子序数和原子的坐标（包括周期性结构的晶胞参数）
  * **训练**：`master`分支（存在loss出现nan问题）
  * **推理**：`evoluate-model`分支
  
* **ML-DFT** 【[源码](https://github.com/Ramprasad-Group/ML-DFT/tree/main) | [文章](https://www.nature.com/articles/s41524-023-01115-3)】
  * 简介：ML-DFT 是各种深度学习模型的组合，可在 DFT 级别预测分子和聚合物电子结构的各种属性：电子密度、态密度和总势能（带有力和应力张量）。唯一需要的输入是 POSCAR 格式的结构信息。
  * `master`分支，包含三个子模型：CHG, DOS, Energy
  * **训练模块**：DOS_retrain, Energy_retrain
  * **推理模块**：CHG_predict, DOS_predict, Energy_predict

* **equivariant_electron_density** 【[源码](https://github.com/JoshRackers/equivariant_electron_density/tree/main) | [文章](https://www.cell.com/biophysj/pdf/S0006-3495(22)00727-5.pdf)】
  * Predicting accurate ab initio DNA electron densities with equivariant neural networks
  * 一种基于等变欧几里得神经网络框架的机器学习模型，以获得对于传统量子方法来说太大的任意 DNA 结构的准确的从头算电子密度。
  * `EEDM`
  
* **ML-DFTXC potential model** 【[源码](https://github.com/zhouyyc6782/oep-wy-xcnn/tree/master) | [文章](https://pubs.acs.org/doi/10.1021/acs.jpclett.9b02838) | [教程](https://www.sciencedirect.com/science/article/abs/pii/B978032390049200010X)】
  * 模型简介：一种基于三维卷积神经网络（3D-CNN）的深度学习方法，该方法通过映射准局部电子密度到局部交换-相关（xc）势来确定电子密度泛函理论（DFT）的精确xc势。该模型利用小分子和离子的高精度电子密度函数及其对应的xc势进行训练，展示了其从小分子到大分子系统的可迁移性，能够有效地捕捉范德华相互作用，并在准确性和可转移性方面优于传统的B3LYP方法。
