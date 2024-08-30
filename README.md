# MindDFT

### 代码集成

* **DeepDFT** 【[源码](https://github.com/peterbjorgensen/DeepDFT) | [文章](https://www.nature.com/articles/s41524-022-00863-y)】
  * 模型简介：该模型基于图上的等变消息传递，并使用插入到图中的特殊探测节点，计算其密度。与使用GFN-xTB半经验电子结构方法计算的特征的OrbNet-Equi模型46相比，该方法纯粹是数据驱动的，因为预测所需的唯一输入是原子序数和原子的坐标（包括周期性结构的晶胞参数）
  * **训练**：`master`分支（存在loss出现nan问题）
  * **推理**：`evoluate-model`分支
  
* **ML-DFT** 【[源码](https://github.com/Ramprasad-Group/ML-DFT/tree/main) | [文章](https://www.nature.com/articles/s41524-023-01115-3)】
  * 简介：ML-DFT 是各种深度学习模型的组合，可在 DFT 级别预测分子和聚合物电子结构的各种属性：电子密度、态密度和总势能（带有力和应力张量）。唯一需要的输入是 POSCAR 格式的结构信息。
  * `master`

* **equivariant_electron_density** 【[源码](https://github.com/JoshRackers/equivariant_electron_density/tree/main) | [文章](https://www.cell.com/biophysj/pdf/S0006-3495(22)00727-5.pdf)】
  * Predicting accurate ab initio DNA electron densities with equivariant neural networks
  * 一种基于等变欧几里得神经网络框架的机器学习模型，以获得对于传统量子方法来说太大的任意 DNA 结构的准确的从头算电子密度。
  * `EEDM`


### TODO List

* 合并DeepDFT代码
* 整理ML-DFT代码
* 实现e3nn中gate_points_2101模型