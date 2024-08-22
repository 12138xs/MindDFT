# 关于utils
`dft_utils`文件夹下有几个.py文件，是对原训练过程中一些其他函数的分离或是一些辅助调试的工具。
- `checkpoint.py`是用于保存mindspore下模型检查点的工具。
- `dataset_utils.py`是一些用于对`dataset.py`中定义的数据类（对象）操作的方法。例如获取路劲下所有数据文件、分割数据集等。
- `parser.py`只是原程序中argparser相关的函数。
- `train_utils.py`包含了一些训练是的工具函数。注意，convert_batch_int32和flatten两个函数是历史遗留产物（我接手薛同学的工作时），后来我在重构代码后，int类型不匹配问题已经解决，不需要在train loop中再进行convert。而flatten属于其对dataloader数据结构不了解所致，也不必要。CustomWithLossCell尚有用，是我用于静态图模式时的一个自定义类型。
- `debuger.py`是一个我自己辅助调试的工具，可以快速检查变量等，您可以删除之。

# 关于动态图
当前版本是开发版，可能遗留了很多“脚手架”在代码中。  
比较稳定的版本见 https://github.com/12138xs/MindDFT/tree/pynative-mode-dev

## 遗留问题1
其他环节效率基本一致，但是eval部分明显慢。  
可能是引入了太多py原生操作导致的，在这方面ms可能确实做的没torch好。  
```
[Mindspore]
data_time=data_timer 1.965663 (1.965663), transfer_time=transfer_timer 0.000031 (0.000031), train_time=train_timer 2.186244 (2.186244), eval_time=eval_time 9.805295 (9.805295)
data_time=data_timer 0.003139 (0.315696), transfer_time=transfer_timer 0.000028 (0.000028), train_time=train_timer 0.787914 (1.114506), eval_time=eval_time 6.529138 (8.632532)

[Torch]
data_timer 0.206424 (0.206424) transfer_timer 0.000028 (0.000028) train_timer 1.103761 (1.103761) eval_time 1.938447 (1.938447)
data_timer 0.005508 (0.048105) transfer_timer 0.000103 (0.000100) train_timer 0.710822 (0.789908) eval_time 1.731535 (1.834991)
```

## 遗留问题2
[Mindspore]从step>=5开始sqrt_loss值和val都会到NaN
densitymodel里的问题，一定有地方出现非法运算了（除0或log一个非正数，或是某个操作不可导）。

## CheckPoint Feature
ms的序列化不支持自由的字典形式。checkpoint限制较多。
原torch实现可以将所有相关对象都作为字典序列化为一个文件进行保存，但minds pore首先必须单独序列化模型；其次并没有lr_scheduler对象。换言之，optimizer只接受规定学习率计划的函数，而并不维护本身的学习率状态。因此重新读取时只恢复optimizer,根据step数恢复即可。
checkpoint部分保存为一个文件夹，目录下有：
- `extra_info.json`记录保存时的step数以及best score
- `model.ckpt`保存模型
- `optimizer`保存优化器
如上所述，不需要再保存scheduler.


# 关于静态图
文件夹`gm`中是一些关于graph mode重写的代码，如density model; 相应的`dm`是pynative mode.
runner_graph是静态图的runner，而runner_pynative是动态图的runner，命名是根据mindspore的模式设置来的。

编译是通不过的，报错实在是各种各样。我修不好，希望您能解决。