import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
import numpy as np
from mindspore import Tensor
from inp_params import batch_size_fp, num_gamma, cut_off_rad, widest_gaussian, narrowest_gaussian
import math


x = ms.Tensor([0] * 340, dtype=ms.float32)
#output = ops.pad(x, [1, 0, 0, 1], mode='constant', value=0.0)
#print(output)
#print(output.shape)





"""
a = np.ones((2,4,3))
b = np.ones((2,4,3))
c = np.ones((2,4,3))
# 假设 X_C 是一个包含多个数组的列表
X_C = [a,b,c]

# 使用 np.vstack 将 X_C 中的数组垂直堆叠
X_C_stacked = np.concatenate(X_C,axis=0)


print(X_C_stacked.shape)
# 输出：
# [[1 2]
#  [5 6]
#  [3 4]
#  [7 8]
#  [9 10],
#  [11 12]]








# 假设我们有一个二维数组，表示文本数据
array = np.array([[1.0, 2, 3], [4, 5, 6], [7, 8, 9]])

# 指定最大序列长度为 5
max_length = 5

# 使用 pad 函数进行填充
# 在最后一个轴上添加填充
padded_array = np.pad(array, pad_width=((0, 0), (0, max_length - array.shape[1])), mode='constant')

print(padded_array)




class Input_parameters:
    cut_off_rad = cut_off_rad
    batch_size_fp = batch_size_fp
    widest_gaussian = widest_gaussian
    narrowest_gaussian = narrowest_gaussian
    num_gamma = num_gamma


inp_args = Input_parameters()

inp_args.list_sigma = np.logspace(math.log10(inp_args.narrowest_gaussian), math.log10(inp_args.widest_gaussian),
                                  num_gamma)
inp_args.list_gamma = 0.5 / inp_args.list_sigma ** 2
inp_args.num_gamma = num_gamma


for nth_fp, gamma in enumerate(inp_args.list_gamma):
    norm = np.power(gamma / np.pi, 1.5)
    exp_term.append(norm * np.exp(-gamma * rad ** 2))
    tens1 = norm * ops.exp(-gamma * rad ** 2)
    tens2 = rad ** 2
    p = tf.math.divide_no_nan(tens1, rad)
    h = tf.math.divide_no_nan(tens1, tens2)
    exp_term_vec.append(p)
    exp_term_ten.append(h)
    exp_term[nth_fp] = exp_term[nth_fp] * cut_off_func_K
    exp_term_vec[nth_fp] = exp_term_vec[nth_fp] * cut_off_func_K
    exp_term_ten[nth_fp] = exp_term_ten[nth_fp] * cut_off_func_K
    rad_list.append(K.sum(exp_term[nth_fp], axis=0))






# case 1 : Reasonable application of broadcast mechanism
input = Tensor(np.arange(6*4*3).reshape(6, 4, 3), ms.float32)
other = Tensor(np.arange(6*3*1).reshape(6, 3, 1), ms.float32)
output = ops.matmul(input, other)
print(output.shape)






x = ms.Tensor(np.random.random([32, 10, 3]), ms.float32)
dense = ms.nn.Dense(3, 6)
net = ms.nn.TimeDistributed(dense,time_axis=1, reshape_with_axis=0)
output = net(x)
print(output.shape)


class a(nn.Cell):
    def __init__(self):
        super(a, self).__init__()
        self.dense1 = nn.Dense(360, 340,activation='tanh')
        self.dense2 = nn.Dense(200, 200,activation='tanh')
        self.dense3 = nn.Dense(200, 340)
    def construct(self,input1):
        model_out_allC = self.dense1(input1)
        model_out_expC, model_out_coefsC = ops.split(model_out_allC, [93, 247], axis=1)
        model_out_expC = ops.abs(model_out_expC)
        model_out_allC = ops.concat((model_out_expC, model_out_coefsC), axis=1)
        return model_out_allC

model_a = a()
data = [1, 0, 1, 0, 1]
x_data = ms.Tensor(data)
X = ops.ones((2,360), ms.float32)
print(type(X))
print(type(x_data))
#out = model_a(X)
#out = model_a(x_data)
#print(out)
"""