import mindspore.nn as nn
import mindspore.ops as ops

class CNN_GGA_1(nn.Cell):
    def __init__(self):
        super(CNN_GGA_1, self).__init__()
        self.rho_type = "GGA"
        self.conv1 = nn.Conv3d(4,  8, 4, has_bias=True, pad_mode='valid') # 4@9x9x9 ->  8@6x6x6, 4x4x4 kernel
        self.conv2 = nn.Conv3d(8, 16, 3, has_bias=True, pad_mode='valid') # 8@6x6x6 -> 16@4x4x4, 3x3x3 kernel
        self.fc1 = nn.Dense(128, 64)
        self.fc2 = nn.Dense(64, 32)
        self.fc3 = nn.Dense(32, 16)
        self.fc4 = nn.Dense(16, 1)

    def construct(self, x):
        # x shape: 4 x 9 x 9 x 9
        # for GGA-like NN, use electron density and its gradients

        x = ops.elu(self.conv1(x))
        x = ops.elu(self.conv2(x))
        x = ops.max_pool3d(x, 2)
        x = x.view(-1, self.num_flat_features(x))
        x = ops.elu(self.fc1(x))
        x = ops.elu(self.fc2(x))
        x = ops.elu(self.fc3(x))
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.shape[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CNN_GGA_1_zsym(nn.Cell):
    def __init__(self):
        super(CNN_GGA_1_zsym, self).__init__()
        self.rho_type = "GGA"

        self.conv1 = nn.Conv3d(4,  8, 4, has_bias=True, pad_mode='valid') # 4@9x9x9 ->  8@6x6x6, 4x4x4 kernel
        self.conv2 = nn.Conv3d(8, 16, 3, has_bias=True, pad_mode='valid') # 8@6x6x6 -> 16@4x4x4, 3x3x3 kernel
        self.fc1 = nn.Dense(128, 64)
        self.fc2 = nn.Dense(64, 32)
        self.fc3 = nn.Dense(32, 16)
        self.fc4 = nn.Dense(16, 1)

    def construct(self, x):
        # x shape: batch_size x 2 x 4 x 9 x 9 x 9

        xp = x[:, 0]
        xp = ops.elu(self.conv1(xp))
        xp = ops.elu(self.conv2(xp))
        xp = ops.max_pool3d(xp, 2)
        xp = xp.view(-1, self.num_flat_features(xp))
        xp = ops.elu(self.fc1(xp))
        xp = ops.elu(self.fc2(xp))
        xp = ops.elu(self.fc3(xp))
        xp = self.fc4(xp)

        xm = x[:, 1]
        xm = ops.elu(self.conv1(xm))
        xm = ops.elu(self.conv2(xm))
        xm = ops.max_pool3d(xm, 2)
        xm = xm.view(-1, self.num_flat_features(xm))
        xm = ops.elu(self.fc1(xm))
        xm = ops.elu(self.fc2(xm))
        xm = ops.elu(self.fc3(xm))
        xm = self.fc4(xm)

        return (xm, xp)

    def num_flat_features(self, x):
        size = x.shape[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

