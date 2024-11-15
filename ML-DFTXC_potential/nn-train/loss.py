import mindspore.nn as nn
import mindspore.ops as ops

class RMSELoss(nn.LossBase):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def construct(self, output, target):
        _pointwise_loss = lambda a, b: ((a - b) / b)**2
        d = _pointwise_loss(output, target)
        return self.get_loss(d)

class WMSELoss(nn.LossBase):
    def __init__(self):
        super(WMSELoss, self).__init__()
    
    def construct(self, output, target):
        _pointwise_loss = lambda a, b: ((a - b) ** 2) * ops.exp(b)
        d = _pointwise_loss(output, target)
        return self.get_loss(d)

class MSELoss_zsym(nn.LossBase):
    def __init__(self, coef=1.):
        super(MSELoss_zsym, self).__init__()
        self.coef = coef

    def construct(self, output, target):
        # output contains prediction from two channels.
        # true output = (xm + xp) / 2
        # LOSS = MSELoss(true output, target) + c * (xm - xp)**2
        _pointwise_loss = lambda a, b: ((a[0] + a[1]) / 2. - b)**2 + self.coef * ((a[0] - a[1])**2)
        d = _pointwise_loss(output, target)
        return self.get_loss(d)

