from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        max_z_keep = Z.max(axis=1, keepdims=True)
        max_z = Z.max(axis=1)
        return Z - array_api.log(array_api.sum(array_api.exp(Z - max_z_keep), axis=1, keepdims=True)) - max_z

    def gradient(self, out_grad, node):
        z = node.inputs[0]
        lse = logsumexp(z, axes=(1,))
        lse_reshaped = lse.reshape((out_grad.shape[0], 1))
        softmax = exp(z - lse_reshaped)
        
        sum_grad = summation(out_grad, axes=(1,))
        sum_grad_reshaped = sum_grad.reshape((out_grad.shape[0], 1))
        
        return out_grad - sum_grad_reshaped * softmax


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        max_z = Z.max(axis=self.axes, keepdims=True)
        max_z_ = max_z.max(axis=self.axes)
        return array_api.log(
            array_api.sum(array_api.exp(Z - array_api.broadcast_to(max_z, Z.shape)), axis=self.axes)) + max_z_

    def gradient(self, out_grad, node):
        z = node.inputs[0]
        max_z = Tensor(z.realize_cached_data().max(axis=self.axes, keepdims=True), device=z.device)
        exp_z = exp(z - max_z.broadcast_to(z.shape))
        sum_exp_z = summation(exp_z, axes=self.axes)
        grad_sum_exp_z = out_grad / sum_exp_z
        expand_shape = list(z.shape)
        if self.axes is None:
            axes = range(len(expand_shape))
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            axes = self.axes
        for axis in axes:
            expand_shape[axis] = 1
        grad_exp_z = grad_sum_exp_z.reshape(expand_shape).broadcast_to(z.shape)
        return grad_exp_z * exp_z

def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

