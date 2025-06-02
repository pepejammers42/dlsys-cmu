from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        max_z = array_api.max(Z, axis=1, keepdims=True)
        return Z - array_api.log(array_api.sum(array_api.exp(
            Z - array_api.max(Z, axis=1, keepdims=True)), axis=1, keepdims=True)) - max_z

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
        max_z = array_api.max(Z, self.axes)
        return array_api.log(array_api.sum(array_api.exp(Z - array_api.max(Z, axis=self.axes, keepdims=True)), axis=self.axes)) + max_z

    def gradient(self, out_grad, node):
        z = node.inputs[0]
        original_shape = z.shape

        if self.axes is None:
            bcast_shape = (1,) * len(original_shape)
        else:
            axes = (self.axes if isinstance(self.axes, tuple) else (self.axes,))
            bcast_shape = list(original_shape)
            for ax in axes:
                bcast_shape[ax] = 1

        lse_reshaped = node.reshape(bcast_shape)
        grad_reshaped = out_grad.reshape(bcast_shape)

        softmax = exp(z - lse_reshaped)
        return grad_reshaped * softmax



def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

