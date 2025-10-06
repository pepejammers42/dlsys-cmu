"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a ** b
        
    def gradient(self, out_grad, node):
        a, b = node.inputs
        return out_grad * b * a ** (b-1), out_grad * (a**b) * log(a)

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad, node):
        return out_grad *  self.scalar * (node.inputs[0] **  (self.scalar - 1))


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        return out_grad / b, - a * out_grad / b ** 2


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        dims = list(range(len(a.shape)))
        if not self.axes:
            x1, x2, = len(dims) - 2, len(dims) - 1
        elif len(self.axes) == 2:
            x1 = self.axes[0]
            x2 = self.axes[1]
        dims[x1], dims[x2] = dims[x2], dims[x1]
        return a.permute(tuple(dims))

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return a.compact().reshape(self.shape)

    def gradient(self, out_grad, node):
        return reshape(out_grad, node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        out_shape = node.shape
        ori_shape = node.inputs[0].shape
        shrink = [i for i, (o, t) in enumerate(zip(ori_shape, out_shape)) if o == 1 and t != 1]
        shrink = tuple(sorted(shrink, reverse=True))
        return out_grad.sum(shrink).reshape(ori_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if isinstance(self.axes, tuple):
            axes = self.axes
        elif isinstance(self.axes, int):
            axes = self.axes,
        elif self.axes is None:
            axes = list(range(len(a.shape)))
        else:
            raise TypeError
        axes = sorted((ax % a.ndim for ax in axes), reverse=True)
        out = a
        for ax in axes:
            out = out.sum(ax)
        return out

    def gradient(self, out_grad, node):
        in_shape = node.inputs[0].shape
        if self.axes is None:
            axes = tuple(range(len(in_shape)))
        elif isinstance(self.axes, int):
            axes = (self.axes % len(in_shape),)
        else:
            axes = tuple(ax % len(in_shape) for ax in self.axes)
        expand = list(in_shape)
        for ax in axes:
            expand[ax] = 1
        return out_grad.reshape(tuple(expand)).broadcast_to(in_shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        first_grad = out_grad.matmul(rhs.transpose())
        second_grad = lhs.transpose().matmul(out_grad)
        first_grad_len = len(lhs.shape)
        second_grad_len = len(rhs.shape)
        if first_grad_len > second_grad_len:
            second_grad = second_grad.sum(tuple(range(first_grad_len - second_grad_len)))
        if second_grad_len > first_grad_len:
            first_grad = first_grad.sum(tuple(range(second_grad_len - first_grad_len)))

        return first_grad, second_grad

def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return - a

    def gradient(self, out_grad, node):
        return - out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        return out_grad * Tensor(node.inputs[0].realize_cached_data() > 0, device=out_grad.device)


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        return a.tanh()

    def gradient(self, out_grad, node):
        return out_grad * (1.0 - tanh(node.inputs[0]) ** 2)

def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ndims = len(args) 
        new_shape = list(args[0].shape) 
        new_shape.insert(self.axis, ndims)
        new_tensor = array_api.empty(new_shape, device=args[0].device)
        for i, arr in enumerate(args):
            s = tuple(i if self.axis == idx else slice(None) for idx, dim in enumerate(new_shape))
            new_tensor[s] = arr
        return new_tensor

    def gradient(self, out_grad, node):
        return split(out_grad, self.axis)


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        n = A.shape[self.axis]
        new_shape = list(A.shape)
        new_shape.pop(self.axis)
        s = []
        for i in range(n):
            tup = tuple(i if self.axis == idx else slice(None) for idx in range(len(A.shape)))
            s.append(A[tup].compact().reshape(new_shape))
        return tuple(s)

    def gradient(self, out_grad, node):
        return stack(out_grad, self.axis)


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return a.flip(self.axes)

    def gradient(self, out_grad, node):
        return flip(out_grad, self.axes)


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ndims = tuple(d * (self.dilation + 1) if i in self.axes else d for i, d in enumerate(a.shape))
        s = tuple(slice(None, None, self.dilation + 1) if i in self.axes else slice(None) for i in range(len(a.shape)))
        new_tensor = array_api.full(ndims, 0, device=a.device)
        new_tensor[s] = a
        return new_tensor

    def gradient(self, out_grad, node):
        return undilate(out_grad, self.axes, self.dilation)


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        # in a sense u just have to select what u just dilated 5head deluxe
        return a[tuple(slice(None, None, self.dilation + 1) if i in self.axes else slice(None) for i in range(len(a.shape)))]

    def gradient(self, out_grad, node):
        return dilate(out_grad, self.axes, self.dilation)


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        # fries in the bag look ipynb provided lowkey don't fully get it but i also get it
        A = A.pad(((0,0), (self.padding, self.padding), (self.padding, self.padding),(0,0)))
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides

        inner_dim = K * K * C_in
        Hp = (H-K+1) // self.stride
        Wp = (W-K+1) // self.stride
        A_as =  A.as_strided(shape=(N, Hp, Wp, K, K, C_in), strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs)).compact().reshape((N*Hp*Wp, inner_dim))
        conv = A_as @ B.compact().reshape((K*K*C_in, C_out))
        return conv.compact().reshape((N, Hp, Wp, C_out))

    def gradient(self, out_grad, node):
        A, B = node.inputs
        K = B.shape[0]
        
        # dilate to account for the strides
        if self.stride > 1:
            out_grad = dilate(out_grad, (1,2), self.stride - 1)

        # Note out_grad is N, Hp, Wp, C_out
        # dL/dA ~= conv(out_grad, B)
        # B originally is K, K, C_in, C_out
        #              => K, K, C_out, C_in
        Bp = transpose(flip(B, (0,1)), (2,3))
        grad_A = conv(out_grad, Bp, 1, K - self.padding - 1)
        
        # dL/dB ~= conv(A, out_grad)
        # A is N, H, W, C_in
        #   => C_in, H, W, N
        Ap = transpose(A, (0, 3))
        # out_grad is now H, W, N, C_out
        out_gradp = transpose(transpose(out_grad, (0, 1)), (1, 2))
        grad_B = conv(Ap, out_gradp, 1, self.padding)
        grad_B = transpose(transpose(grad_B, (0, 1)), (1, 2))

        return grad_A, grad_B


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


