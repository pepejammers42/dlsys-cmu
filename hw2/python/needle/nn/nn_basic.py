"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(init.kaiming_uniform(in_features, out_features))
        if bias:
            # gives back (in * out) + bias but that is (1 * out) so need to transpose this.
            self.bias = Parameter(init.kaiming_uniform(out_features, 1).transpose())
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        res = X.matmul(self.weight)
        if self.bias:
            res += self.bias.broadcast_to((res.shape))
        return res


class Flatten(Module):
    def forward(self, X):
        N = X.shape[0]
        rest = 1
        for dim in X.shape[1:]:
            rest *= dim
        return X.reshape((N, rest))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        res = x
        for m in self.modules:
            res = m(res)
        return res


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        N, C = logits.shape
        y_oh = init.one_hot(C, y)
        
        lse = ops.logsumexp(logits, axes=(1,))
        z_y = ops.summation((y_oh * logits), axes=(1,))
        
        return ops.summation((lse-z_y))/ N


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(dim, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, requires_grad=True))
        self.running_mean = init.zeros(dim)
        self.running_var = init.ones(dim)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            # NOTE: we have here (dim, ) so need to get back to (B, dim) after
            # also, each of the moving averages also have (dim, )
            expectation = x.sum(axes=(0,)) / x.shape[0]
            var = ((x - expectation.reshape((1, self.dim)).broadcast_to(x.shape)) ** 2).sum(axes=(0,)) / x.shape[0]
            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * expectation.data
            self.running_var = (1-self.momentum) * self.running_var + self.momentum * var.data
            # the reshape back
            norm = (x-(expectation.reshape((1, x.shape[1])).broadcast_to(x.shape))) / (((var.reshape((1, x.shape[1])).broadcast_to(x.shape)) + self.eps)**0.5)
        else:
            norm = (x-(self.running_mean.reshape((1, x.shape[1])).broadcast_to(x.shape))) / (((self.running_var.reshape((1, x.shape[1])).broadcast_to(x.shape)) + self.eps)**0.5)
        return self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, requires_grad=True))

    def forward(self, x: Tensor) -> Tensor:
        # NOTE: this is not numpy, so you need to broadcast it yourself, cannot do it implicitly. 
        expectation = (x.sum(axes=(1,)) / x.shape[1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        var = (((x-expectation) ** 2).sum(axes=(1,))/x.shape[1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        norm = (x-expectation) / ((var + self.eps)**0.5)
        return self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            # binary mask
            dist = init.randb(*x.shape, p=1-self.p)
            return x * dist / (1-self.p)
        else:
            return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)
