"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for param in self.params:
            if self.weight_decay > 0:
                # so this becomes grad of f(theta_t) + weight_decay * theta_t
                grad_update = param.grad.data + self.weight_decay * param.data
            else:
                grad_update = param.grad.data
            # u_t doesn't exist initially is just 0
            if param in self.u:
                self.u[param] = self.momentum * self.u[param] + (1-self.momentum) * grad_update
            else:
                self.u[param] = (1-self.momentum) * grad_update
            param.data = param.data - self.lr * self.u[param]
        

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.u = {}
        self.v = {}

    def step(self):
        self.t += 1
        for param in self.params:
            if self.weight_decay > 0:
                grad_new = param.grad.data + self.weight_decay * param.data
            else:
                grad_new = param.grad.data
            if param in self.u:
                self.u[param] = self.beta1 * self.u[param] + (1-self.beta1) * grad_new
            else:
                self.u[param] = (1-self.beta1) * grad_new
            if param in self.v:
                self.v[param] = self.beta2 * self.v[param] + (1-self.beta2) * (grad_new ** 2)
            else:
                self.v[param] = (1-self.beta2) * (grad_new ** 2)
            corrected_u = self.u[param] / (1- self.beta1 ** self.t)
            corrected_v = self.v[param] / (1-self.beta2 ** self.t)
            param.data = param.data - self.lr * corrected_u / (corrected_v ** 0.5 + self.eps)