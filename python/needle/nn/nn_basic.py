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

        self.biased = bias
        self.device = device
        
        self.weight = Parameter(init.kaiming_uniform(
            fan_in=in_features,
            fan_out=out_features,
            device=device,
            dtype=dtype
        ))

        if not self.biased:
            self.bias = None
            return
        self.bias = Parameter(init.kaiming_uniform(
                fan_in=out_features,
                fan_out=1,
                device=device,
                dtype=dtype
            ).reshape(
                (1, out_features)
            ))

    def forward(self, X: Tensor) -> Tensor:
        if not isinstance(self.bias, Tensor):
            return (
                X @ self.weight
            )
            
        return (
            X @ self.weight
            + self.bias.broadcast_to(
                (X.shape[0], self.out_features)
            )
        )


class Flatten(Module):
    def forward(self, X):
        if len(X.shape) <= 1:
            return X
        
        shape_res_mul = 1
        for i in range(1, len(X.shape)):
            shape_res_mul *= X.shape[i]
        return ops.reshape(X, (X.shape[0], shape_res_mul))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module.forward(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        lse = ops.logsumexp(logits, axes=(1,))
        zy = ops.summation(logits * init.one_hot(logits.shape[-1], y, requires_grad=True), axes=(1,)) 
        return ops.summation(lse - zy) / logits.shape[0]


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(dim, requires_grad=True, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, requires_grad=True, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, requires_grad=False, device=device, dtype=dtype)
        self.running_var = init.ones(dim, requires_grad=False, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        batches = x.shape[0]
        broadcasted_b = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        broadcasted_w = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        if self.training:
            E = x.sum(axes=(0,)) / batches
            broadcasted_E = E.reshape((1, self.dim)).broadcast_to(x.shape)
            Var = ((x - broadcasted_E) ** 2).sum(axes=(0,)) / batches
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * E.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * Var.data
            
            x_std = (Var.reshape((1, self.dim)).broadcast_to(x.shape) + self.eps) ** 0.5
            x_normed = (x - broadcasted_E) / x_std
            
        else:
            x_normed = (x - self.running_mean.reshape((1, self.dim)).broadcast_to(x.shape)) / \
                            ((self.running_var + self.eps) ** 0.5).reshape((1, self.dim)).broadcast_to(x.shape)
                            
        return broadcasted_w * x_normed + broadcasted_b

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones((dim), requires_grad=True, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros((dim), requires_grad=True, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        # It seems that E and Var still requires grad?? Why? 
        # Because they are calculated from x and interferes computation of y.
        batches = x.shape[0]
        broadcasted_E = (x.sum(axes=(1,)).reshape((batches, 1)) / self.dim).broadcast_to(x.shape)
        broadcasted_w = self.weight.broadcast_to(x.shape)
        broadcasted_b = self.bias.broadcast_to(x.shape)
        Var = ((x - broadcasted_E) ** 2).sum(axes=(1,)).reshape((batches, 1)) / self.dim
        y = broadcasted_w * ((x - broadcasted_E) / ((Var.broadcast_to(x.shape) + self.eps) ** 0.5)) + broadcasted_b
        return y

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        z = init.randb(*(x.shape), p = 1 - self.p)
        z /= 1 - self.p
        return x * z


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn.forward(x) + x