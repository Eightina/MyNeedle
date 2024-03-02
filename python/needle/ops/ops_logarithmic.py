from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes


    def compute(self, Z):
        maxZ = Z.max(axis=self.axes, keepdims=True)  # max with og dim (shape changed)
        maxZ_reduce = Z.max(axis=self.axes)          # og val with reduced dim
        res = array_api.log(
            array_api.summation(
                    array_api.exp(Z - maxZ.broadcast_to(Z.shape)),  # expand shape back
                    axis=self.axes,
                ) # summation reduces dim
            ) + maxZ_reduce 
        return res

    def gradient(self, out_grad, node):
        Z = node.inputs[0] # max operation is const, its gradient is 0
        maxZ = Tensor(Z.realize_cached_data().max(axis=self.axes, keepdims=True), device=Z.device)
        expZ = exp(Z - maxZ.broadcast_to(Z.shape))
        grad_sum_expZ = 1 / summation(expZ, axes=self.axes)
        expand_shape = list(Z.shape)
        axes = range(len(expand_shape)) if self.axes is None else self.axes
        for axis in axes:
            expand_shape[axis] = 1
        grad_expZ = grad_sum_expZ.reshape(expand_shape).broadcast_to(Z.shape)
        return out_grad * expZ * grad_expZ 

def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

