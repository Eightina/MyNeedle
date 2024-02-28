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
        # keepdims is important
        maxZ = array_api.max(Z, axis = self.axes, keepdims=True)
        res = array_api.log(
            array_api.sum(
                    array_api.exp(Z - maxZ),
                    axis=self.axes,
                    keepdims=True 
                )
            ) + maxZ
        if self.axes:
            res_shape = []
            for i, size in enumerate(Z.shape):
                if i not in set(self.axes):
                    res_shape.append(size)
            # print(res_shape)
            return res.reshape(tuple(res_shape))
        else:
            return res.reshape(-1) 
        # when axes is none, all elements are summed, so result's dimension is 1,
        # and can be automatically reshaped by appointing the shape param to be -1

    def gradient(self, out_grad, node):
        Z = node.inputs[0]
        if self.axes:
            shape = [1] * len(Z.shape)
            s = set(self.axes)
            j = 0
            for i in range(len(shape)):
                if i not in s:
                    shape[i] = node.shape[j]
                    j += 1
            node_new = node.reshape(shape)
            grad_new = out_grad.reshape(shape)
        else:
            node_new = node
            grad_new = out_grad
        return (grad_new * exp(Z - node_new),)


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

