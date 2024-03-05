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
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad, node):
        (ipt,) = node.inputs
        return (out_grad * self.scalar * ipt ** (self.scalar - 1),)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        l, r = node.inputs
        return (out_grad / r, -out_grad * l / r ** (2))


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return (out_grad / self.scalar,)


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes:
            ax0, ax1 = self.axes[0], self.axes[1]
        else:
            ax0, ax1 = a.ndim - 2, a.ndim - 1
        permute_axes = list(range(a.ndim))
        permute_axes[ax0], permute_axes[ax1] = ax1, ax0
        return a.permute(permute_axes)

    def gradient(self, out_grad, node):
        return out_grad.transpose(self.axes)

def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        return (out_grad.reshape(node.inputs[0].shape),)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ipt_shape = list(node.inputs[0].shape)
        axes = []
        ipt_shape = [1] * (len(self.shape) - len(ipt_shape)) + ipt_shape
        for i, s in enumerate(self.shape):
            if i >= len(ipt_shape) or s != ipt_shape[i]:
                axes.append(i)
        return (reshape(summation(out_grad, tuple(axes)), ipt_shape),)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if isinstance(self.axes, (list, tuple)) and len(self.axes) > 1:
            # multiple axes case
            for axis in reversed(sorted(self.axes)):
                a = a.sum(axis = axis)
            return a
        return a.sum(axis=self.axes)

    def gradient(self, out_grad, node):
        new_shape = list(node.inputs[0].shape)
        if self.axes is None:
            axes = range(len(new_shape))
        elif isinstance(self.axes, tuple):
            axes = self.axes
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            raise ValueError("Unsupported axes type, must be int, tuple or None!")
        for axis in axes:
            new_shape[axis] = 1
        return out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)

def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        l: Tensor
        r: Tensor
        l_grad: Tensor
        r_grad: Tensor

        (l, r) = node.inputs
        l_grad = out_grad @ r.transpose()
        r_grad = l.transpose() @ out_grad

        if l_grad.shape != l.shape:
            l_grad = l_grad.sum(
                tuple(range(l_grad.ndim - l.ndim))
            )  # doing summation on the dims representing batches

        if r_grad.shape != r.shape:
            r_grad = r_grad.sum(tuple(range(r_grad.ndim - r.ndim)))

        return (l_grad, r_grad)
        # caution! eighter of l or r can be batched like 6 * 6 * 5 * 4, 4 * 3


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return array_api.negative(a)

    def gradient(self, out_grad, node):
        return (-out_grad,)



def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        (ipt,) = node.inputs
        return (out_grad / ipt,)


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return (exp(node.inputs[0]) * out_grad,)


def exp(a):
    return Exp()(a)


class Binary(TensorOp):
    def compute(self, a: NDArray):
        return a.astype(array_api.bool8).astype(array_api.float32)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (
            Tensor(
                [
                    [0.0 for i in range(node.inputs[0].shape[1])]
                    for j in range(node.inputs[0].shape[0])
                ]
            ),
        )

def binary(a):
    return Binary()(a)

class ReLU(TensorOp):
    def compute(self, a):
        return a.maximum(0)

    def gradient(self, out_grad, node):
        a = node.inputs[0].realize_cached_data()
        mask = Tensor(a > 0, requires_grad=False)
        return (out_grad * mask,)


def relu(a):
    return ReLU()(a)

class Tanh(TensorOp):
    def compute(self, a):
        # return array_api.tanh(a)
        return a.tanh()

    def gradient(self, out_grad, node):
        # return (out_grad * (1 - array_api.tanh(node.inputs[0])**2),)
        return out_grad * (1 - tanh(node.inputs[0])**2)


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
        if (not args): raise IndexError("No arrays to stack");
        std_shape = args[0].shape
        for ndarray in args:
            assert ndarray.shape == std_shape
        
        new_shape = list(std_shape)
        new_shape.insert(self.axis, len(args))
        
        res = array_api.empty(shape=tuple(new_shape), dtype=args[0].dtype, device=args[0].device)
        slices = [slice(0, n) for n in new_shape]
        for (i, ndarray) in enumerate(args):
            slices[self.axis] = slice(i, i + 1)
            res[tuple(slices)] = ndarray
        return res
        


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
        split_n = A.shape[self.axis]
        if (split_n < 1): raise ValueError("Ilegal dim for split");
        if (len(A.shape) < 2): raise ValueError("Ilegal shape for split")

        new_shape = list(A.shape)
        del new_shape[self.axis]
        
        slices = [slice(0, n) for n in A.shape]
        res = []
        for i in range(split_n):
            slices[self.axis] = slice(i, i + 1)
            res.append(A[tuple(slices)].compact().reshape(new_shape))
        return tuple(res)
        

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
        new_shape = list(a.shape)
        for ax in self.axes:
            new_shape[ax] *= (self.dilation + 1)
        res = array_api.full(tuple(new_shape), 0.0, dtype=a.dtype, device=a.device)
        slices = [slice(0, l) for l in new_shape]  
        for ax in self.axes:
            slices[ax] = slice(0, slices[ax].stop, self.dilation + 1)
        res[tuple(slices)] = a
        return res

    def gradient(self, out_grad, node):
        return undilate(out_grad, self.axes, self.dilation)


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        slices = [slice(0, l) for l in a.shape]
        for ax in self.axes:
            slices[ax] = slice(0, slices[ax].stop, self.dilation + 1)
        res = a[tuple(slices)].compact()
        return res

    def gradient(self, out_grad, node):
        return dilate(out_grad, self.axes, self.dilation)


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        A = A.pad(((0,0), (self.padding, self.padding), (self.padding, self.padding),(0,0)))
        N, H, W, C_in = A.shape            # A: (N, H+2p, W+2p, Cin)
        K, K_, C_in_, C_out = B.shape      # B: (K, K, Cin, Cout)
        Ns, Hs, Ws, Cs = A.strides
        assert K == K_, "Conv kernel should be a square tensor"
        assert C_in == C_in_, "Conv kernel and input are not compatible"
        
        inner_dim = K * K_ * C_in          # dim for matmul
        out_H, out_W = (H - K + 1) // self.stride, (W - K + 1) // self.stride 
        im2col =   A.as_strided(  
                                shape=(N, out_H, out_W, K, K, C_in),
                                strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs)
                                )\
                    .compact()\
                    .reshape((N * out_H * out_W, inner_dim))
    
        out = im2col @ B.compact().reshape((K * K_ * C_in_, C_out)) # (N * out_H * out_W, inner_dim) @ (inner_dim, C_out)
        
        return out.compact().reshape((N, out_H, out_W, C_out))      # out: (N, (H+2p-K+1)/s, (W+2p-K+1)/s, Cout)

    def gradient(self, out_grad, node):
        X, W = node.inputs
        K, _, _, _ = W.shape

        if self.stride > 1:
            out_grad = dilate(out_grad, (1, 2), self.stride - 1)        # out_grad      : (N, H+2p-K+1, W+2p-K+1, Cout) from above forward result
        W_permute = transpose(flip(W, (0, 1)), (2, 3))                  # W_permute     : (K, K, Cout, Cin) as kernel
        X_grad = conv(out_grad, W_permute, padding=K-1-self.padding)    # X_grad        : (N, (H+2p-K+1)+2(K-1-p)-K+1, (W+2p-K+1)+2(K-1-p)-K+1, Cin)
                                                                        # ~             : (N, H, W, Cin)

        X_permute = transpose(X, (0, 3))                                # X_permute     : (Cin, H, W, N) 
        grad_permute = transpose(transpose(out_grad, (0, 1)), (1, 2))   # grad_permute  : (H+2p-K+1, W+2p-K+1, N, Cout) as kernel
        W_grad = conv(X_permute, grad_permute, padding=self.padding)    # W_grad        : (Cin, H+2p-(H+2p-K+1)+1, W+2p-(W+2p-K+1)+1, Cout)
        W_grad = transpose(transpose(W_grad, (0, 1)), (1, 2))           # W_grad        : (Cin, K, K, Cout)
        return X_grad, W_grad

def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
