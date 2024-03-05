"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.biased = bias
        self.device = device
        self.dtype = dtype
        
        self.weight = Parameter(init.kaiming_uniform(
            fan_in=(kernel_size ** 2) * in_channels,
            fan_out=(kernel_size ** 2) * out_channels,
            shape=(kernel_size, kernel_size, in_channels, out_channels),
            device=device,
            dtype=dtype
        ))
        
        if not self.biased:
            self.bias = None
        else:
            bias_bound = 1.0 / (in_channels * kernel_size ** 2) ** 0.5
            self.bias = Parameter(init.rand(
                self.out_channels,
                low=-bias_bound,
                high=bias_bound,
                device=device,
                dtype=dtype
            ))
        
        self.padding = (kernel_size - 1) // 2
        
        return

    def forward(self, x: Tensor) -> Tensor:
        # x is originally NCHW
        # we need NHWC
        x = ops.ops_mathematic.transpose(
            Tensor.transpose(x, (1, 2)),
            (2, 3)
        ) # this is NHWC
        
        conved = ops.ops_mathematic.conv(
            a = x,
            b = self.weight,
            stride = self.stride,
            padding = self.padding
        ) # this is NHWC
        
        if self.biased:
            reshaped_bias = self.bias.reshape((1, 1, 1, self.out_channels))
            conved += ops.ops_mathematic.broadcast_to(
                reshaped_bias, conved.shape
            ) # this is NHWC
            
        conved = ops.ops_mathematic.transpose(
            Tensor.transpose(conved, (2, 3)),
            (1, 2)
        ) # return NCHW
        
        return conved
        