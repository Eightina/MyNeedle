import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    high = gain * ((6 / (fan_in + fan_out)) ** (1 / 2))
    kwargs['low'] = -high
    kwargs['high'] = high
    return rand(fan_in, fan_out, **kwargs)


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    std = gain * ((2 / (fan_in + fan_out)) ** (1/2))
    kwargs['std'] = std
    return randn(fan_in, fan_out, **kwargs)



def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = 2 ** (1 / 2)
    bound = gain * ((3 / fan_in) ** (1 / 2))
    kwargs['high'] = bound
    kwargs['low'] = -bound
    return rand(fan_in, fan_out, **kwargs)

def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = 2 ** (1 / 2)
    std = gain * ((fan_in) ** (-1 / 2))
    kwargs['std'] = std
    return randn(fan_in, fan_out, **kwargs) 