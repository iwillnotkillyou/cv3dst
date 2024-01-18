from numbers import Number
from typing import Callable

import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def nd2col(
    input_nd,
    kernel_size,
    stride=1,
    padding=0,
    output_padding=0,
    dilation=1,
    # transposed=False,
):
    """
    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, *kernel_size, *L_{out})` where
          :math:`L_{out} = floor((L_{in} + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)` for non-transposed
          :math:`L_{out} = (L_{in} - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding` for transposed
    """
    n_dims = len(input_nd.shape[2:])
    kernel_size = (
        (kernel_size,) * n_dims if isinstance(kernel_size, Number) else kernel_size
    )
    stride = (stride,) * n_dims if isinstance(stride, Number) else stride
    padding = (padding,) * n_dims if isinstance(padding, Number) else padding
    output_padding = (
        (output_padding,) * n_dims
        if isinstance(output_padding, Number)
        else output_padding
    )
    dilation = (dilation,) * n_dims if isinstance(dilation, Number) else dilation

    (bs, nch), in_sz = input_nd.shape[:2], input_nd.shape[2:]
    # NOTE: make a possible task to implement the correct output size of the convolution operation
    out_sz = tuple(
        [
            ((i + 2 * p - d * (k - 1) - 1) // s + 1)
            for (i, k, d, p, s) in zip(in_sz, kernel_size, dilation, padding, stride)
        ]
    )

    output = F.unfold(input_nd, kernel_size, dilation, padding, stride)
    out_shape = (bs, nch) + tuple(kernel_size) + out_sz
    output = output.view(*out_shape).contiguous()
    return output


def packernel2d(
    input: torch.Tensor,
    kernel_size=0,
    stride=1,
    padding=0,
    output_padding=0,
    dilation=1,
):
    kernel_size = _pair(kernel_size)
    dilation = _pair(dilation)
    padding = _pair(padding)
    output_padding = _pair(output_padding)
    stride = _pair(stride)

    bs, k_ch, in_h, in_w = input.shape

    ########################################################################
    # TODO:                                                                #
    # Compute the pixel adaptive kernel. The kernel is supposed to follow  #
    # a fixed parametric form of a Gaussian.                               #
    # NOTE: unfolding the input makes computing the kernel easier. This is #
    # a trick also used in the implementation of the convolution operation #
    # NOTE: the unfolding operation is implemented in the nd2col function  #
    # Check the function signature for more details on the parameters      #
    # NOTE: you do not need to reshape the kernel to the correct shape.    #
    # This is done for you.                                                #
    ########################################################################


    pass

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    output = x.view(*(x.shape[:2] + tuple(kernel_size) + x.shape[-2:])).contiguous()

    return output


def pacconv2d(input, kernel, weight, bias=None, stride=1, padding=0, dilation=1):
    # TODO: check shape. Give the shape of the input, kernel, weight, bias
    kernel_size = tuple(weight.shape[-2:])
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    ########################################################################
    # TODO:                                                                #
    # Perform the pixel-adaptive convolution operation. For this you are   #
    # given the input feature map, the previously computed pixel adaptive  #
    # kernel and the weights of the convolution. An optional bias          #
    # is also given. Use it if it is not None.                             #
    # NOTE: Again, the nd2col can be used to unfold the input feature map. #
    # NOTE: Check if the output has the correct shape.                     #
    ########################################################################


    pass

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return output
