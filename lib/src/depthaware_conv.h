#pragma once

#include "cpu/depthaware_conv_cpu.h"

#ifdef WITH_CUDA
#include "cuda/depthaware_conv_cuda.h"
#endif


at::Tensor
depthaware_conv_forward(const at::Tensor &input,
                    const at::Tensor &depth, 
                    const at::Tensor &weight,
                    const at::Tensor &bias,
                    const int kernel_h,
                    const int kernel_w,
                    const int stride_h,
                    const int stride_w,
                    const int pad_h,
                    const int pad_w,
                    const int dilation_h,
                    const int dilation_w,
                    const int group, 
                    const int im2col_step)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return depthaware_conv_cuda_forward(input, depth, weight, bias, 
                                   kernel_h, kernel_w,
                                   stride_h, stride_w,
                                   pad_h, pad_w,
                                   dilation_h, dilation_w,
                                   group,
                                   im2col_step);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor>
depthaware_conv_backward(const at::Tensor &input,
                     const at::Tensor &depth, 
                     const at::Tensor &weight,
                     const at::Tensor &bias,
                     const at::Tensor &grad_output,
                     const int kernel_h, 
                     const int kernel_w,
                     const int stride_h, 
                     const int stride_w,
                     const int pad_h, 
                     const int pad_w,
                     const int dilation_h, 
                     const int dilation_w,
                     const int group,
                     const int im2col_step)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return depthaware_conv_cuda_backward(input,
                                    depth, 
                                    weight,
                                    bias,
                                    grad_output,
                                    kernel_h, kernel_w,
                                    stride_h, stride_w,
                                    pad_h, pad_w,
                                    dilation_h, dilation_w,
                                    group,
                                    im2col_step);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

