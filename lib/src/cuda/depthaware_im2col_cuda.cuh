#include <ATen/ATen.h> 
#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>
#include <cstring>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// #include <THC/THC.h>
#include <THC/THCAtomics.cuh>
// #include <THC/THCDeviceUtils.cuh>

#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 256;
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


template <typename scalar_t>
__global__ void depthaware_im2col_gpu_kernel(const int n,
                                                       const scalar_t *data_im, const scalar_t* data_depth,
                                                       const int height, const int width, 
                                                       const int kernel_h, const int kernel_w,
                                                       const int pad_h, const int pad_w,
                                                       const int stride_h, const int stride_w,
                                                       const int dilation_h, const int dilation_w,
                                                       const int batch_size, const int num_channels,
                                                       const int height_col, const int width_col,
                                                       scalar_t *data_col)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        const int w_col = index % width_col;
        const int h_col = (index / width_col) % height_col;
        const int b_col = (index / width_col / height_col) % batch_size;
        const int c_im = (index / width_col / height_col) / batch_size;
        const int c_col = c_im * kernel_h * kernel_w;

        const int h_in = h_col * stride_h - pad_h;
        const int w_in = w_col * stride_w - pad_w;

        scalar_t *data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
        //const scalar_t* data_im_ptr = data_im + ((b_col * num_channels + c_im) * height + h_in) * width + w_in;
        const scalar_t *data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
        const scalar_t *data_depth_ptr = data_depth + b_col * height * width; 

        bool valid = true; 
        scalar_t Di = static_cast<scalar_t>(0);
        if ((h_in + dilation_h * (kernel_h - 1) / 2) >= 0 && 
            (w_in + dilation_w * (kernel_w - 1) / 2) >= 0 && 
            (h_in + dilation_h * (kernel_h - 1) / 2) < height && 
            (w_in + dilation_w * (kernel_w - 1) / 2) < width) 
            Di = data_depth_ptr[(h_in + dilation_h * (kernel_h - 1) / 2) * width + w_in + dilation_w * (kernel_w - 1) / 2];
        else
            valid = false;

        for (int i = 0; i < kernel_h; ++i)
        {
          for (int j = 0; j < kernel_w; ++j)
          {
            scalar_t val = static_cast<scalar_t>(0);
            scalar_t Dval = static_cast<scalar_t>(0);
            const int h_im = h_in + i * dilation_h;
            const int w_im = w_in + j * dilation_w;
            if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
              val = data_im_ptr[h_im*width + w_im];
              if (valid) 
                Dval = data_depth_ptr[h_im*width + w_im];
            }
            *data_col_ptr = val * exp(-abs(Di - Dval));
            data_col_ptr += batch_size * height_col * width_col;
          }
        }
    }
}

template <typename scalar_t>
__global__ void depthaware_col2im_gpu_kernel(const int n,
                                                       const scalar_t *data_col, const scalar_t *data_depth,
                                                       const int channels, const int height, const int width,
                                                       const int kernel_h, const int kernel_w,
                                                       const int pad_h, const int pad_w,
                                                       const int stride_h, const int stride_w,
                                                       const int dilation_h, const int dilation_w,
                                                       const int batch_size, 
                                                       const int height_col, const int width_col,
                                                       scalar_t *grad_im)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    const int j = (index / width_col / height_col / batch_size) % kernel_w;
    const int i = (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
    const int c = index / width_col / height_col / batch_size / kernel_w / kernel_h;
    // compute the start and end of the output

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int b = (index / width_col / height_col) % batch_size;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const scalar_t *data_depth_ptr = data_depth + b * height * width; 

    bool valid = true; 
    scalar_t Di = static_cast<scalar_t>(0);
    if ((h_in + dilation_h * (kernel_h - 1) / 2) >= 0 && 
        (w_in + dilation_w * (kernel_w - 1) / 2) >= 0 && 
        (h_in + dilation_h * (kernel_h - 1) / 2) < height && 
        (w_in + dilation_w * (kernel_w - 1) / 2) < width) 
        Di = data_depth_ptr[(h_in + dilation_h * (kernel_h - 1) / 2) * width + w_in + dilation_w * (kernel_w - 1) / 2];
    else
        valid = false;

    const int h_im = h_in + i * dilation_h;
    const int w_im = w_in + j * dilation_w;

    scalar_t Dval = static_cast<scalar_t>(0);
    if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
      if (valid) 
        Dval = data_depth_ptr[h_im*width + w_im];
    }
    
    const scalar_t cur_top_grad = data_col[index] * exp(-abs(Di - Dval));
    int cur_bottom_grad_pos = ((b * channels + c) * height + h_im) * width + w_im;
    atomicAdd(grad_im + cur_bottom_grad_pos, cur_top_grad);    
  }
}

template <typename scalar_t>
void depthaware_im2col_cuda(cudaStream_t stream,
  const scalar_t* data_im, const scalar_t* data_depth,
  const int batch_size, const int channels, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
  const int dilation_h, const int dilation_w,
  scalar_t* data_col) {
  // num_axes should be smaller than block size
  const int num_kernels = channels * batch_size * height_col * width_col;
  depthaware_im2col_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(
      num_kernels, data_im, data_depth, height_im, width_im, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      batch_size, channels, height_col, width_col, data_col);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in depthaware_im2col_cuda: %s\n", cudaGetErrorString(err));
  }

}

template <typename scalar_t>
void depthaware_col2im_cuda(cudaStream_t stream,
  const scalar_t* data_col, const scalar_t* data_depth, 
  const int batch_size, const int channels, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
  const int dilation_h, const int dilation_w, 
  scalar_t* grad_im){

  const int num_kernels = channels * kernel_h * kernel_w * batch_size * height_col * width_col;
  depthaware_col2im_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(
        num_kernels, data_col, data_depth, channels, height_im, width_im,
        kernel_h, kernel_w, pad_h, pad_h, stride_h, stride_w,
        dilation_h, dilation_w,
        batch_size, height_col, width_col, grad_im);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in depthaware_col2im_cuda: %s\n", cudaGetErrorString(err));
  }

}