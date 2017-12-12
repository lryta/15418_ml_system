#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "tensor.h"
#include "cstdio"

namespace TinyML {
using std::vector;

float* tensor::getGPUData() {
  if (data_gpu_ == NULL)
    cudaMalloc(&data_gpu_, sizeof(float) * shape_.getTotal());
  return data_gpu_;
}

float* tensor::getGPUGrad() {
  if (grad_gpu_ == NULL)
    cudaMalloc((void**)&grad_gpu_, sizeof(float) * shape_.getTotal());
  return grad_gpu_;
}

void tensor::SyncDataCPUToGPU() {
  if (data_gpu_ == NULL)
    cudaMalloc(&data_gpu_, sizeof(float) * shape_.getTotal());
  cudaMemcpy(data_gpu_, data_, sizeof(float) * shape_.getTotal(), cudaMemcpyHostToDevice);
}

void tensor::SyncDataGPUToCPU() {
  assert(data_gpu_ != NULL && data_ != NULL);
  cudaMemcpy(data_, data_gpu_, sizeof(float) * shape_.getTotal(), cudaMemcpyDeviceToHost);
}

void tensor::SyncGradCPUToGPU() {
  assert(grad_ != NULL);
  if (grad_gpu_ == NULL)
    cudaMalloc(&grad_gpu_, sizeof(float) * shape_.getTotal());
  cudaMemcpy(grad_gpu_, grad_, sizeof(float) * shape_.getTotal(), cudaMemcpyHostToDevice);
}

void tensor::SyncGradGPUToCPU() {
  assert(grad_gpu_ != NULL);
  getGrad();
  // make sure grad is allocated;
  cudaMemcpy(grad_, grad_gpu_, sizeof(float) * shape_.getTotal(), cudaMemcpyDeviceToHost);
}

}
