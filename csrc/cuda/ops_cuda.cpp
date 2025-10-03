#include <torch/extension.h>

// Forward declarations from bke_kernels.cu
torch::Tensor bke_std_cuda_forward(const torch::Tensor& input, torch::Tensor labels);
torch::Tensor bke_ic_cuda_forward(const torch::Tensor& input, torch::Tensor labels);
