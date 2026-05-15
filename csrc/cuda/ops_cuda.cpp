#include <torch/extension.h>

// Forward declarations from bke_kernels.cu
torch::Tensor bke_std_cuda_forward(const torch::Tensor& input, torch::Tensor labels);
torch::Tensor bke_ic_cuda_forward(const torch::Tensor& input, torch::Tensor labels);
torch::Tensor get_unique_labels_cuda(const torch::Tensor& labels, bool exclude_background, bool collapse_consecutive);
torch::Tensor get_component_masks_cuda(const torch::Tensor& labels, std::optional<torch::Tensor> unique_labels, bool exclude_background, bool collapse_consecutive);
