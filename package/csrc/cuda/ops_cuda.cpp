#include <torch/extension.h>

#include <tuple>

// Forward declarations from bke_kernels.cu. Signatures must match the
// definitions there exactly (labels is std::optional for auto-allocation).
torch::Tensor bke_std_cuda_forward(const torch::Tensor& input, std::optional<torch::Tensor> labels);
torch::Tensor bke_ic_cuda_forward(const torch::Tensor& input, std::optional<torch::Tensor> labels);
torch::Tensor get_unique_labels_cuda(const torch::Tensor& labels, bool exclude_background, bool collapse_consecutive);
torch::Tensor get_component_masks_cuda(const torch::Tensor& labels, std::optional<torch::Tensor> unique_labels, bool exclude_background, bool collapse_consecutive);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> component_stats_cuda(const torch::Tensor& dense_labels, int64_t num_components);
