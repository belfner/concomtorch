#include <torch/extension.h>

// Forward declarations from bke_kernels.cu
torch::Tensor bke_std_cuda_forward(const torch::Tensor& input, std::optional<torch::Tensor> labels);
torch::Tensor bke_ic_cuda_forward(const torch::Tensor& input, std::optional<torch::Tensor> labels);
torch::Tensor get_unique_labels_cuda(const torch::Tensor& labels, bool exclude_background, bool collapse_consecutive);
torch::Tensor get_component_masks_cuda(const torch::Tensor& labels, std::optional<torch::Tensor> unique_labels, bool exclude_background, bool collapse_consecutive);

// Register the operators with PyTorch dispatcher
TORCH_LIBRARY(concomtorch, m) {
    // Main API - defaults to BKE_IC for best performance
    m.def("connected_components(Tensor input, Tensor? labels=None) -> Tensor");

    // Algorithm variants for advanced users
    m.def("connected_components_bke(Tensor input, Tensor? labels=None) -> Tensor");
    m.def("connected_components_bke_ic(Tensor input, Tensor? labels=None) -> Tensor");

    // Unique label extraction
    m.def("get_unique_labels(Tensor labels, bool exclude_background=True, bool collapse_consecutive=True) -> Tensor");

    // Component mask extraction
    m.def("get_component_masks(Tensor labels, Tensor? unique_labels=None, bool exclude_background=True, bool collapse_consecutive=True) -> Tensor");
}

// Register CUDA implementations (BKE only)
TORCH_LIBRARY_IMPL(concomtorch, CUDA, m) {
    // Default to BKE_IC for best performance
    m.impl("connected_components",         &bke_ic_cuda_forward);
    m.impl("connected_components_bke",     &bke_std_cuda_forward);
    m.impl("connected_components_bke_ic",  &bke_ic_cuda_forward);
    m.impl("get_unique_labels",            &get_unique_labels_cuda);
    m.impl("get_component_masks",          &get_component_masks_cuda);
}
