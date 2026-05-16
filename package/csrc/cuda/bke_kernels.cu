#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>

// ============================================================================
// Block-based Komura Equivalence (BKE) Implementation
// Based on: "Optimized Block-Based Algorithms to Label Connected Components
// on GPUs" (IEEE TPDS 2019)
// ============================================================================

// Background sentinel for the intermediate union-find phase. Block ids are
// non-negative raster indices (the top-left block has id 0), so 0 is a valid
// foreground root and cannot also mean background. A negative sentinel keeps
// background distinguishable from every block id.
constexpr int32_t kBackgroundParent = -1;

// Union-Find helper functions (from paper Algorithm 1)

__device__ __forceinline__ int32_t find(int32_t* labels, int32_t index) {
    while (labels[index] != index) {
        index = labels[index];
    }
    return index;
}

__device__ __forceinline__ int32_t find_n_compress(int32_t* labels, int32_t index) {
    int32_t root = find(labels, index);
    labels[index] = root;
    return root;
}

__device__ __forceinline__ int32_t find_inline_compress(int32_t* labels, int32_t index) {
    int32_t id = index;
    while (labels[index] != index) {
        index = labels[index];
        labels[id] = index;
    }
    return index;
}

__device__ __forceinline__ uint8_t read_info_byte(
    const int32_t* __restrict__ labels,
    const int64_t xL,
    const int64_t x,
    const int64_t y,
    const int64_t width,
    const int64_t height,
    const int64_t step_L) {

    // Info byte is stored in top-right pixel, or bottom-left for odd-width edges
    if (x + 1 < width) {
        // Normal case: stored in top-right pixel
        return static_cast<uint8_t>(labels[xL + 1]);
    } else if (y + 1 < height) {
        // Odd width, last column: stored in bottom-left pixel
        return static_cast<uint8_t>(labels[xL + step_L]);
    }
    // Single-pixel block at the bottom-right corner (odd height AND odd
    // width): no spare cell exists to store the info byte. Its only pixel
    // is the block's top-left pixel, foreground iff the intermediate parent
    // is not the background sentinel. The corner's only predecessor pixels
    // (above, left, up-left) are mutually 8-adjacent, so the corner needs
    // no deferred-union slot of its own, only this foreground bit.
    return labels[xL] >= 0 ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
}

__device__ void atomic_union(int32_t* labels, int32_t index_a, int32_t index_b) {
    bool done = false;
    while (!done) {
        index_a = find(labels, index_a);
        index_b = find(labels, index_b);

        if (index_a < index_b) {
            int32_t index_old = atomicMin(&labels[index_b], index_a);
            done = (index_old == index_b);
            index_b = index_old;
        } else if (index_b < index_a) {
            int32_t index_old = atomicMin(&labels[index_a], index_b);
            done = (index_old == index_a);
            index_a = index_old;
        } else {
            done = true;
        }
    }
}

// Helper function to check if a bit is set
__device__ __forceinline__ bool has_bit(uint16_t bitset, int pos) {
    return (bitset >> pos) & 1;
}

__device__ __forceinline__ bool is_internal_fg(uint8_t info_byte, int pixel_idx) {
    return (info_byte >> pixel_idx) & 1;
}

__device__ __forceinline__ bool needs_union(uint8_t info_byte, int neighbor_idx) {
    return (info_byte >> (5 + neighbor_idx)) & 1;
}

// ============================================================================
// Kernel 1: BKE Initialization (following Algorithm 2 from the paper)
// ============================================================================

template <typename scalar_t>
__global__ void bke_init_kernel(
    const scalar_t* __restrict__ input,
    int32_t* __restrict__ labels,
    const int64_t height,
    const int64_t width,
    const int64_t step_I,
    const int64_t step_L) {

    // Each thread handles one 2x2 block
    const int64_t bx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t by = blockIdx.y * blockDim.y + threadIdx.y;

    const int64_t blocks_w = (width + 1) / 2;
    const int64_t blocks_h = (height + 1) / 2;

    if (bx >= blocks_w || by >= blocks_h) return;

    const int64_t y = by * 2;  // Top-left pixel row of this block
    const int64_t x = bx * 2;  // Top-left pixel col of this block
    const int64_t xI = y * step_I + x;  // Input image index
    const int64_t xL = y * step_L + x;  // Labels image index

    // Initialize block label to its own raster index
    int32_t min_label = xL;

    // Information byte: bits 0-3 for internal pixel flags, bits 5-7 for union flags
    uint8_t info_byte = 0;

    // Build bitset representing which external pixels need checking
    // This represents a 4x4 grid centered on our 2x2 block
    uint16_t bit_set = 0;

    // Check all 4 internal pixels and update bitset
    // Pixel (0,0) - top-left
    if (y < height && x < width && input[xI] > 0) {
        bit_set |= 0x777;  // Set neighbors of pixel (0,0) in 4x4 grid
        info_byte |= (1 << 0);  // Mark pixel as foreground
    }

    // Pixel (0,1) - top-right
    if (y < height && x + 1 < width && input[xI + 1] > 0) {
        bit_set |= (0x777 << 1);  // Set neighbors of pixel (0,1)
        info_byte |= (1 << 1);
    }

    // Pixel (1,0) - bottom-left
    if (y + 1 < height && x < width && input[xI + step_I] > 0) {
        bit_set |= (0x777 << 4);  // Set neighbors of pixel (1,0)
        info_byte |= (1 << 2);
    }

    // Pixel (1,1) - bottom-right
    if (y + 1 < height && x + 1 < width && input[xI + step_I + 1] > 0) {
        // Don't modify bitset - this pixel doesn't connect to any checked neighbors
        info_byte |= (1 << 3);
    }

    // If this block has no foreground pixels, just initialize and return
    // Check if this block has any foreground pixels
    if ((info_byte & 0x0F) == 0) {
        labels[xL] = kBackgroundParent;  // Background block
        // Zero the info cell wherever it lives. The block bottom-right cell
        // is written only in final labeling and is never read before then,
        // so it needs no initialization here.
        if (x + 1 < width) {
            labels[xL + 1] = 0;
        } else if (y + 1 < height) {
            labels[xL + step_L] = 0;
        }
        return;
    }

    // Now check neighbor blocks P, Q, R, S and find minimum connected label
    // Block P: upper-left diagonal (offset: -2*step_L - 2)
    if (y >= 2 && x >= 2) {
        // Check connectivity via the corner pixel
        if (has_bit(bit_set, 0) && input[xI - step_I - 1] > 0) {
            int32_t label_P = xL - 2 * step_L - 2;
            if (label_P < min_label) {
                min_label = label_P;
            }
        }
    }

    // Track which neighbors are connected
    bool connects_Q = false, connects_R = false, connects_S = false;

    // Block Q: directly above
    if (y >= 2) {
        if ((has_bit(bit_set, 1) && x < width && input[xI - step_I] > 0) ||
            (has_bit(bit_set, 2) && x + 1 < width && input[xI - step_I + 1] > 0)) {
            connects_Q = true;
            int32_t label_Q = xL - 2 * step_L;
            if (label_Q < min_label) {
                min_label = label_Q;
            }
        }
    }

    // Block R: upper-right diagonal
    if (y >= 2 && x + 2 < width) {
        if (has_bit(bit_set, 3) && input[xI - step_I + 2] > 0) {
            connects_R = true;
            int32_t label_R = xL - 2 * step_L + 2;
            if (label_R < min_label) {
                min_label = label_R;
            }
        }
    }

    // Block S: directly left
    if (x >= 2) {
        if ((has_bit(bit_set, 4) && y < height && input[xI - 1] > 0) ||
            (has_bit(bit_set, 8) && y + 1 < height && input[xI + step_I - 1] > 0)) {
            connects_S = true;
            int32_t label_S = xL - 2;
            if (label_S < min_label) {
                min_label = label_S;
            }
        }
    }

    // Mark for union all connected neighbors EXCEPT the one we linked to
    if (connects_Q && (xL - 2 * step_L) != min_label) {
        info_byte |= (1 << 5);
    }
    if (connects_R && (xL - 2 * step_L + 2) != min_label) {
        info_byte |= (1 << 6);
    }
    if (connects_S && (xL - 2) != min_label) {
        info_byte |= (1 << 7);
    }

    // Write the block label (linked to minimum connected neighbor)
    labels[xL] = min_label;

    // Store information byte in the top-right pixel of the block
    // Handle odd-width edge case: if no top-right pixel, use bottom-left
    if (x + 1 < width) {
        // Normal case: store in top-right pixel
        labels[xL + 1] = static_cast<int32_t>(info_byte);
    } else if (y + 1 < height) {
        // Odd width, last column: store in bottom-left pixel
        labels[xL + step_L] = static_cast<int32_t>(info_byte);
    }
    // A 1x1 bottom-right block (odd height AND odd width) has no spare cell
    // for the info byte; read_info_byte derives its single foreground bit
    // from the sign of labels[xL] instead, so no storage is needed here.
}

// ============================================================================
// Kernel 2: BKE Compression
// Flattens union-find trees so each block points directly to its root
// ============================================================================

__global__ void bke_compress_kernel(
    int32_t* __restrict__ labels,
    const int64_t height,
    const int64_t width,
    const int64_t step_L,
    const bool use_inline_compress) {

    // Each thread handles one 2x2 block
    const int64_t bx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t by = blockIdx.y * blockDim.y + threadIdx.y;

    const int64_t blocks_w = (width + 1) / 2;
    const int64_t blocks_h = (height + 1) / 2;

    if (bx >= blocks_w || by >= blocks_h) return;

    const int64_t y = by * 2;  // Top-left pixel row of this block
    const int64_t x = bx * 2;  // Top-left pixel col of this block
    const int64_t xL = y * step_L + x;  // Labels image index

    // Skip background blocks (negative sentinel; 0 is a valid block id)
    if (labels[xL] < 0) return;

    // Compress the path to root (updates labels[xL] to point directly to root)
    if (use_inline_compress) {
        find_inline_compress(labels, xL);
    } else {
        find_n_compress(labels, xL);
    }
}

// ============================================================================
// Kernel 3: BKE Reduction
// Performs union operations with neighbors that were marked during initialization
// ============================================================================

__global__ void bke_reduction_kernel(
    int32_t* __restrict__ labels,
    const int64_t height,
    const int64_t width,
    const int64_t step_L) {

    // Each thread handles one 2x2 block
    const int64_t bx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t by = blockIdx.y * blockDim.y + threadIdx.y;

    const int64_t blocks_w = (width + 1) / 2;
    const int64_t blocks_h = (height + 1) / 2;

    if (bx >= blocks_w || by >= blocks_h) return;

    const int64_t y = by * 2;  // Top-left pixel row of this block
    const int64_t x = bx * 2;  // Top-left pixel col of this block
    const int64_t xL = y * step_L + x;  // Labels image index

    // Read the current block's label
    int32_t block_label = labels[xL];

    // Skip background blocks (negative sentinel; 0 is a valid block id)
    if (block_label < 0) return;

    // Read the information byte
    uint8_t info_byte = read_info_byte(labels, xL, x, y, width, height, step_L);

    // Check if block has any foreground pixels
    if ((info_byte & 0x0F) == 0) return;

    // Perform unions with neighbors that were marked during initialization
    // Bit 5: Union with block Q (directly above)
    if (needs_union(info_byte, 0) && y >= 2) {
        int32_t label_Q = xL - 2 * step_L;
        atomic_union(labels, xL, label_Q);
    }

    // Bit 6: Union with block R (upper-right diagonal)
    if (needs_union(info_byte, 1) && y >= 2 && x + 2 < width) {
        int32_t label_R = xL - 2 * step_L + 2;
        atomic_union(labels, xL, label_R);
    }

    // Bit 7: Union with block S (directly left)
    if (needs_union(info_byte, 2) && x >= 2) {
        int32_t label_S = xL - 2;
        atomic_union(labels, xL, label_S);
    }
}

// ============================================================================
// Kernel 5: BKE Final Labeling
// Copies block labels to individual pixels to produce the final output
// ============================================================================

__global__ void bke_final_labeling_kernel(
    int32_t* __restrict__ labels,
    const int64_t height,
    const int64_t width,
    const int64_t step_L) {

    // Each thread handles one 2x2 block
    const int64_t bx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t by = blockIdx.y * blockDim.y + threadIdx.y;

    const int64_t blocks_w = (width + 1) / 2;
    const int64_t blocks_h = (height + 1) / 2;

    if (bx >= blocks_w || by >= blocks_h) return;

    const int64_t y = by * 2;  // Top-left pixel row of this block
    const int64_t x = bx * 2;  // Top-left pixel col of this block
    const int64_t xL = y * step_L + x;  // Labels image index

    // Read the block label (already compressed to root)
    int32_t block_label = labels[xL];

    // Read the information byte
    uint8_t info_byte = read_info_byte(labels, xL, x, y, width, height, step_L);

    // If no foreground pixels in this block, set all to 0. A background block
    // has an all-zero info byte, so this also catches the kBackgroundParent
    // sentinel without a separate label check.
    if ((info_byte & 0x0F) == 0) {
        labels[xL] = 0;
        if (x + 1 < width) {
            labels[xL + 1] = 0;
        }
        if (y + 1 < height) {
            labels[xL + step_L] = 0;
            if (x + 1 < width) {
                labels[xL + step_L + 1] = 0;
            }
        }
        return;
    }

    // Shift labels to start at 1 (0 reserved for background)
    // The block_label is the raster index, so we add 1 to make it a valid label
    int32_t final_label = block_label + 1;

    // Copy block label to each foreground pixel, 0 to background pixels
    // Pixel (0,0) - top-left
    labels[xL] = is_internal_fg(info_byte, 0) ? final_label : 0;

    // Pixel (0,1) - top-right
    if (x + 1 < width) {
        labels[xL + 1] = is_internal_fg(info_byte, 1) ? final_label : 0;
    }

    // Pixel (1,0) - bottom-left
    if (y + 1 < height) {
        labels[xL + step_L] = is_internal_fg(info_byte, 2) ? final_label : 0;

        // Pixel (1,1) - bottom-right
        if (x + 1 < width) {
            labels[xL + step_L + 1] = is_internal_fg(info_byte, 3) ? final_label : 0;
        }
    }
}

// ============================================================================
// Component Mask Creation
// ============================================================================

// One grid-stride kernel produces every component mask in a single launch.
// masks is contiguous (N, H, W) uint8, so the flat index decomposes as
// component * num_pixels + pixel. This replaces the previous one-launch-per-
// component loop, which scaled poorly for thousands of components.
__global__ void create_masks_fused_kernel(
    const int32_t* __restrict__ labels,
    uint8_t* __restrict__ masks,
    const int32_t* __restrict__ unique_labels,
    const int64_t num_components,
    const int64_t num_pixels
) {
    const int64_t total = num_components * num_pixels;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += stride) {
        const int64_t component_idx = idx / num_pixels;
        const int64_t pixel_idx = idx % num_pixels;
        masks[idx] = (labels[pixel_idx] == unique_labels[component_idx])
                         ? static_cast<uint8_t>(1)
                         : static_cast<uint8_t>(0);
    }
}

// ============================================================================
// Component Statistics (fused single-pass kernel)
// ============================================================================

// Accumulates per-component area, bounding box, and coordinate sums in a
// single pass over the label image. Labels passed here are dense ids in
// [0, num_components) so each is a safe direct index into the stat arrays
// (raw BKE labels are sparse and can approach 2^31). area/sum_x/sum_y use
// 64-bit atomics; bbox uses 32-bit atomicMin/atomicMax.
__global__ void component_stats_kernel(
    const int32_t* __restrict__ dense_labels,
    const int64_t height,
    const int64_t width,
    int64_t* __restrict__ area,
    int64_t* __restrict__ sum_row,
    int64_t* __restrict__ sum_col,
    int32_t* __restrict__ min_row,
    int32_t* __restrict__ max_row,
    int32_t* __restrict__ min_col,
    int32_t* __restrict__ max_col
) {
    const int64_t num_pixels = height * width;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < num_pixels;
         idx += stride) {
        const int32_t comp = dense_labels[idx];
        if (comp < 0) continue;  // defensive; dense ids are >= 0
        const int32_t row = static_cast<int32_t>(idx / width);
        const int32_t col = static_cast<int32_t>(idx % width);
        atomicAdd(reinterpret_cast<unsigned long long*>(&area[comp]), 1ULL);
        atomicAdd(reinterpret_cast<unsigned long long*>(&sum_row[comp]),
                  static_cast<unsigned long long>(row));
        atomicAdd(reinterpret_cast<unsigned long long*>(&sum_col[comp]),
                  static_cast<unsigned long long>(col));
        atomicMin(&min_row[comp], row);
        atomicMax(&max_row[comp], row);
        atomicMin(&min_col[comp], col);
        atomicMax(&max_col[comp], col);
    }
}


// ============================================================================
// Main BKE forward function
// ============================================================================

torch::Tensor bke_cuda_forward(
    const torch::Tensor& input,
    torch::Tensor labels,  // Can be empty for auto-allocation
    const bool inline_compress) {

    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D (height, width)");
    TORCH_CHECK(input.scalar_type() == torch::kUInt8 || input.scalar_type() == torch::kBool,
                "Input must be uint8 or bool");

    // Bind every allocation and kernel launch to the input's device so a
    // multi-GPU process does not run on the ambient (wrong) device.
    const at::cuda::CUDAGuard device_guard(input.device());

    auto input_cont = input.contiguous();
    int64_t height = input_cont.size(0);
    int64_t width = input_cont.size(1);

    // Block ids and final labels are int32. Reject inputs whose pixel count
    // cannot be represented before any narrowing happens.
    TORCH_CHECK(height == 0 || width <= std::numeric_limits<int32_t>::max() / height,
                "Input too large for int32 BKE labels (height * width must be < 2^31)");

    // Allocate or validate labels tensor
    if (!labels.defined() || labels.numel() == 0) {
        labels = torch::empty({height, width},
                             torch::TensorOptions().dtype(torch::kInt32).device(input_cont.device()));
    } else {
        TORCH_CHECK(labels.is_cuda(), "Labels must be on CUDA device");
        TORCH_CHECK(labels.dtype() == torch::kInt32, "Labels must be int32");
        TORCH_CHECK(labels.device() == input.device(),
                   "Labels buffer device mismatch: input is on ", input.device(),
                   " but labels buffer is on ", labels.device(),
                   ". The buffer and input must be on the same CUDA device.");
        TORCH_CHECK(labels.dim() == 2 && labels.size(0) == height && labels.size(1) == width,
                   "Labels buffer shape mismatch: expected (", height, ", ", width,
                   ") to match input on ", input.device(),
                   ", got ", labels.sizes(), " on ", labels.device());
        TORCH_CHECK(labels.is_contiguous(),
                   "Preallocated labels buffer must be contiguous "
                   "(it is written in place)");
    }

    // Empty image: nothing to label. Returning here avoids a zero-sized
    // CUDA grid (an invalid launch configuration).
    if (height == 0 || width == 0) {
        return labels;
    }

    const int64_t step_I = width;  // Contiguous row-major storage
    const int64_t step_L = width;

    int64_t blocks_w = (width + 1) / 2;
    int64_t blocks_h = (height + 1) / 2;


    // Configure kernel launch (2D grid, no batch dimension)
    const dim3 threads(16, 16);
    const dim3 blocks(
        (blocks_w + threads.x - 1) / threads.x,
        (blocks_h + threads.y - 1) / threads.y
    );

    // Launch on the caller's current stream so this op orders correctly with
    // producer/consumer work instead of racing on the legacy default stream.
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // BKE Algorithm: 5 kernels following the paper

    // Kernel 1: Initialization (build BitSet, link to minimum neighbor, mark deferred unions)
    AT_DISPATCH_INTEGRAL_TYPES_AND(at::ScalarType::Bool, input_cont.scalar_type(), "bke_init_kernel", [&] {
        bke_init_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            input_cont.data_ptr<scalar_t>(),
            labels.data_ptr<int32_t>(),
            height, width,
            step_I, step_L
        );
    });

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "BKE init kernel failed: ", cudaGetErrorString(err));

    // Kernel 2: First Compression
    bke_compress_kernel<<<blocks, threads, 0, stream>>>(
        labels.data_ptr<int32_t>(),
        height, width, step_L,
        inline_compress
    );

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "BKE compress kernel #1 failed: ", cudaGetErrorString(err));

    // Kernel 3: Reduction (perform deferred unions)
    bke_reduction_kernel<<<blocks, threads, 0, stream>>>(
        labels.data_ptr<int32_t>(),
        height, width, step_L
    );

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "BKE reduction kernel failed: ", cudaGetErrorString(err));

    // Kernel 4: Second Compression (flatten trees after reduction)
    bke_compress_kernel<<<blocks, threads, 0, stream>>>(
        labels.data_ptr<int32_t>(),
        height, width, step_L,
        inline_compress
    );

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "BKE compress kernel #2 failed: ", cudaGetErrorString(err));

    // Kernel 5: Final Labeling (copy block labels to pixels)
    bke_final_labeling_kernel<<<blocks, threads, 0, stream>>>(
        labels.data_ptr<int32_t>(),
        height, width, step_L
    );

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "BKE final labeling kernel failed: ", cudaGetErrorString(err));

    return labels;
}

// Variant with InlineCompression
torch::Tensor bke_ic_cuda_forward(
    const torch::Tensor& input,
    std::optional<torch::Tensor> labels) {
    torch::Tensor labels_tensor = labels.has_value() ? labels.value() : torch::Tensor();
    return bke_cuda_forward(input, labels_tensor, true);
}

// Standard version
torch::Tensor bke_std_cuda_forward(
    const torch::Tensor& input,
    std::optional<torch::Tensor> labels) {
    torch::Tensor labels_tensor = labels.has_value() ? labels.value() : torch::Tensor();
    return bke_cuda_forward(input, labels_tensor, false);
}


// ============================================================================
// Unique Label Extraction
// ============================================================================

torch::Tensor get_unique_labels_cuda(
    const torch::Tensor& labels,
    const bool exclude_background,
    const bool collapse_consecutive
) {
    // Validation
    TORCH_CHECK(labels.is_cuda(), "Labels must be a CUDA tensor");
    TORCH_CHECK(labels.dim() == 2, "Labels must be 2D (height, width)");
    TORCH_CHECK(labels.scalar_type() == torch::kInt32, "Labels must be int32");

    const at::cuda::CUDAGuard device_guard(labels.device());

    auto labels_cont = labels.contiguous();

    torch::Tensor unique_labels_gpu;

    // Compute unique labels using at::_unique2 (stays on GPU)
    if (collapse_consecutive) {
        // Collapse consecutive duplicates first, then unique (faster for CCL)
        // unique_consecutive returns tuple of (output, inverse, counts)
        auto consecutive_result = torch::unique_consecutive(
            labels_cont.view({-1}),
            false,  // return_inverse
            false,  // return_counts
            std::nullopt  // dim
        );
        auto consecutive = std::get<0>(consecutive_result);

        // Now get unique values from the consecutive-collapsed result
        auto unique_result = at::_unique2(consecutive, true, false, false);
        unique_labels_gpu = std::get<0>(unique_result);
    } else {
        // Standard unique (sorts entire array)
        auto unique_result = at::_unique2(labels_cont, true, false, false);
        unique_labels_gpu = std::get<0>(unique_result);
    }

    // Filter background if needed. CCL labels are non-negative (0 is the
    // only background value), so a GPU-side != 0 mask drops exactly the
    // background entry without copying a scalar to the host.
    if (exclude_background && unique_labels_gpu.size(0) > 0) {
        unique_labels_gpu = unique_labels_gpu.masked_select(unique_labels_gpu != 0);
    }

    return unique_labels_gpu;
}


// ============================================================================
// Component Mask Extraction
// ============================================================================

torch::Tensor get_component_masks_cuda(
    const torch::Tensor& labels,
    std::optional<torch::Tensor> unique_labels_opt,
    const bool exclude_background,
    const bool collapse_consecutive
) {
    // Validation
    TORCH_CHECK(labels.is_cuda(), "Labels must be a CUDA tensor");
    TORCH_CHECK(labels.dim() == 2, "Labels must be 2D (height, width)");
    TORCH_CHECK(labels.scalar_type() == torch::kInt32, "Labels must be int32");

    const at::cuda::CUDAGuard device_guard(labels.device());

    auto labels_cont = labels.contiguous();
    const int64_t height = labels_cont.size(0);
    const int64_t width = labels_cont.size(1);
    const int64_t num_pixels = height * width;

    torch::Tensor unique_labels_gpu;

    // If unique_labels provided, use directly; otherwise compute
    if (unique_labels_opt.has_value() && unique_labels_opt.value().defined()) {

        auto unique_labels_tensor = unique_labels_opt.value();

        // Validate provided unique_labels
        TORCH_CHECK(unique_labels_tensor.is_cuda(), "unique_labels must be a CUDA tensor");
        TORCH_CHECK(unique_labels_tensor.dim() == 1, "unique_labels must be 1D");
        TORCH_CHECK(unique_labels_tensor.scalar_type() == torch::kInt32, "unique_labels must be int32");
        TORCH_CHECK(unique_labels_tensor.device() == labels.device(),
                    "unique_labels device mismatch: labels are on ", labels.device(),
                    " but unique_labels is on ", unique_labels_tensor.device());

        unique_labels_gpu = unique_labels_tensor.contiguous();
    } else {
        // Auto-compute using the extracted function
        unique_labels_gpu = get_unique_labels_cuda(labels_cont, exclude_background, collapse_consecutive);
    }

    const int64_t num_components = unique_labels_gpu.size(0);

    // Handle empty case
    if (num_components == 0) {
        return torch::empty({0, height, width},
            torch::TensorOptions().dtype(torch::kUInt8).device(labels_cont.device()));
    }

    // Empty label map with caller-supplied components: return correctly
    // shaped all-zero masks without launching a zero-sized grid.
    if (num_pixels == 0) {
        return torch::zeros({num_components, height, width},
            torch::TensorOptions().dtype(torch::kUInt8).device(labels_cont.device()));
    }

    // Allocate output tensor: (N, H, W) uint8. The fused kernel writes every
    // element, so an uninitialized buffer is safe and avoids a memset.
    auto masks = torch::empty({num_components, height, width},
        torch::TensorOptions().dtype(torch::kUInt8).device(labels_cont.device()));

    // One fused launch over all N*H*W mask elements (grid-stride), instead of
    // one kernel launch per component.
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int threads_per_block = 256;
    const int64_t total = num_components * num_pixels;
    int64_t num_blocks = (total + threads_per_block - 1) / threads_per_block;
    if (num_blocks > 65535) {
        num_blocks = 65535;  // grid-stride loop covers the remainder
    }

    create_masks_fused_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        labels_cont.data_ptr<int32_t>(),
        masks.data_ptr<uint8_t>(),
        unique_labels_gpu.data_ptr<int32_t>(),
        num_components,
        num_pixels
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Mask creation failed: ", cudaGetErrorString(err));

    return masks;
}


// ============================================================================
// Component Statistics
// ============================================================================

// Computes per-component area, bounding box, and centroid in a single pass
// over the label image. dense_labels must hold contiguous ids in
// [0, num_components) (the Python wrapper relabels first), so each value
// directly indexes the per-component accumulators.
//
// Returns a 3-tuple:
//   area     : int64  (N,)      pixel count per component
//   bbox     : int32  (N, 4)    [min_row, min_col, max_row, max_col] inclusive
//   centroid : float64 (N, 2)   [row, col] = coordinate mean
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> component_stats_cuda(
    const torch::Tensor& dense_labels,
    const int64_t num_components
) {
    TORCH_CHECK(dense_labels.is_cuda(), "dense_labels must be a CUDA tensor");
    TORCH_CHECK(dense_labels.dim() == 2, "dense_labels must be 2D (height, width)");
    TORCH_CHECK(dense_labels.scalar_type() == torch::kInt32, "dense_labels must be int32");
    TORCH_CHECK(num_components >= 0, "num_components must be non-negative");

    const at::cuda::CUDAGuard device_guard(dense_labels.device());

    auto labels_cont = dense_labels.contiguous();
    const int64_t height = labels_cont.size(0);
    const int64_t width = labels_cont.size(1);
    const auto device = labels_cont.device();

    auto i64_opts = torch::TensorOptions().dtype(torch::kInt64).device(device);
    auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto f64_opts = torch::TensorOptions().dtype(torch::kFloat64).device(device);

    auto area = torch::zeros({num_components}, i64_opts);
    auto sum_row = torch::zeros({num_components}, i64_opts);
    auto sum_col = torch::zeros({num_components}, i64_opts);
    auto min_row = torch::full({num_components}, std::numeric_limits<int32_t>::max(), i32_opts);
    auto max_row = torch::full({num_components}, -1, i32_opts);
    auto min_col = torch::full({num_components}, std::numeric_limits<int32_t>::max(), i32_opts);
    auto max_col = torch::full({num_components}, -1, i32_opts);

    const int64_t num_pixels = height * width;
    if (num_components > 0 && num_pixels > 0) {
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        const int threads_per_block = 256;
        int64_t num_blocks = (num_pixels + threads_per_block - 1) / threads_per_block;
        if (num_blocks > 65535) {
            num_blocks = 65535;  // grid-stride loop covers the remainder
        }
        component_stats_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            labels_cont.data_ptr<int32_t>(),
            height, width,
            area.data_ptr<int64_t>(),
            sum_row.data_ptr<int64_t>(),
            sum_col.data_ptr<int64_t>(),
            min_row.data_ptr<int32_t>(),
            max_row.data_ptr<int32_t>(),
            min_col.data_ptr<int32_t>(),
            max_col.data_ptr<int32_t>()
        );
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "Component stats failed: ", cudaGetErrorString(err));
    }

    auto bbox = torch::stack({min_row, min_col, max_row, max_col}, 1);

    // Centroid = coordinate sum / area. Clamp the denominator so a component
    // with zero pixels yields 0 rather than inf/nan (defensive; dense ids
    // derived from present labels always have area >= 1).
    auto area_f = area.to(torch::kFloat64).clamp_min(1.0);
    auto centroid = torch::stack(
        {sum_row.to(torch::kFloat64) / area_f, sum_col.to(torch::kFloat64) / area_f}, 1);

    return std::make_tuple(area, bbox, centroid);
}