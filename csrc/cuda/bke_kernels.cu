#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

// ============================================================================
// Block-based Komura Equivalence (BKE) Implementation
// Based on: "Optimized Block-Based Algorithms to Label Connected Components
// on GPUs" (IEEE TPDS 2019)
// ============================================================================

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
    // TODO Handle edge case
    // Edge case: single-pixel block at bottom-right corner
    // This would need separate storage, but is extremely rare
    return 0;
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
    int32_t block_label = xL;
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
        labels[xL] = 0;  // Background block
        // TODO unsure about this part
        // Store info_byte byte
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
    // If this is the very last block (1x1 at bottom-right corner of odd-sized image),
    // we'd need separate storage, but this is a rare edge case
    // Todo handle this edge case
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

    // Skip background blocks (label 0)
    if (labels[xL] == 0) return;

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

    // Skip background blocks
    if (block_label == 0) return;

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

    // If no foreground pixels in this block, set all to 0
    if ((info_byte & 0x0F) == 0 || block_label == 0) {
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

    auto input_cont = input.contiguous();
    int64_t height = input_cont.size(0);
    int64_t width = input_cont.size(1);

    // Allocate or validate labels tensor
    if (!labels.defined() || labels.numel() == 0) {
        labels = torch::empty({height, width},
                             torch::TensorOptions().dtype(torch::kInt32).device(input_cont.device()));
    } else {
        TORCH_CHECK(labels.is_cuda(), "Labels must be on CUDA device");
        TORCH_CHECK(labels.dtype() == torch::kInt32, "Labels must be int32");
        TORCH_CHECK(labels.dim() == 2 && labels.size(0) == height && labels.size(1) == width,
                   "Labels shape mismatch");
        labels = labels.contiguous();
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

    // BKE Algorithm: 5 kernels following the paper

    // Kernel 1: Initialization (build BitSet, link to minimum neighbor, mark deferred unions)
    AT_DISPATCH_INTEGRAL_TYPES(input_cont.scalar_type(), "bke_init_kernel", [&] {
        bke_init_kernel<scalar_t><<<blocks, threads>>>(
            input_cont.data_ptr<scalar_t>(),
            labels.data_ptr<int32_t>(),
            height, width,
            step_I, step_L
        );
    });

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "BKE init kernel failed: ", cudaGetErrorString(err));

    // Kernel 2: First Compression
    bke_compress_kernel<<<blocks, threads>>>(
        labels.data_ptr<int32_t>(),
        height, width, step_L,
        inline_compress
    );

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "BKE compress kernel #1 failed: ", cudaGetErrorString(err));

    // Kernel 3: Reduction (perform deferred unions)
    bke_reduction_kernel<<<blocks, threads>>>(
        labels.data_ptr<int32_t>(),
        height, width, step_L
    );

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "BKE reduction kernel failed: ", cudaGetErrorString(err));

    // Kernel 4: Second Compression (flatten trees after reduction)
    bke_compress_kernel<<<blocks, threads>>>(
        labels.data_ptr<int32_t>(),
        height, width, step_L,
        inline_compress
    );

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "BKE compress kernel #2 failed: ", cudaGetErrorString(err));

    // Kernel 5: Final Labeling (copy block labels to pixels)
    bke_final_labeling_kernel<<<blocks, threads>>>(
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
