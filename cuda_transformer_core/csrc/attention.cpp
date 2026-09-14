#include <torch/extension.h>

torch::Tensor scaled_dot_product_attention_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    double scale,
    bool causal);

torch::Tensor scaled_dot_product_attention(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    double scale,
    bool causal) {
    TORCH_CHECK(queries.is_cuda(), "queries must be CUDA tensors");
    TORCH_CHECK(keys.is_cuda(), "keys must be CUDA tensors");
    TORCH_CHECK(values.is_cuda(), "values must be CUDA tensors");
    TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
    TORCH_CHECK(keys.scalar_type() == torch::kFloat32, "keys must be float32");
    TORCH_CHECK(values.scalar_type() == torch::kFloat32, "values must be float32");
    TORCH_CHECK(queries.dim() == 4 && keys.dim() == 4 && values.dim() == 4, "expected rank-4 tensors");
    TORCH_CHECK(queries.is_contiguous() && keys.is_contiguous() && values.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(queries.size(0) == keys.size(0) && queries.size(0) == values.size(0), "batch dimensions must match");
    TORCH_CHECK(queries.size(1) == keys.size(1) && queries.size(1) == values.size(1), "head dimensions must match");
    TORCH_CHECK(queries.size(2) == keys.size(2) && queries.size(2) == values.size(2), "sequence dimensions must match");
    TORCH_CHECK(queries.size(3) == keys.size(3) && queries.size(3) == values.size(3), "feature dimensions must match");
    TORCH_CHECK(queries.size(2) <= 1024, "sequence length must be <= 1024");
    TORCH_CHECK(queries.size(3) <= 1024, "head dimension must be <= 1024");
    return scaled_dot_product_attention_cuda(queries, keys, values, scale, causal);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward", &scaled_dot_product_attention, "Scaled dot-product attention");
}
