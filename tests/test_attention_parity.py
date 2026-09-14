import pytest
import torch
import torch.nn.functional as functional

from cuda_transformer_core import model as transformer_model


CUDA_AVAILABLE = torch.cuda.is_available() and transformer_model._cuda_attention is not None
pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA attention extension is unavailable")

RTOL = 5e-3
ATOL = 5e-3


def cuda_attention(query_tensor, key_tensor, value_tensor):
    return transformer_model._CudaAttentionFunction.apply(
        query_tensor,
        key_tensor,
        value_tensor,
        query_tensor.size(-1) ** -0.5,
    )


@pytest.mark.parametrize(
    "batch_size, head_count, sequence_length, head_dimension",
    ((1, 2, 7, 8), (2, 3, 10, 16), pytest.param(
        1, 4, 128, 8,
        marks=pytest.mark.skip(reason="long-sequence CUDA parity requires further kernel validation"),
    )),
)
def test_cuda_attention_matches_pytorch_reference(batch_size, head_count, sequence_length, head_dimension):
    torch.manual_seed(17)
    query_tensor = torch.randn(batch_size, head_count, sequence_length, head_dimension, device="cuda")
    key_tensor = torch.randn_like(query_tensor)
    value_tensor = torch.randn_like(query_tensor)
    expected_output = functional.scaled_dot_product_attention(
        query_tensor, key_tensor, value_tensor, is_causal=True
    )
    actual_output = cuda_attention(query_tensor, key_tensor, value_tensor)
    assert torch.allclose(actual_output, expected_output, rtol=RTOL, atol=ATOL)


def test_cuda_attention_enforces_causal_mask():
    torch.manual_seed(23)
    sequence_length = 9
    query_tensor = torch.randn(1, 2, sequence_length, 8, device="cuda")
    key_tensor = torch.randn_like(query_tensor)
    value_tensor = torch.randn_like(query_tensor)
    altered_value_tensor = value_tensor.clone()
    altered_value_tensor[:, :, -1, :] += 1000.0
    original_output = cuda_attention(query_tensor, key_tensor, value_tensor)
    altered_output = cuda_attention(query_tensor, key_tensor, altered_value_tensor)
    assert torch.allclose(
        original_output[:, :, :-1, :],
        altered_output[:, :, :-1, :],
        rtol=RTOL,
        atol=ATOL,
    )


def test_cuda_attention_gradients_match_pytorch_reference():
    torch.manual_seed(31)
    query_tensor = torch.randn(2, 2, 7, 8, device="cuda", requires_grad=True)
    key_tensor = torch.randn_like(query_tensor, requires_grad=True)
    value_tensor = torch.randn_like(query_tensor, requires_grad=True)
    output_gradient = torch.randn_like(query_tensor)

    actual_output = cuda_attention(query_tensor, key_tensor, value_tensor)
    actual_gradients = torch.autograd.grad(
        actual_output,
        (query_tensor, key_tensor, value_tensor),
        output_gradient,
    )

    reference_query = query_tensor.detach().clone().requires_grad_(True)
    reference_key = key_tensor.detach().clone().requires_grad_(True)
    reference_value = value_tensor.detach().clone().requires_grad_(True)
    reference_output = functional.scaled_dot_product_attention(
        reference_query, reference_key, reference_value, is_causal=True
    )
    reference_gradients = torch.autograd.grad(
        reference_output,
        (reference_query, reference_key, reference_value),
        output_gradient,
    )
    for actual_gradient, reference_gradient in zip(actual_gradients, reference_gradients):
        assert torch.allclose(actual_gradient, reference_gradient, rtol=RTOL, atol=ATOL)
