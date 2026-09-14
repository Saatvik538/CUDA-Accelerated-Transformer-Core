import statistics

import torch
import torch.nn.functional as functional

try:
    from cuda_transformer_core import _cuda_attention
except ImportError:
    _cuda_attention = None


WARMUP_RUNS = 10
MEASURED_RUNS = 50
HEAD_COUNT = 8
HEAD_DIMENSION = 64
SEQUENCE_LENGTHS = (128, 256, 512, 1024)
BATCH_SIZES = (1, 8, 32)


def timed_attention(attention_callable):
    for _ in range(WARMUP_RUNS):
        attention_callable()
    torch.cuda.synchronize()
    elapsed_milliseconds = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(MEASURED_RUNS):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        attention_callable()
        end_event.record()
        torch.cuda.synchronize()
        elapsed_milliseconds.append(start_event.elapsed_time(end_event))
    peak_memory_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return statistics.median(elapsed_milliseconds), peak_memory_mib


def main():
    if not torch.cuda.is_available():
        print("CUDA is unavailable; benchmark requires an NVIDIA GPU and CUDA-enabled PyTorch.")
        return
    if _cuda_attention is None:
        print("The CUDA attention extension is unavailable; run python setup.py build_ext --inplace first.")
        return

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print("| Shape | PyTorch ms | CUDA ms | Speedup | PyTorch peak MiB | CUDA peak MiB |")
    print("|---|---:|---:|---:|---:|---:|")
    for batch_size in BATCH_SIZES:
        for sequence_length in SEQUENCE_LENGTHS:
            attention_inputs = [
                tensor.contiguous()
                for tensor in torch.randn(
                    3,
                    batch_size,
                    HEAD_COUNT,
                    sequence_length,
                    HEAD_DIMENSION,
                    device=device,
                    dtype=torch.float32,
                )
            ]
            query_tensor, key_tensor, value_tensor = attention_inputs
            scaling_factor = HEAD_DIMENSION ** -0.5
            pytorch_milliseconds, pytorch_memory_mib = timed_attention(
                lambda: functional.scaled_dot_product_attention(
                    query_tensor,
                    key_tensor,
                    value_tensor,
                    scale=scaling_factor,
                    is_causal=True,
                )
            )
            cuda_milliseconds, cuda_memory_mib = timed_attention(
                lambda: _cuda_attention.forward(
                    query_tensor,
                    key_tensor,
                    value_tensor,
                    scaling_factor,
                    True,
                )
            )
            speedup = pytorch_milliseconds / cuda_milliseconds
            shape = f"[{batch_size}, {HEAD_COUNT}, {sequence_length}, {HEAD_DIMENSION}]"
            print(
                f"| `{shape}` | {pytorch_milliseconds:.3f} | {cuda_milliseconds:.3f} | "
                f"{speedup:.2f}x | {pytorch_memory_mib:.1f} | {cuda_memory_mib:.1f} |"
            )


if __name__ == "__main__":
    main()
