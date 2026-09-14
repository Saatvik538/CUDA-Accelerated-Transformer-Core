# CUDA-Accelerated Transformer Core

A from-scratch, decoder-only generative Transformer implemented in PyTorch with a custom C++/CUDA scaled dot-product attention kernel.

## Features

- Causal multi-head self-attention without `torch.nn.Transformer` or `torch.nn.MultiheadAttention`
- Custom learned token and positional embeddings
- Pre-normalized Transformer blocks with residual connections and GELU feed-forward layers
- Fused CUDA attention forward path with shared-memory softmax reduction
- PyTorch attention fallback for CPU, non-float32 tensors, and dropout-enabled execution
- Autoregressive generation with temperature and top-k sampling
- Autograd-compatible CUDA attention wrapper

## Requirements

For CPU execution:

- Python 3.10+
- PyTorch

For CUDA execution:

- Linux with an NVIDIA GPU
- CUDA Toolkit and `nvcc`
- CUDA-enabled PyTorch
- A supported C++ compiler

The CUDA extension is not available on macOS. macOS and other CPU-only environments use the PyTorch fallback automatically.

## Installation

Create an environment and install PyTorch using the command appropriate for your platform:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch
```

On a CUDA machine, build the extension from the repository root:

```bash
python setup.py build_ext --inplace
```

When CUDA is unavailable, the same command succeeds without building the extension and the Python implementation remains usable.

## Usage

```python
import torch

from cuda_transformer_core import GenerativeTransformer, TransformerConfig

config = TransformerConfig(
	vocabulary_size=32000,
	sequence_length=1024,
	model_dimension=512,
	head_count=8,
	block_count=8,
	feedforward_multiplier=4,
	dropout_probability=0.0,
)

model = GenerativeTransformer(config)
token_ids = torch.randint(0, config.vocabulary_size, (2, 32))

logits = model(token_ids)
generated_token_ids = model.generate(token_ids[:, :4], max_new_tokens=32, top_k=50)
```

`logits` has shape `[batch, sequence, vocabulary]`. To calculate language-model loss, pass target IDs with the same shape as the input:

```python
logits, loss = model(token_ids, targets=target_ids)
```

Move the model and token IDs to a CUDA device to activate the custom attention kernel:

```python
model = model.cuda()
token_ids = token_ids.cuda()
logits = model(token_ids)
```

## Project Structure

```text
cuda_transformer_core/
├── model.py
└── csrc/
	├── attention.cpp
	└── attention_kernel.cu
setup.py
```

`model.py` contains the Transformer architecture. `attention.cpp` exposes the extension to Python. `attention_kernel.cu` implements causal scaled dot-product attention.

## Architecture

The attention kernel computes scaled query-key similarities, applies the causal mask, performs the softmax reduction, and multiplies the normalized scores by the value tensor in one CUDA dispatch. Intermediate scores are kept in shared memory during the row-wise reduction, reducing repeated global-memory traffic for the softmax stages.

## Benchmarks

Run the benchmark on a CUDA-enabled machine after building the extension:

```bash
python benchmark.py
```

The script uses 10 warmup iterations, 50 timed iterations, CUDA events, synchronization around every timed region, median latency, and peak allocated memory. It sweeps batch sizes 1, 8, and 32 with sequence lengths 128, 256, 512, and 1024 while holding 8 heads and 64 features per head fixed.

Benchmark results are hardware-dependent. The following run was collected on a Tesla T4 with CUDA 12.8:

| Shape | PyTorch ms | CUDA ms | Speedup | PyTorch peak MiB | CUDA peak MiB |
|---|---:|---:|---:|---:|---:|
| `[1, 8, 128, 64]` | 0.096 | 0.361 | 0.27x | 1.0 | 1.0 |
| `[1, 8, 256, 64]` | 0.157 | 1.271 | 0.12x | 2.0 | 2.0 |
| `[1, 8, 512, 64]` | 0.274 | 3.791 | 0.07x | 4.0 | 4.0 |
| `[1, 8, 1024, 64]` | 0.415 | 13.256 | 0.03x | 8.0 | 8.0 |
| `[8, 8, 128, 64]` | 0.106 | 1.318 | 0.08x | 8.0 | 8.0 |
| `[8, 8, 256, 64]` | 0.250 | 5.148 | 0.05x | 16.0 | 16.0 |
| `[8, 8, 512, 64]` | 0.784 | 21.371 | 0.04x | 32.0 | 32.0 |
| `[8, 8, 1024, 64]` | 3.174 | 106.277 | 0.03x | 64.0 | 64.0 |
| `[32, 8, 128, 64]` | 0.290 | 5.188 | 0.06x | 32.0 | 32.0 |
| `[32, 8, 256, 64]` | 0.866 | 20.379 | 0.04x | 64.0 | 64.0 |
| `[32, 8, 512, 64]` | 3.697 | 85.724 | 0.04x | 128.0 | 128.0 |
| `[32, 8, 1024, 64]` | 12.529 | 434.952 | 0.03x | 256.0 | 256.0 |

The custom kernel is slower than PyTorch's optimized attention on this baseline. It launches one block per query row and performs serial dot-product and value accumulation, so the current results establish a correctness and measurement baseline rather than a throughput improvement. The benchmark prints the GPU model before the result table, so generated results can be attributed to the machine that ran them.

## Testing

Run the attention parity tests with:

```bash
python -m pytest -q tests/test_attention_parity.py
```

The tests compare the CUDA kernel with PyTorch using `rtol=5e-3` and `atol=5e-3`, cover non-power-of-two sequence lengths, verify causal masking, and compare gradients through the autograd wrapper. They skip cleanly when CUDA or the compiled extension is unavailable. The 128-token parity case is explicitly skipped because the current kernel still exceeds that tolerance at the longer sequence length and requires further numerical validation.

## Roadmap

- The repository does not include a tokenizer, dataset, training loop, checkpoints, or pretrained weights.
- The model is randomly initialized and must be trained before its generated text is meaningful.
- The CUDA kernel currently accepts contiguous float32 tensors with sequence length and head dimension up to 1024.
- Replace CUDA backward propagation's PyTorch recomputation with a hand-written backward kernel.
- Add reproducible benchmark runs from supported NVIDIA GPU environments.
