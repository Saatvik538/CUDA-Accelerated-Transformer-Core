from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as functional

try:
    from . import _cuda_attention
except ImportError:
    _cuda_attention = None


class _CudaAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(context, queries, keys, values, scaling_factor):
        context.save_for_backward(queries, keys, values)
        context.scaling_factor = scaling_factor
        return _cuda_attention.forward(queries, keys, values, scaling_factor, True)

    @staticmethod
    def backward(context, output_gradient):
        queries, keys, values = context.saved_tensors
        with torch.enable_grad():
            differentiable_queries = queries.detach().requires_grad_(True)
            differentiable_keys = keys.detach().requires_grad_(True)
            differentiable_values = values.detach().requires_grad_(True)
            reference_output = functional.scaled_dot_product_attention(
                differentiable_queries,
                differentiable_keys,
                differentiable_values,
                is_causal=True,
            )
            query_gradient, key_gradient, value_gradient = torch.autograd.grad(
                reference_output,
                (differentiable_queries, differentiable_keys, differentiable_values),
                output_gradient,
            )
        return query_gradient, key_gradient, value_gradient, None


@dataclass(frozen=True)
class TransformerConfig:
    vocabulary_size: int
    sequence_length: int
    model_dimension: int = 512
    head_count: int = 8
    block_count: int = 8
    feedforward_multiplier: int = 4
    dropout_probability: float = 0.0

    def __post_init__(self):
        if self.model_dimension % self.head_count:
            raise ValueError("model_dimension must divide evenly across head_count")
        if self.vocabulary_size < 1 or self.sequence_length < 1:
            raise ValueError("vocabulary_size and sequence_length must be positive")


class PositionalEncoding(nn.Module):
    def __init__(self, sequence_length, model_dimension):
        super().__init__()
        self.position_table = nn.Embedding(sequence_length, model_dimension)

    def forward(self, token_states):
        token_count = token_states.size(1)
        positions = torch.arange(token_count, device=token_states.device)
        return token_states + self.position_table(positions).unsqueeze(0)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, model_dimension, head_count, dropout_probability=0.0):
        super().__init__()
        self.head_count = head_count
        self.head_dimension = model_dimension // head_count
        self.query_key_value = nn.Linear(model_dimension, model_dimension * 3, bias=False)
        self.output_projection = nn.Linear(model_dimension, model_dimension, bias=False)
        self.dropout_probability = dropout_probability

    def forward(self, token_states):
        batch_size, token_count, model_dimension = token_states.shape
        query_key_value = self.query_key_value(token_states)
        query_tensor, key_tensor, value_tensor = query_key_value.chunk(3, dim=-1)
        query_tensor = query_tensor.view(batch_size, token_count, self.head_count, self.head_dimension).transpose(1, 2).contiguous()
        key_tensor = key_tensor.view(batch_size, token_count, self.head_count, self.head_dimension).transpose(1, 2).contiguous()
        value_tensor = value_tensor.view(batch_size, token_count, self.head_count, self.head_dimension).transpose(1, 2).contiguous()
        scaling_factor = self.head_dimension ** -0.5
        if (
            token_states.is_cuda
            and _cuda_attention is not None
            and token_states.dtype == torch.float32
            and self.dropout_probability == 0.0
        ):
            attended_states = _CudaAttentionFunction.apply(query_tensor, key_tensor, value_tensor, scaling_factor)
        else:
            attended_states = functional.scaled_dot_product_attention(
                query_tensor, key_tensor, value_tensor,
                dropout_p=self.dropout_probability if self.training else 0.0,
                is_causal=True,
            )
        attended_states = attended_states.transpose(1, 2).reshape(batch_size, token_count, model_dimension)
        return self.output_projection(attended_states)


class TransformerBlock(nn.Module):
    def __init__(self, model_dimension, head_count, feedforward_multiplier=4, dropout_probability=0.0):
        super().__init__()
        self.attention_normalization = nn.LayerNorm(model_dimension)
        self.attention = MultiHeadSelfAttention(model_dimension, head_count, dropout_probability)
        self.feedforward_normalization = nn.LayerNorm(model_dimension)
        feedforward_dimension = model_dimension * feedforward_multiplier
        self.feedforward = nn.Sequential(
            nn.Linear(model_dimension, feedforward_dimension, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(feedforward_dimension, model_dimension, bias=False),
        )
        self.dropout_probability = dropout_probability

    def forward(self, token_states):
        token_states = token_states + functional.dropout(
            self.attention(self.attention_normalization(token_states)),
            self.dropout_probability,
            self.training,
        )
        token_states = token_states + functional.dropout(
            self.feedforward(self.feedforward_normalization(token_states)),
            self.dropout_probability,
            self.training,
        )
        return token_states


class FinalOutputProjection(nn.Module):
    def __init__(self, model_dimension, vocabulary_size):
        super().__init__()
        self.normalization = nn.LayerNorm(model_dimension)
        self.vocabulary_projection = nn.Linear(model_dimension, vocabulary_size, bias=False)

    def forward(self, token_states):
        return self.vocabulary_projection(self.normalization(token_states))


class GenerativeTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.model_dimension)
        self.position_encoding = PositionalEncoding(config.sequence_length, config.model_dimension)
        self.transformer_blocks = nn.ModuleList(
            TransformerBlock(
                config.model_dimension,
                config.head_count,
                config.feedforward_multiplier,
                config.dropout_probability,
            )
            for _ in range(config.block_count)
        )
        self.output_projection = FinalOutputProjection(config.model_dimension, config.vocabulary_size)

    def forward(self, token_ids, targets=None):
        if token_ids.dim() != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.size(1) > self.config.sequence_length:
            raise ValueError("token sequence exceeds configured sequence_length")
        token_states = self.position_encoding(self.token_embedding(token_ids))
        for transformer_block in self.transformer_blocks:
            token_states = transformer_block(token_states)
        logits = self.output_projection(token_states)
        if targets is None:
            return logits
        loss = functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens, temperature=1.0, top_k=None):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        for _ in range(max_new_tokens):
            context_ids = token_ids[:, -self.config.sequence_length:]
            next_token_logits = self(context_ids)[:, -1, :] / temperature
            if top_k is not None:
                cutoff = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1))).values[..., -1, None]
                next_token_logits = next_token_logits.masked_fill(next_token_logits < cutoff, float("-inf"))
            next_token = torch.multinomial(next_token_logits.softmax(dim=-1), 1)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids
