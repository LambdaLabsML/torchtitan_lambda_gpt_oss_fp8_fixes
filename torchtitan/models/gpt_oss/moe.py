# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.models.common.moe import GroupedExperts, MoE
from torchtitan.protocols.module import Module


class ScaleBiasForward(torch.autograd.Function):
    """
    Custom autograd function that scales bias in forward pass but not in backward.

    For tensor parallel MoE, we need to scale the bias by 1/tp_degree in forward
    to cancel the extra reduction effect, but keep the gradient unchanged in backward.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, bias, tp_degree):
        ctx.tp_degree = tp_degree
        if tp_degree > 1:
            return bias / tp_degree
        return bias

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        # Don't scale the gradient - pass it through as-is
        return grad_output, None


def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)


def _run_experts_for_loop(
    mlp1_weight: torch.Tensor,
    mlp1_bias: torch.Tensor,
    mlp2_weight: torch.Tensor,
    mlp2_bias: torch.Tensor,
    swiglu_limit: float,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    tp_degree: int = 1,
) -> torch.Tensor:
    # NOTE: this would incur a synchronization between device and host
    # pyrefly: ignore [bad-assignment]
    num_tokens_per_expert = num_tokens_per_expert.tolist()

    # a tuple of tensors indexed by experts
    # each with shape (tokens_per_expert(varying), dim)
    # pyrefly: ignore [bad-assignment]
    x = torch.split(
        x[: sum(num_tokens_per_expert)],
        # pyrefly: ignore [bad-argument-type]
        split_size_or_sections=num_tokens_per_expert,
        dim=0,
    )
    out_experts_splits = []
    for expert_idx, x_expert in enumerate(x):
        h = (
            torch.matmul(x_expert, mlp1_weight[expert_idx].transpose(-2, -1))
            + mlp1_bias[expert_idx]
        )
        h = swiglu(h, limit=swiglu_limit)
        # Apply custom autograd function to scale bias in forward but not in backward
        b2 = ScaleBiasForward.apply(mlp2_bias[expert_idx], tp_degree)
        h = torch.matmul(h, mlp2_weight[expert_idx].transpose(-2, -1)) + b2
        out_experts_splits.append(h)
    out = torch.cat(out_experts_splits, dim=0)

    return out


_FP8_WRAPPER_TYPE = None


def _get_fp8_wrapper_type():
    global _FP8_WRAPPER_TYPE
    if _FP8_WRAPPER_TYPE is None:
        try:
            from torchao.prototype.moe_training.tensor import (
                Float8TrainingWeightWrapperTensor,
            )

            _FP8_WRAPPER_TYPE = Float8TrainingWeightWrapperTensor
        except ImportError:
            _FP8_WRAPPER_TYPE = type(None)
    return _FP8_WRAPPER_TYPE


def _run_experts_grouped_mm(
    mlp1_weight: torch.Tensor,
    mlp1_bias: torch.Tensor,
    mlp2_weight: torch.Tensor,
    mlp2_bias: torch.Tensor,
    swiglu_limit: float,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    tp_degree: int = 1,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    if isinstance(mlp1_weight, _get_fp8_wrapper_type()):
        # torch._scaled_grouped_mm (used by FP8 path) requires each expert's
        # token count to be divisible by 16.  We pad to that alignment, run
        # the two GEMMs in padded space, then gather only the real tokens back.
        # All index arithmetic stays on-device — no D2H syncs.
        FP8_ALIGN = 16
        num_experts = num_tokens_per_expert.shape[0]

        # Padded per-expert counts and cumulative end offsets.
        zero_i32 = offsets.new_zeros(1)
        padded_counts = (
            (num_tokens_per_expert.to(torch.int32) + FP8_ALIGN - 1)
            // FP8_ALIGN
            * FP8_ALIGN
        )  # (E,)
        padded_offsets = torch.cumsum(padded_counts, dim=0, dtype=torch.int32)  # (E,)

        # Extended start arrays covering experts 0..E-1 plus the slack region
        # (index E), so that searchsorted's return value is always a valid index.
        orig_starts_ext = torch.cat([zero_i32, offsets]).to(torch.int64)  # (E+1,)
        padded_starts_ext = torch.cat([zero_i32, padded_offsets]).to(torch.int64)  # (E+1,)

        # Build a token→padded-destination index map entirely on device.
        src_idx = torch.arange(x.shape[0], device=x.device, dtype=torch.int64)
        expert_id = torch.searchsorted(
            offsets.to(torch.int64), src_idx, right=False
        )  # in [0, E]; E means "slack region"
        offset_within = src_idx - orig_starts_ext[expert_id]
        dst_idx = padded_starts_ext[expert_id] + offset_within  # (N,)

        # Upper-bound padded buffer size (Python int — no D2H needed).
        # Must itself be a multiple of FP8_ALIGN: after the backward transposes
        # padded_A, the outer stride becomes max_padded bytes, and the FP8
        # CUTLASS kernel asserts that stride is divisible by 16.
        _raw = x.shape[0] + (num_experts + 1) * (FP8_ALIGN - 1)
        max_padded = (_raw + FP8_ALIGN - 1) // FP8_ALIGN * FP8_ALIGN

        # Scatter x into the aligned-padded buffer (differentiable; backward
        # of index_put is a gather, which is efficient on CUDA).
        x_padded = x.new_zeros(max_padded, x.shape[-1])
        x_padded = x_padded.index_put((dst_idx,), x)

        # Tail entry absorbs max_padded - padded_offsets[-1] leftover rows so
        # that repeat_interleave's sum == output_size (required by the API).
        padded_tail = (max_padded - padded_offsets[-1]).unsqueeze(0)
        padded_counts_ext = torch.cat(
            [padded_counts, padded_tail.to(padded_counts.dtype)]
        ).long()  # (E+1,)

        # Use torch._grouped_mm with the Float8TrainingWeightWrapperTensor directly.
        # __torch_function__ on the wrapper intercepts this call and dispatches to
        # torchao's _to_fp8_rowwise_then_scaled_grouped_mm, which uses
        # torch._scaled_grouped_mm (CUTLASS) — no custom Triton path, no caching
        # complexity, no graph-break differences vs the BF16 path.
        # We pre-pad the activations (pad_token_groups_for_grouped_mm=False in config).
        h = torch._grouped_mm(
            x_padded, mlp1_weight.transpose(-2, -1), offs=padded_offsets
        )
        b1 = torch.cat([mlp1_bias, mlp1_bias.new_zeros(1, mlp1_bias.shape[-1])])
        b1 = b1.repeat_interleave(padded_counts_ext, dim=0, output_size=max_padded)
        h = h + b1

        h = swiglu(h, limit=swiglu_limit)
        h = torch._grouped_mm(
            h, mlp2_weight.transpose(-2, -1), offs=padded_offsets
        )
        b2 = torch.cat([mlp2_bias, mlp2_bias.new_zeros(1, mlp2_bias.shape[-1])])
        b2 = b2.repeat_interleave(padded_counts_ext, dim=0, output_size=max_padded)
        b2 = ScaleBiasForward.apply(b2, tp_degree)
        h = h + b2

        # Gather real (and slack) tokens back to the original buffer layout.
        # Backward of h[dst_idx] is a scatter — correct gradient placement.
        return h[dst_idx]
    else:
        # BF16 path — no alignment requirement.
        # Pad num_tokens_per_expert with tail slack so that repeat_interleave
        # with output_size=x.shape[0] directly produces a static-shaped output,
        # avoiding the D2H sync that repeat_interleave incurs without output_size.
        tail_slack = (x.shape[0] - offsets[-1]).unsqueeze(0).to(num_tokens_per_expert.dtype)
        num_tokens_per_expert_long = torch.cat([num_tokens_per_expert, tail_slack]).long()

        h = torch._grouped_mm(
            x.bfloat16(), mlp1_weight.transpose(-2, -1).bfloat16(), offs=offsets
        )
        b1 = torch.cat([mlp1_bias, mlp1_bias.new_zeros(1, mlp1_bias.shape[-1])])
        b1 = b1.repeat_interleave(num_tokens_per_expert_long, dim=0, output_size=x.shape[0])
        h = h + b1.to(h.dtype)

        h = swiglu(h, limit=swiglu_limit)
        h = torch._grouped_mm(h, mlp2_weight.transpose(-2, -1).bfloat16(), offs=offsets)

        # Apply custom autograd function to scale bias in forward but not in backward
        b2 = torch.cat([mlp2_bias, mlp2_bias.new_zeros(1, mlp2_bias.shape[-1])])
        b2 = b2.repeat_interleave(num_tokens_per_expert_long, dim=0, output_size=x.shape[0])
        b2 = ScaleBiasForward.apply(b2, tp_degree)
        h = h + b2.to(h.dtype)

        return h


class GptOssGroupedExperts(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        swiglu_limit: float = 7.0

    def __init__(self, config: Config):
        super().__init__()
        dim = config.dim
        hidden_dim = config.hidden_dim
        num_experts = config.num_experts
        self.num_experts = num_experts
        self.use_grouped_mm = config.use_grouped_mm
        self.swiglu_limit = config.swiglu_limit

        self.mlp1_weight = nn.Parameter(
            torch.empty((num_experts, hidden_dim * 2, dim))
        )  # (num_experts, out_dim, in_dim)
        self.mlp1_bias = nn.Parameter(torch.empty((num_experts, hidden_dim * 2)))
        self.mlp2_weight = nn.Parameter(
            torch.empty((num_experts, dim, hidden_dim))
        )  # (num_experts, out_dim, in_dim)
        self.mlp2_bias = nn.Parameter(torch.empty((num_experts, dim)))

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.mlp1_weight, DTensor):
            # Convert parameters from DTensors to plain Tensors, to work with
            # dynamic-shape inputs in EP which cannot be easily expressed as DTensors.
            mlp1_weight = self.mlp1_weight.to_local()
            # pyrefly: ignore [missing-attribute]
            mlp1_bias = self.mlp1_bias.to_local()
            # pyrefly: ignore [missing-attribute]
            mlp2_weight = self.mlp2_weight.to_local()
            # pyrefly: ignore [missing-attribute]
            mlp2_bias = self.mlp2_bias.to_local()
        else:
            mlp1_weight = self.mlp1_weight
            mlp1_bias = self.mlp1_bias
            mlp2_weight = self.mlp2_weight
            mlp2_bias = self.mlp2_bias

        # Determine tp_degree from device mesh if available
        tp_degree = 1
        if isinstance(self.mlp1_weight, DTensor):
            mesh_dim_names = self.mlp1_weight.device_mesh.mesh_dim_names
            # pyrefly: ignore[not-iterable]
            if "tp" in mesh_dim_names:
                # pyrefly: ignore [missing-attribute]
                tp_dim_idx = mesh_dim_names.index("tp")
                tp_degree = self.mlp1_weight.device_mesh.size(tp_dim_idx)

        if self.use_grouped_mm:
            return _run_experts_grouped_mm(
                mlp1_weight,
                mlp1_bias,
                mlp2_weight,
                mlp2_bias,
                self.swiglu_limit,
                x,
                num_tokens_per_expert,
                tp_degree,
            )
        else:
            return _run_experts_for_loop(
                mlp1_weight,
                mlp1_bias,
                mlp2_weight,
                mlp2_bias,
                self.swiglu_limit,
                x,
                num_tokens_per_expert,
                tp_degree,
            )


class GptOssMoE(MoE):
    """GptOss MoE implementation that inherits from the base MoE class."""

    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        swiglu_limit: float = 7.0

    def __init__(self, config: Config):
        # Initialize the base MoE class
        super().__init__(config)

        # Override the base GroupedExperts with GptOssGroupedExperts
        gptoss_experts_config = GptOssGroupedExperts.Config(
            dim=config.experts.dim,
            hidden_dim=config.experts.hidden_dim,
            num_experts=config.experts.num_experts,
            swiglu_limit=config.swiglu_limit,
            use_grouped_mm=config.experts.use_grouped_mm,
            param_init=config.experts.param_init,
        )
        # pyrefly: ignore [bad-assignment]
        self.experts = gptoss_experts_config.build()
