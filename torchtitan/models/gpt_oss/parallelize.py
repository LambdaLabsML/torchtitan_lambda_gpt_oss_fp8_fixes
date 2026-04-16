# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch._inductor.config
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_tensor, Partial, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.components.quantization.float8 import find_float8_linear_config
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.compile import apply_compile_sparse
from torchtitan.distributed.context_parallel import apply_cp_to_attention_module
from torchtitan.distributed.expert_parallel import (
    ExpertParallel,
    ReordererSequenceParallel,
    TorchAOExpertParallel,
)
from torchtitan.distributed.tensor_parallel import NoParallel
from torchtitan.models.gpt_oss.model import GptOssModel
from torchtitan.models.llama4.parallelize import apply_fsdp
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger

from .expert_parallel import (
    GptossDeepEPExpertParallel,
    GptossExpertTensorParallel,
    GptossTensorParallel,
)
from .moe import GptOssDeepEPMoE


# Adapted from llama4/infra/parallelize.py
def parallelize_gptoss(
    model: GptOssModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if parallel_dims.tp_enabled:
        if parallelism.enable_async_tensor_parallel and not model_compile_enabled:
            raise RuntimeError("Async TP requires torch.compile")

        float8_config = find_float8_linear_config(model_converters.converters)
        enable_float8_linear = float8_config is not None
        float8_is_rowwise = float8_config is not None and float8_config.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )

        # For now, float8 all-gather with TP is only supported for tensorwise
        # float8 scaling recipes. For rowwise recipes, we use regular TP and
        # all-gather happens in high precision.
        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise

        enable_sp = parallelism.enable_sequence_parallel

        apply_non_moe_tp(
            model,
            parallel_dims.get_mesh("tp"),
            enable_loss_parallel=not parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
            enable_async_tp=False,
            enable_sp=enable_sp,
        )

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        from torchtitan.components.quantization import find_pad_multiple

        pad_multiple = find_pad_multiple(model_converters.converters)

        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            ep_etp_mesh=parallel_dims.get_optional_mesh(["ep", "etp"]),
            etp_enabled=parallel_dims.etp_enabled,
            pad_multiple=pad_multiple,
            comm_backend=parallelism.expert_parallel_comm_backend,
        )

    if parallel_dims.cp_enabled:
        apply_cp_to_attention_module(
            # pyrefly: ignore [missing-attribute, not-callable]
            [block.attention.inner_attention for block in model.layers.values()],
            parallel_dims.get_mesh("cp"),
        )

    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
        )

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if model_compile_enabled:
        apply_compile_sparse(model, compile_config, parallel_dims.ep_enabled)
        if parallel_dims.ep_enabled:
            _apply_compile_gptoss_ep(compile_config)

    dp_mesh_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

    edp_mesh = None
    if parallel_dims.ep_enabled:
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

    apply_fsdp(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
    )

    logger.info("Applied fully_shard to the model")

    if parallel_dims.cp_enabled:
        logger.info("Applied Context Parallel to the model")

    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the model")

    return model


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    enable_loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
    enable_async_tp: bool,
    enable_sp: bool = True,
):
    """Apply tensor parallelism."""
    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer
    sp_layout = Shard(1) if enable_sp else Replicate()
    embed_plan = RowwiseParallel(
        input_layouts=Replicate(),
        output_layouts=sp_layout,
        use_local_output=enable_sp,
    )

    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": embed_plan,
            "norm": SequenceParallel() if enable_sp else NoParallel(),
            "output": ColwiseParallel(
                input_layouts=sp_layout,
                output_layouts=Shard(-1) if enable_loss_parallel else Replicate(),
                use_local_output=not enable_loss_parallel,
            ),
        },
    )

    # Apply tensor + sequence parallelism to every transformer block
    norm_plan = SequenceParallel() if enable_sp else NoParallel()
    rowwise_output_plan = RowwiseParallel(
        output_layouts=sp_layout, use_local_output=enable_sp
    )

    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():
        layer_plan = {
            "attention_norm": norm_plan,
            "attention": PrepareModuleInput(
                input_layouts=(sp_layout, Replicate(), None, None),
                desired_input_layouts=(Replicate(), Replicate(), None, None),
            ),
            "attention.wq": ColwiseParallel(use_local_output=False),
            "attention.wk": ColwiseParallel(use_local_output=False),
            "attention.wv": ColwiseParallel(use_local_output=False),
            "attention.wo": rowwise_output_plan,
            "ffn_norm": norm_plan,
        }

        # shard attention.sinks across heads
        # pyrefly: ignore [missing-attribute]
        attn = transformer_block.attention
        attn.register_parameter(
            "sinks",
            nn.Parameter(distribute_tensor(attn.sinks, tp_mesh, [Shard(0)])),
        )

        parallelize_module(
            # pyrefly: ignore [bad-argument-type]
            module=transformer_block,
            device_mesh=tp_mesh,
            # pyrefly: ignore [bad-argument-type]
            parallelize_plan=layer_plan,
        )

    if enable_async_tp:
        torch._inductor.config._micro_pipeline_tp = True

    logger.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}{'Async ' if enable_async_tp else ''}"
        "Tensor Parallelism to the model"
    )


def apply_moe_ep_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
    ep_etp_mesh: DeviceMesh | None,
    etp_enabled: bool,
    pad_multiple: int | None = None,
    comm_backend: str = "standard",
):
    assert ep_mesh is not None or tp_mesh is not None

    use_deepep = comm_backend in ("deepep", "hybridep")

    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():
        # pyrefly: ignore [missing-attribute]
        if not transformer_block.moe_enabled:
            continue

        # Switch the MoE forward to the DeepEP variant when requested.
        # We replace the class on the existing instance so the weights and
        # buffers are preserved without re-initializing the module.
        if use_deepep and ep_mesh is not None and not etp_enabled:
            # pyrefly: ignore [missing-attribute]
            moe = transformer_block.moe
            moe.__class__ = GptOssDeepEPMoE
            # DeepEP handles routing; the TokenReorderer is not used.
            moe.reorderer = None

        if tp_mesh is not None:
            moe_layer_plan = {
                # input / output sharding on the seqlen dim
                # all-gather for input, reduce-scatter for output
                "moe": PrepareModuleInputOutput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                    # Keep input as a DTensor from SequenceParallel, do not wrap with to_local.
                    use_local_input=False,
                    output_layouts=(Partial(),),
                    desired_output_layouts=(Shard(1),),
                ),
                # replicate computation for the router
                "moe.router.gate": NoParallel(
                    local_output_grad_placements=(Partial(),),
                ),
            }
            if ep_mesh is not None and not etp_enabled and not use_deepep:
                # If TP is borrowed for EP, then split the tokens across TP ranks so that
                # the reorderer, the all-to-all comms, and routed experts computation
                # are effectively running Sequence Parallel (split along the folded bs*slen dim)
                # pyrefly: ignore [no-matching-overload]
                moe_layer_plan.update({"moe.reorderer": ReordererSequenceParallel()})

            parallelize_module(
                # pyrefly: ignore [bad-argument-type]
                module=transformer_block,
                device_mesh=tp_mesh,
                # pyrefly: ignore [bad-argument-type]
                parallelize_plan=moe_layer_plan,
            )

        experts_mesh, experts_plan = None, None
        if ep_mesh is None:
            experts_mesh = tp_mesh
            # input Replicate, output Partial
            experts_plan = GptossTensorParallel()
        elif tp_mesh is None or not etp_enabled:
            experts_mesh = ep_mesh
            if use_deepep:
                # DeepEP dispatch/combine replaces all-to-all; pad_multiple is
                # handled after dispatch inside _run_experts_grouped_mm (FP8 path).
                # pyrefly: ignore [missing-attribute]
                experts_plan = GptossDeepEPExpertParallel(
                    score_before_experts=transformer_block.moe.score_before_experts,
                    comm_backend=comm_backend,
                    pad_multiple=pad_multiple,
                )
            elif pad_multiple is not None:
                experts_plan = TorchAOExpertParallel(pad_multiple)
            else:
                # input / output sharding on the batch / tokens dim
                experts_plan = ExpertParallel()
        else:
            if pad_multiple is not None:
                raise NotImplementedError(
                    "Quantized grouped GEMMs (FP8/MXFP8) with Expert Tensor "
                    "Parallelism (ETP) is not yet supported."
                )
            experts_mesh = ep_etp_mesh
            experts_plan = GptossExpertTensorParallel()

        parallelize_module(
            # pyrefly: ignore [missing-attribute]
            module=transformer_block.moe.experts,
            device_mesh=experts_mesh,
            parallelize_plan=experts_plan,
        )


def _apply_compile_gptoss_ep(compile_config: CompileConfig) -> None:
    """Compile gpt_oss._run_experts_grouped_mm and wrap it for EP dynamic shapes.

    Called from parallelize_gptoss when both compile and EP are enabled.
    Patches the module-level function so GptOssGroupedExperts.forward picks it up
    without any changes to that class.

    The wrapper:
    - Short-circuits zero-token ranks: the FP8 CUTLASS kernel fails on empty
      tensors (backward produces a [hidden, 0] shaped grad); return an empty
      tensor directly instead.
    - Marks token count dim (dim 0) as dynamic so dynamo compiles once across
      all token counts rather than re-specializing every step.
    """
    import torchtitan.models.gpt_oss.moe as gpt_oss_moe

    # Guard against double-patching (apply_compile_sparse is called once per PP
    # stage, so this function may be called multiple times).
    already_patched = (
        "_gptoss_run_experts_grouped_mm_dynamic"
        in gpt_oss_moe._run_experts_grouped_mm.__qualname__
    )
    if already_patched:
        return

    # With EP the token count per expert varies every step.  Raise the cache
    # limit so dynamo doesn't bail out before mark_dynamic takes full effect.
    torch._dynamo.config.cache_size_limit = 64  # pyrefly: ignore [bad-assignment]

    gpt_oss_moe._run_experts_grouped_mm = torch.compile(
        gpt_oss_moe._run_experts_grouped_mm,
        backend=compile_config.backend,
        fullgraph=True,
    )
    compiled_fn = gpt_oss_moe._run_experts_grouped_mm

    def _gptoss_run_experts_grouped_mm_dynamic(
        mlp1_weight: torch.Tensor,
        mlp1_bias: torch.Tensor,
        mlp2_weight: torch.Tensor,
        mlp2_bias: torch.Tensor,
        swiglu_limit: float,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        tp_degree: int = 1,
    ) -> torch.Tensor:
        # A rank that received 0 tokens after DeepEP dispatch must not enter
        # the grouped-MM path: the FP8 CUTLASS backward would produce a
        # [hidden_dim, 0]-shaped gradient that the kernel cannot handle.
        if x.shape[0] == 0:
            return x.new_zeros(0, x.shape[-1])
        # Tell dynamo the token count is data-dependent so it doesn't
        # re-specialize (and recompile) for each distinct value.
        torch._dynamo.mark_dynamic(x, 0)
        return compiled_fn(
            mlp1_weight,
            mlp1_bias,
            mlp2_weight,
            mlp2_bias,
            swiglu_limit,
            x,
            num_tokens_per_expert,
            tp_degree,
        )

    gpt_oss_moe._run_experts_grouped_mm = _gptoss_run_experts_grouped_mm_dynamic
