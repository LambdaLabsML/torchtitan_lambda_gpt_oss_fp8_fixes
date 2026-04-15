# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.tools.profiling import ProfilingConfig
from torchtitan.trainer import Trainer

from torchtitan.components.quantization.float8 import (
    Float8GroupedMMConverter,
    Float8LinearConverter,
)
from torchtitan.protocols.model_converter import ModelConvertersContainer

from . import model_registry


def gpt_oss_debugmodel() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("debugmodel"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
        ),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="none",
        ),
        validator=Validator.Config(
            freq=5,
            steps=10,
        ),
    )


def gpt_oss_20b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/gpt-oss-20b",
        model_spec=model_registry("20b"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=8192,
            steps=10000,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
    )


def gpt_oss_120b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/gpt-oss-120b",
        model_spec=model_registry("120b"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=8192,
            steps=10000,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
    )


def gpt_oss_20b_fp8() -> Trainer.Config:
    """Full FP8 (dense + expert layers) without compile.
    Reference config showing regression vs BF16: ~49 TFLOPs, 77 GiB.
    FP8 without compile is 3.6x slower — compile is required for FP8 to be beneficial."""
    return Trainer.Config(
        hf_assets_path="./assets/hf/gpt-oss-20b",
        model_spec=model_registry("20b"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        model_converters=ModelConvertersContainer.Config(
            converters=[
                Float8LinearConverter.Config(
                    enable_fsdp_float8_all_gather=True,
                    precompute_float8_dynamic_scale_for_fsdp=True,
                    # The MoE router gate runs inside torch.autocast(..., dtype=torch.float32)
                    # for routing stability. TorchAO's Float8TrainingTensor only supports
                    # float16/bfloat16 autocast, so we must exclude the gate from FP8.
                    filter_fqns=["router.gate", "output"],
                ),
                # GptOssGroupedExperts stores weights as raw nn.Parameter (not nn.Linear),
                # so Float8LinearConverter misses them entirely — they are 94% of total params.
                # Float8GroupedMMConverter wraps these via Float8TrainingWeightWrapperTensor,
                # which intercepts torch._grouped_mm via __torch_function__ and performs
                # dynamic FP8 row-wise quantization on grouped GEMM operations.
                Float8GroupedMMConverter.Config(fqns=["experts"]),
            ],
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=8192,
            steps=10000,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
    )


def gpt_oss_20b_compile() -> Trainer.Config:
    """Full FP8 (dense + expert layers) + compile.
    Reference config showing regression vs BF16+compile: ~159 TFLOPs, 73 GiB.
    Expert FP8 via Float8GroupedMMConverter causes graph breaks that hurt performance.
    Use gpt_oss_20b_dense_fp8_only instead for the best FP8 result."""
    return Trainer.Config(
        hf_assets_path="./assets/hf/gpt-oss-20b",
        model_spec=model_registry("20b"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        model_converters=ModelConvertersContainer.Config(
            converters=[
                Float8LinearConverter.Config(
                    enable_fsdp_float8_all_gather=True,
                    precompute_float8_dynamic_scale_for_fsdp=True,
                    # The MoE router gate runs inside torch.autocast(..., dtype=torch.float32)
                    # for routing stability. TorchAO's Float8TrainingTensor only supports
                    # float16/bfloat16 autocast, so we must exclude the gate from FP8.
                    filter_fqns=["router.gate", "output"],
                ),
                # GptOssGroupedExperts stores weights as raw nn.Parameter (not nn.Linear),
                # so Float8LinearConverter misses them entirely — they are 94% of total params.
                # Float8GroupedMMConverter wraps these via Float8TrainingWeightWrapperTensor,
                # which intercepts torch._grouped_mm via __torch_function__ and performs
                # dynamic FP8 row-wise quantization on grouped GEMM operations.
                Float8GroupedMMConverter.Config(fqns=["experts"]),
            ],
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=8192,
            steps=10000,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        compile=CompileConfig(enable=True, components=["model", "loss"]),
    )


def _profiling_config() -> ProfilingConfig:
    """Profile one step after skipping the first compile-warmup cycle."""
    return ProfilingConfig(
        enable_profiling=True,
        save_traces_folder="profile_traces",
        profile_freq=10,
        profiler_warmup=2,
        profiler_active=1,
        profiler_repeat=1,
        profiler_skip_first=1,
    )


def gpt_oss_20b_profile_bf16() -> Trainer.Config:
    """BF16 + compile baseline for profiling comparison."""
    return Trainer.Config(
        hf_assets_path="./assets/hf/gpt-oss-20b",
        model_spec=model_registry("20b"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=8192,
            steps=30,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(interval=1000),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        compile=CompileConfig(enable=True),
        profiling=_profiling_config(),
    )


def gpt_oss_20b_profile_fp8() -> Trainer.Config:
    """Dense-only FP8 + compile for profiling — matches gpt_oss_20b_dense_fp8_only but short."""
    return Trainer.Config(
        hf_assets_path="./assets/hf/gpt-oss-20b",
        model_spec=model_registry("20b"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        model_converters=ModelConvertersContainer.Config(
            converters=[
                Float8LinearConverter.Config(
                    enable_fsdp_float8_all_gather=True,
                    precompute_float8_dynamic_scale_for_fsdp=True,
                    filter_fqns=["router.gate", "output"],
                ),
            ],
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=8192,
            steps=30,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(interval=1000),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        compile=CompileConfig(enable=True),
        profiling=_profiling_config(),
    )


def gpt_oss_20b_bf16_compile() -> Trainer.Config:
    """BF16 + compile — upper bound reference for FP8+compile.
    Result: 194.1 TFLOPs, 58.2 GiB."""
    return Trainer.Config(
        hf_assets_path="./assets/hf/gpt-oss-20b",
        model_spec=model_registry("20b"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=8192,
            steps=10000,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        compile=CompileConfig(enable=True, components=["model", "loss"]),
    )


def gpt_oss_20b_dense_fp8_only() -> Trainer.Config:
    """Dense-layer-only FP8 + compile — the best FP8 configuration.

    Applies Float8LinearConverter to all linear layers (attention + dense MLP)
    except the router gate (float32 for routing stability) and the output projection
    (excluding output avoids memory overhead from FP8 all-gather on the large vocab
    weight). Expert weights (GptOssGroupedExperts) stay in BF16.

    Result: 195.7 TFLOPs, 57.15 GiB — beats BF16+compile (194.1 TFLOPs, 58.2 GiB)
    on both metrics. Adding Float8GroupedMMConverter for expert weights causes graph
    breaks that regress TFLOPs from 195.7 to ~159 and increase memory to ~73 GiB.
    """
    return Trainer.Config(
        hf_assets_path="./assets/hf/gpt-oss-20b",
        model_spec=model_registry("20b"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        model_converters=ModelConvertersContainer.Config(
            converters=[
                Float8LinearConverter.Config(
                    enable_fsdp_float8_all_gather=True,
                    precompute_float8_dynamic_scale_for_fsdp=True,
                    filter_fqns=["router.gate", "output"],
                ),
            ],
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=8192,
            steps=10000,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        compile=CompileConfig(enable=True, components=["model", "loss"]),
    )
