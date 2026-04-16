# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch.nn as nn
from torch.distributed.tensor import DeviceMesh, distribute_tensor, DTensor, Replicate, Shard

from torchtitan.distributed.expert_parallel import (
    DeepEPExpertParallel,
    ExpertTensorParallel,
    TensorParallel,
)


# implementation of Tensor Parallel for the GroupedExperts in MoE
class GptossTensorParallel(TensorParallel):
    def _partition_fn(self, name, module, device_mesh):
        module.register_parameter(
            "mlp1_weight",
            nn.Parameter(
                distribute_tensor(module.mlp1_weight, device_mesh, [Shard(1)])
            ),
        )  # Column-wise sharding
        module.register_parameter(
            "mlp1_bias",
            nn.Parameter(distribute_tensor(module.mlp1_bias, device_mesh, [Shard(1)])),
        )  # Column-wise sharding
        module.register_parameter(
            "mlp2_weight",
            nn.Parameter(
                distribute_tensor(module.mlp2_weight, device_mesh, [Shard(2)])
            ),
        )  # Row-wise sharding
        module.register_parameter(
            "mlp2_bias",
            nn.Parameter(
                distribute_tensor(module.mlp2_bias, device_mesh, [Replicate()])
            ),
        )  # Replicate


# This class is for dp2ep with TP (without TP we can just use GptossExpertParallel)
class GptossExpertTensorParallel(ExpertTensorParallel):
    def _partition_fn(self, name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        mod.register_parameter(
            "mlp1_weight",
            nn.Parameter(
                # pyrefly: ignore [bad-argument-type]
                distribute_tensor(mod.mlp1_weight, device_mesh, [Shard(0), Shard(1)])
            ),
        )  # Column-wise sharding
        mod.register_parameter(
            "mlp1_bias",
            nn.Parameter(
                # pyrefly: ignore [bad-argument-type]
                distribute_tensor(mod.mlp1_bias, device_mesh, [Shard(0), Shard(1)])
            ),
        )  # Column-wise sharding
        mod.register_parameter(
            "mlp2_weight",
            nn.Parameter(
                # pyrefly: ignore [bad-argument-type]
                distribute_tensor(mod.mlp2_weight, device_mesh, [Shard(0), Shard(2)])
            ),
        )  # Row-wise sharding
        mod.register_parameter(
            "mlp2_bias",
            nn.Parameter(
                # pyrefly: ignore [bad-argument-type]
                distribute_tensor(mod.mlp2_bias, device_mesh, [Shard(0), Replicate()])
            ),
        )  # Replicate


class GptossDeepEPExpertParallel(DeepEPExpertParallel):
    """DeepEPExpertParallel adapted for GptOssGroupedExperts.

    GptOssGroupedExperts uses mlp1_weight / mlp2_weight instead of w1 / w2 / w3.
    This subclass overrides _token_dispatch to derive num_local_experts from
    mlp1_weight rather than w1.  All other dispatch/combine logic is inherited
    from DeepEPExpertParallel unchanged.
    """

    def _token_dispatch(self, mod: nn.Module, inputs: tuple, device_mesh: DeviceMesh):
        hidden_states, _, selected_experts_indices, top_scores, num_experts = inputs
        # GptOssGroupedExperts uses mlp1_weight; DeepEPExpertParallel uses w1.
        if isinstance(mod.mlp1_weight, DTensor):
            num_local_experts = mod.mlp1_weight.to_local().shape[0]
        else:
            num_local_experts = mod.mlp1_weight.shape[0]
        ep_group = device_mesh.get_group()

        if self.comm_backend == "hybridep":
            from torchtitan.distributed.deepep.hybridep import dispatch_tokens

            hidden_states, tokens_per_expert, self._state = dispatch_tokens(
                hidden_states,
                selected_experts_indices,
                top_scores,
                num_local_experts,
                num_experts,
                ep_group,
                score_before_experts=self.score_before_experts,
                non_blocking_expert_capacity_factor=self.hybridep_non_blocking_expert_capacity_factor,
                pad_multiple=self.pad_multiple,
            )
        else:
            from torchtitan.distributed.deepep.deepep import dispatch_tokens

            hidden_states, tokens_per_expert, self._state = dispatch_tokens(
                hidden_states,
                selected_experts_indices,
                top_scores,
                num_local_experts,
                num_experts,
                ep_group,
                score_before_experts=self.score_before_experts,
            )

        return hidden_states, tokens_per_expert
