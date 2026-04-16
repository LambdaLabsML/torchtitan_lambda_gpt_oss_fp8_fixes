# GPT-OSS-20B Training Optimizations

Changes relative to upstream `pytorch/torchtitan` (`e24e465c`).
Measured on 8× H100 80 GB HBM3, single node, `seq_len=8192`, `local_batch_size=1`,
full activation checkpointing.

## Performance Summary

| Configuration | TFLOPs | Memory (GiB) |
|---|---|---|
| BF16, no compile (upstream baseline) | 179.7 | 70.4 |
| BF16 + compile | 194.1 | 58.2 |
| Full FP8 + compile (both dense and expert layers) | 159.3 | 73.1 |
| **Dense-only FP8 + compile (recommended)** | **195.7** | **57.2** |

Dense-only FP8 + compile beats BF16 + compile on both throughput (+0.8%) and
memory (-1.7%). The recommended training config is `gpt_oss_20b_dense_fp8_only`.

---

## Bug Fixes

### 1. Tyro discriminant collision between FP8 converter configs (`float8.py`)

**File:** `torchtitan/components/quantization/float8.py`

`Float8LinearConverter.Config` and `Float8GroupedMMConverter.Config` are both inner
classes named `Config`. Tyro uses `__name__` (not `__qualname__`) to generate union
subcommand discriminants, so both classes resolved to the same name `'Config'` and
collided when both appeared together in the converters list.

**Fix:** Set `__name__ = __qualname__` on both classes immediately after their
definitions, making their names unique (`'Float8LinearConverter.Config'` and
`'Float8GroupedMMConverter.Config'`).

```python
Float8LinearConverter.Config.__name__ = Float8LinearConverter.Config.__qualname__
Float8GroupedMMConverter.Config.__name__ = Float8GroupedMMConverter.Config.__qualname__
```

**Impact:** Without this fix, using both converters together in a config raises a
tyro collision error at startup.

---

### 2. Memory leak in custom FP8 expert GEMM path (`moe.py`)

**File:** `torchtitan/models/gpt_oss/moe.py`

The previous implementation included `_FusedFP8GroupedExpertsMM`, a custom
`torch.autograd.Function` that used Triton kernels for FP8 expert GEMMs. It
maintained a module-level cache (`_fp8_weight_forward_cache`) keyed by `id(weight)`
to avoid re-quantizing expert weights during activation checkpointing (AC) recompute.

**Bug:** With FSDP2, `fully_shard()` allocates new weight buffers for each forward
pass (all-gather + free cycle). The `id()` of the weight tensor therefore changes
between the initial forward pass (when the cache entry was written) and the AC
recompute forward pass (when the cache was expected to be consumed via `.pop()`).
Cache entries were never matched and consumed, accumulating FP8 weight copies
indefinitely — approximately 35 MB per expert group per step, totalling ~14 GiB
of leaked memory over a training run.

**Fix:** Removed `_FusedFP8GroupedExpertsMM` and `_fp8_weight_forward_cache`
entirely. See Optimization 1 below for the replacement.

**Impact:** ~14 GiB memory reduction when using `Float8GroupedMMConverter`.

---

### 3. FP8 expert path caused `torch.compile` graph breaks (`moe.py`)

**File:** `torchtitan/models/gpt_oss/moe.py`

`_FusedFP8GroupedExpertsMM` called Triton kernels (`triton_fp8_rowwise_3d_transpose_rhs`,
`triton_fp8_rowwise_2d_scale_and_cast`, `triton_fp8_per_group_colwise_scales_dual`)
directly via a custom autograd Function. `torch.compile` / `torch.dynamo` could not
trace through this path, triggering graph breaks at every expert GEMM call.

Graph breaks defeat the main benefit of compile (kernel fusion and operator
scheduling), which is why full FP8 + compile ran at only 159 TFLOPs — worse than
BF16 + compile at 194 TFLOPs.

**Fix:** Removed the custom Triton path. See Optimization 1 below.

**Impact:** Eliminated graph breaks in the expert GEMM path, allowing compile to
produce a fully fused execution graph.

---

## Optimizations

### 1. FP8 expert GEMM via `__torch_function__` dispatch (`moe.py`)

**File:** `torchtitan/models/gpt_oss/moe.py`

Replaced `_FusedFP8GroupedExpertsMM` with a clean dispatch path that routes through
torchao's existing CUTLASS FP8 kernel via `__torch_function__`.

When expert weights are wrapped as `Float8TrainingWeightWrapperTensor` (applied by
`Float8GroupedMMConverter`), calling `torch._grouped_mm` on them triggers the
`__torch_function__` protocol, which torchao intercepts and dispatches to
`_to_fp8_rowwise_then_scaled_grouped_mm` → `torch._scaled_grouped_mm` (CUTLASS).

`torch._scaled_grouped_mm` requires each expert's token count to be divisible by
16. Since MoE routing produces uneven per-expert token counts, activations are
padded on-device before the GEMM and gathered back afterward:

1. Compute padded per-expert counts (round up each to nearest multiple of 16).
2. Build a token→destination index map using `torch.searchsorted` — all on device,
   no host-device syncs.
3. Scatter activations into a zero-padded buffer via `index_put`.
4. Call `torch._grouped_mm` on the padded buffer with padded offsets.
5. Gather real tokens back from the padded output via integer indexing.

The BF16 path is unchanged and continues to use the existing tail-slack
`repeat_interleave` trick (no padding required).

Added `_get_fp8_wrapper_type()` as a lazy-import helper that caches the
`Float8TrainingWeightWrapperTensor` type (or `type(None)` if torchao is not
installed), avoiding repeated import overhead on every forward call.

**Impact:** Eliminates graph breaks, removes the memory-leaking cache, and produces
correct gradients through differentiable scatter/gather (`index_put` backward is a
gather; integer indexing backward is a scatter).

---

### 2. Dense-only FP8 quantization is faster than full FP8 (`config_registry.py`)

**File:** `torchtitan/models/gpt_oss/config_registry.py`

Investigation showed that applying `Float8GroupedMMConverter` to expert weights
(which represent ~94% of total parameters in GPT-OSS-20B) causes compile graph
breaks and net throughput regression, regardless of which FP8 GEMM implementation
is used. The quantization overhead for the expert path (scale computation,
precision casting, alignment padding) outweighs any GEMM speedup.

The optimal configuration applies `Float8LinearConverter` to dense layers only
(attention projections, dense MLPs) and leaves expert weights in BF16:

- `filter_fqns=["router.gate", "output"]`: the router gate runs inside
  `torch.autocast(..., dtype=torch.float32)` for routing stability, and
  `Float8TrainingTensor` only supports float16/bfloat16 autocast — including it
  causes a runtime error. The output/lm_head projection is also excluded: including
  it in FP8 all-gather adds ~8 GiB memory overhead and reduces TFLOPs.
- `enable_fsdp_float8_all_gather=True`: FSDP all-gather communication is performed
  in FP8 rather than BF16, halving all-gather bandwidth for quantized layers.
- `precompute_float8_dynamic_scale_for_fsdp=True`: FP8 scales are precomputed after
  the optimizer step, avoiding scale computation on the critical path during
  the next forward pass.

**Result:** 195.7 TFLOPs / 57.2 GiB — beats BF16 + compile on both metrics.

---

## Named Configs Added (`config_registry.py`)

All configs use `seq_len=8192`, full activation checkpointing, cosine LR schedule
with 2000 warmup steps.

| Config name | Description | TFLOPs | Memory (GiB) |
|---|---|---|---|
| `gpt_oss_20b_dense_fp8_only` | **Recommended.** Dense-only FP8 + compile. | 195.7 | 57.2 |
| `gpt_oss_20b_bf16_compile` | BF16 + compile reference baseline. | 194.1 | 58.2 |
| `gpt_oss_20b_compile` | Full FP8 (dense + expert) + compile. Reference showing regression before fix. | 159.3 | 73.1 |
| `gpt_oss_20b_fp8` | Full FP8 without compile. Slowest — shows compile is required for FP8. | ~49 | ~77 |
| `gpt_oss_20b_profile_bf16` | BF16 + compile, 30 steps, profiling enabled. | — | — |
| `gpt_oss_20b_profile_fp8` | Dense-only FP8 + compile, 30 steps, profiling enabled. | — | — |
