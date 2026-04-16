# Branch Changes: `trying_deepep`

Summary of all code changes relative to upstream `main`. Organized by file.

---

## New Files

### `submit_20b.sh`
Launch script for 8-GPU GPT-OSS-20B training via `torchrun`. Runs
`torchtitan.train` with the `gpt_oss` module. Currently configured with
`gpt_oss_20b_mixed_precision_reduce_bf16`.

### `CHANGES_OLD.md` (formerly `OPTIMIZATIONS.md`)
Experiment log from the FP8 optimization work.

---

## Modified Files

### `torchtitan/components/quantization/float8.py`

**Fix tyro Config class name collision.**

- Added `Float8LinearConverter.Config.__name__ = Float8LinearConverter.Config.__qualname__`
- Added `Float8GroupedMMConverter.Config.__name__ = Float8GroupedMMConverter.Config.__qualname__`

Tyro uses `__name__` (not `__qualname__`) to generate union subcommand
discriminants. Both inner `Config` classes shared `__name__ = "Config"`,
colliding when both appeared together in a converters list. Setting `__name__`
to `__qualname__` makes each class uniquely named (`Float8LinearConverter.Config`
vs `Float8GroupedMMConverter.Config`).

---

### `torchtitan/config/configs.py`

**Allow BF16 reductions in mixed precision.**

- `mixed_precision_reduce: Literal["float32"]` → `Literal["float32", "bfloat16"]`

Enables `mixed_precision_reduce="bfloat16"` as a training config option.

---

### `torchtitan/distributed/compile.py`

**Fix MoE compilation for sparse models (from clowman PR #2781).**

Inside `apply_compile_sparse`, in the `experts` child branch:

- Added `torch.compiler.disable(submod)` — explicitly disables compilation on
  `GroupedExperts`. Previously the loop just `continue`'d without disabling,
  leaving the module in an ambiguous compiled state.
- Added `moe.compile(backend=compile_config.backend, fullgraph=False)` —
  compiles the outer MoE wrapper with `fullgraph=False` to allow graph breaks
  around the FSDP `GroupedExperts` hooks while still compiling the surrounding
  router, reorderer, and shared-expert subgraphs.

---

### `torchtitan/models/gpt_oss/__init__.py`

**20B model config: `score_before_experts`.**

- `score_before_experts=False` in `_20b()` (unchanged from upstream; a
  transient commit set it to `True` but the working tree reverts it).

---

### `torchtitan/models/gpt_oss/moe.py`

**Add FP8 expert computation path and `GptOssDeepEPMoE`.**

#### New: `_get_fp8_wrapper_type()` helper
Lazy-cached importer for `Float8TrainingWeightWrapperTensor` from
`torchao.prototype.moe_training.tensor`. Falls back to `type(None)` if torchao
is not installed, so BF16 training is unaffected.

#### Rewritten: `_run_experts_grouped_mm` — split into two branches

**FP8 branch** (`if isinstance(mlp1_weight, _get_fp8_wrapper_type())`):

`torch._scaled_grouped_mm` (used by the FP8 CUTLASS path via
`Float8TrainingWeightWrapperTensor.__torch_function__`) requires each expert's
token count to be divisible by 16. The original BF16 code had no such
constraint. The FP8 branch handles this by:

1. Computing padded-per-expert counts (ceil to multiple of 16).
2. Building on-device scatter indices via `torch.searchsorted` — no D2H syncs.
3. Scattering activations into an aligned padded buffer with `index_put`
   (backward is a gather — correct gradient placement).
4. Running both GEMMs with `torch._grouped_mm` against the
   `Float8TrainingWeightWrapperTensor` weights directly. `__torch_function__`
   intercepts and dispatches to torchao's
   `_to_fp8_rowwise_then_scaled_grouped_mm` (CUTLASS), with the activations
   already pre-padded (`pad_token_groups_for_grouped_mm=False`).
5. Gathering real (and slack) tokens back to the original layout. Backward of
   `h[dst_idx]` is a scatter — correct gradient.

**BF16 branch** (`else`): Original logic, preserved unchanged.

#### New: `GptOssDeepEPMoE`
Subclass of `GptOssMoE` with no new parameters. Overrides `forward()` to
delegate to `DeepEPMoE.forward(self, x)`, replacing all-to-all collectives with
DeepEP RDMA kernels for expert dispatch and combine. Instantiated by replacing
`__class__` on an existing `GptOssMoE` instance (preserves all weights and
buffers without reinitializing).

---

### `torchtitan/models/gpt_oss/expert_parallel.py`

**Add `GptossDeepEPExpertParallel`.**

New imports: `DTensor`, `DeepEPExpertParallel`.

#### New: `GptossDeepEPExpertParallel` (subclass of `DeepEPExpertParallel`)
Overrides `_token_dispatch` to read `mod.mlp1_weight` instead of `mod.w1`.
`GptOssGroupedExperts` uses `mlp1_weight / mlp2_weight` parameter names while
the base `GroupedExperts` uses `w1 / w2 / w3`; this subclass bridges the
difference without duplicating any dispatch or combine logic.

Dispatches via:
- `torchtitan.distributed.deepep.deepep.dispatch_tokens` when `comm_backend="deepep"`
- `torchtitan.distributed.deepep.hybridep.dispatch_tokens` when `comm_backend="hybridep"`

---

### `torchtitan/models/gpt_oss/parallelize.py`

**Wire DeepEP into GPT-OSS parallelization.**

#### `parallelize_gptoss`
- Passes `comm_backend=parallelism.expert_parallel_comm_backend` to `apply_moe_ep_tp`.
- After sparse compile, calls `_apply_compile_gptoss_ep(compile_config)` when EP is enabled.

#### `apply_moe_ep_tp`
- New `use_deepep = comm_backend in ("deepep", "hybridep")` flag.
- When DeepEP is active and `ep_mesh` is set:
  - Replaces `moe.__class__ = GptOssDeepEPMoE` (class swap preserves weights).
  - Sets `moe.reorderer = None` (DeepEP handles token routing internally; the
    `TokenReorderer` is unused and must be removed so it is not distributed).
- Skips `ReordererSequenceParallel` injection when DeepEP is active.
- Uses `GptossDeepEPExpertParallel` as the `parallelize_plan` for experts
  instead of `TorchAOExpertParallel` or the default all-to-all plan.

#### New: `_apply_compile_gptoss_ep`
Compiles `gpt_oss.moe._run_experts_grouped_mm` with `fullgraph=True` and wraps
it with a dynamic-shape-aware outer function:

- **Zero-token guard:** ranks that receive 0 tokens after DeepEP dispatch return
  `x.new_zeros(0, x.shape[-1])` immediately. The FP8 CUTLASS backward produces
  a `[hidden_dim, 0]`-shaped gradient that the kernel cannot handle.
- **`mark_dynamic(x, 0)`:** tells dynamo that the token-count dimension is
  data-dependent so it compiles one dynamic graph rather than re-specializing
  on each distinct value every step.
- **`cache_size_limit = 64`:** raised from the default (8–16) to let
  `mark_dynamic` take full effect before dynamo gives up.
- Double-patch guard: checks `__qualname__` to avoid re-patching across
  pipeline-parallel stages (function is called once per PP stage).

---

### `torchtitan/models/gpt_oss/config_registry.py`

**Add 9 training configs for 20B FP8 experiments.**

| Config | Description | Result |
|--------|-------------|--------|
| `gpt_oss_20b_fp8` | Full FP8, no compile | ~49 TFLOPs (regression reference) |
| `gpt_oss_20b_compile` | Full FP8 + compile | ~159 TFLOPs (regression reference) |
| `gpt_oss_20b_bf16_compile` | BF16 + compile upper bound | 194.1 TFLOPs, 58.2 GiB |
| `gpt_oss_20b_dense_fp8_only` | Dense-only FP8 + compile | **195.7 TFLOPs, 57.15 GiB** |
| `gpt_oss_20b_ga2` | Dense-only FP8, global batch 64 | — |
| `gpt_oss_20b_mixed_precision_reduce_bf16` | Dense FP8 + BF16 reductions | — |
| `gpt_oss_20b_ep8` | Dense FP8 + DeepEP EP=8 | ~347 TFLOPs |
| `gpt_oss_20b_profile_bf16` | BF16 + compile, 30 steps, profiler on | — |
| `gpt_oss_20b_profile_fp8` | Dense FP8 + compile, 30 steps, profiler on | — |

New imports in the registry: `CompileConfig`, `ProfilingConfig`,
`Float8LinearConverter`, `Float8GroupedMMConverter`, `ModelConvertersContainer`.

---

### `torchtitan/models/common/moe.py`

**Eliminate duplicate `x.bfloat16()` cast in `_run_experts_grouped_mm`.**

- Added `x_bf16 = x.bfloat16()` once after the `offsets` computation.
- Replaced both `x.bfloat16()` calls (lines 63 and 66) with `x_bf16`.

`x` does not change between the two grouped-MM calls; the second cast was a
redundant no-op. Measured result: **+1 TFLOPs** (346 → 347) on models using
the common MoE path. GPT-OSS has its own `_run_experts_grouped_mm` and is
unaffected.

---

## TFLOPs Progression (8×H100, GPT-OSS-20B)

| Config | TFLOPs | Notes |
|--------|--------|-------|
| BF16 + compile (baseline) | 194–198 | upstream starting point |
| FP8 no compile | ~49 | 4× regression |
| Full FP8 + compile | ~159 | graph breaks in expert path |
| Dense-only FP8 + compile | 195.7 | beats BF16 baseline |
| + DeepEP EP=8 | ~346 | 1.77× over dense |
| + `x_bf16` cache (common moe) | ~347 | +1 TFLOPs |
