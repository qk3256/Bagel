# Remaining single-A100 optimization experiments

Date: 2026-11-07  
Hardware: one A100 40 GB  
Dataset: real COCO Caption + VQA v2 + MagicBrush profile subset

## Final results

Two final configurations are reported separately because PyTorch fused AdamW uses a different floating-point update order.

| Configuration | Run medians | Median of runs | Peak allocated | Numerical status |
| --- | --- | ---: | ---: | --- |
| Original real-data non-blocking baseline | 826.45 / 819.67 / 823.06 ms | 823.06 ms | 32168.88 MiB | reference |
| Strict-equivalence synchronization changes | 816.34 / 819.67 / 840.34 ms | 819.67 ms | 32168.88 MiB | baseline loss sequence exact |
| Strict changes + optional fused AdamW | 760.47 / 772.21 / 763.40 ms | 763.40 ms | 27384.89 MiB | deterministic, but not step-exact to foreach AdamW |

The strict changes are 0.41% faster at the median, which is inside run-to-run noise. Their defensible benefit is removal of synchronization and copies. Optional fused AdamW is 6.87% faster than the final strict configuration and reduces peak allocation by 4783.99 MiB (14.9%).

## Strict-equivalence changes

### CPU batch metadata

Commit `8d6ea33` computes two values before H2D:

- `vit_max_seqlen`, replacing `torch.max(vit_token_seqlens).item()` in forward.
- Integer `mse_target_indexes`, replacing GPU boolean indexing and its internal `nonzero` synchronization.

The H2D payload gains one small pinned index tensor, while the model forward loses three synchronizations and three internal copies per step. Four-step MSE/CE losses exactly match the reference.

### Single-rank metrics stay on CPU

Commit `805a773` skips GPU scalar construction and no-op collectives for token, sequence-square, sample, loss-normalization, and memory statistics when `world_size == 1`. Multi-rank behavior is preserved. Four-step losses and logged metric values exactly match the reference.

### Cached timestep frequencies

Commit `2ceda53` creates the FP32 sinusoidal frequency table once and lazily caches an exact CUDA copy outside FSDP buffer casting. The table is bit-identical to the previous per-step expression. VAE token assembly changes were:

| Metric / step | Before | After |
| --- | ---: | ---: |
| CPU range | 11.92 ms | 3.89 ms |
| Synchronizations | 1 | 0 |
| Copies | 1 | 0 |

The first attempt registered the table as an FSDP buffer. FSDP cast it to BF16, changing later losses, so that implementation was rejected. The retained lazy FP32 cache restores the exact baseline loss sequence.

### Combined Trace

Relative to the original real-data Trace, the retained code reduces model-forward synchronizations from 12 to 8 per step. The four removed sites are flex-mask CPU-to-device metadata, ViT `max().item()`, MSE boolean indexing, and timestep-frequency H2D. Model-forward copies fall from 21 to 17 per step.

## Optional fused AdamW

Commit `e93b20c` adds `--fused_adamw True` using PyTorch 2.5's built-in CUDA implementation. Commit `b13b682` ensures that the disabled state passes `fused=None`, preserving PyTorch's original automatic foreach selection.

| Optimizer range / step | Foreach baseline | Fused AdamW | Change |
| --- | ---: | ---: | ---: |
| CPU time | 8.60 ms | 1.95 ms | -77.4% |
| CUDA kernel time | 78.52 ms | 30.50 ms | -61.2% |
| Kernel count | 469 | 68 | -85.5% |

The first forward/loss is identical because no optimizer update has occurred. Later losses differ because fused and foreach AdamW round BF16/FP32 intermediate expressions differently. A focused four-update optimizer comparison measured maximum parameter differences of `1.22e-4` for BF16 and `2.38e-7` for FP32, with no NaN or Inf. Three fused runs reproduce the same loss sequence, but exact equivalence to foreach is not claimed. The option therefore remains disabled by default pending a longer convergence/quality run.

## Fusion feasibility audit

| Candidate | Result | Decision |
| --- | --- | --- |
| PyTorch native RMSNorm | MoT CPU forward improved, but step-0 CE changed from 17.8056 to 17.7083; backward kernels increased from 12680 to 13068/step | rejected |
| FlashAttention Triton LayerNorm | 1328x1152 forward+backward microbenchmark: 1.774 ms versus 0.828 ms for current native LayerNorm | current implementation retained |
| FlashAttention Triton RoPE | 1777x14x64 forward+backward: 1.398 ms versus 1.071 ms reference | rejected |
| `torch.compile` RoPE | 1.381 ms versus 1.104 ms eager reference and not bit-exact | rejected |
| FlashAttention FusedMLP | package exposes the Python class but `fused_dense_lib` is absent | unavailable in current environment |
| Separate-stream H2D prefetch | baseline H2D is only 0.77 ms CPU / 0.19 ms memcpy and DataLoader wait is 0.70 ms | not implemented; complexity exceeds measured upper bound |

PyTorch `nn.LayerNorm` already dispatches to `native_layer_norm`; installing another implementation is not automatically a fusion win. RMSNorm requires a dedicated kernel that preserves BAGEL's FP32 accumulation and FSDP/autocast dtype behavior before it can replace the reference path.

## Reproduction

```bash
# Strict-equivalence final configuration (default optimizer)
BAGEL_DATA_ROOT=/home/why/BAGEL/data/bagel_real_profile \
DATASET_CONFIG=data/configs/single_a100_real_profile.yaml \
TOTAL_STEPS=12 RUN_NAME=real-strict-final \
bash scripts/train_single_a100_smoke.sh --save_final_checkpoint False

# Optional performance configuration
BAGEL_DATA_ROOT=/home/why/BAGEL/data/bagel_real_profile \
DATASET_CONFIG=data/configs/single_a100_real_profile.yaml \
TOTAL_STEPS=12 RUN_NAME=real-single-card-final \
bash scripts/train_single_a100_smoke.sh \
  --fused_adamw True --save_final_checkpoint False
```
