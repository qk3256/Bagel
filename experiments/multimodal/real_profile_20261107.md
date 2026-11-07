# Real-data multimodal profile

Date: 2026-11-07  
Hardware: one A100 40 GB  
Data: 256 COCO Caption, 256 VQA v2, 256 MagicBrush records

## Baseline critical path

The profiler captured two active training steps. Profiler wall time is inflated, so range timings identify hot paths while profiler-free repetitions provide end-to-end Step Time.

| Range | CPU time / step | CUDA kernel time / step | Kernels / step | Critical-path interpretation |
| --- | ---: | ---: | ---: | --- |
| Full profiled step | 1347.25 ms | - | - | Profiler-inflated wall time |
| Backward | 776.93 ms | 118.65 ms | 12632 | Dominant CPU launch/autograd path |
| Model forward | 413.45 ms | 40.82 ms | about 4081 | Critical |
| MoT transformer | 281.56 ms | 29.49 ms | 3331 | Critical |
| MoT MLP experts | 90.86 ms | about 21.0 ms | about 1488 | Main investigated routing hotspot |
| ViT path | 113.46 ms | 9.22 ms | 619 | Critical for understanding samples |
| VAE encode | 15.53 ms | 16.42 ms | about 381 | Critical for edit/generation samples |
| Attention mask | 8.46 ms | 0.19 ms | 52 | CPU/synchronization hotspot, but small globally |
| VAE token assembly | 5.50 ms | about 0.14 ms | about 30 | Minor |
| H2D | 0.77 ms | 0.19 ms memcpy | 15 copies | Not a dominant path |
| DataLoader next | 0.70 ms | 0 | 0 | Hidden effectively by prefetch |
| Text token assembly | 0.34 ms | about 0.02 ms | 3 | Not worth optimizing now |
| MoT index build | about 0.09 ms | negligible | 1 | Not a hotspot |

Peak allocated memory was 32169.01 MiB. The two token distributions are recorded in `experiments/data/real_profile_dataset.md`.

## Correct `IndexPutBackward0` accounting

The exact event name occurred 778 times and its same-name interval union was 469.40 ms across two active steps. Therefore the correct normalization is **234.70 ms per profiled step**, not 470 ms per step.

The autograd engine wrapper occurred 778 times for 476.35 ms and encloses the exact node events. It must not be added to the exact-node total. Same-name union equals inclusive total for the exact nodes, showing that those 778 exact events do not overlap each other.

Forward-sequence attribution of the 469.40 ms exact-node total was:

| Originating forward range | Calls | Two-step inclusive time |
| --- | ---: | ---: |
| MoT MLP experts | 98 | 415.72 ms |
| MoT attention | 578 | 45.56 ms |
| MoT route input | 92 | 6.54 ms |
| Other marked paths | 10 | 1.58 ms |

This establishes MoT MLP output placement as the source of most `IndexPutBackward0` time. It does not establish that the time can be removed: the experiments below show that equivalent gradient gathering transfers the same cost to another node.

## Real-data H2D A/B

Three order-balanced, profiler-free 12-step runs were compared after discarding warm-up steps. Loss sequences and peak memory were identical.

| Pair | Blocking median | Non-blocking median | Non-blocking minus blocking |
| ---: | ---: | ---: | ---: |
| 1 | 809.73 ms | 826.45 ms | +16.72 ms |
| 2 | 790.53 ms | 819.67 ms | +29.15 ms |
| 3 | 829.89 ms | 823.06 ms | -6.83 ms |
| Median of runs | 809.73 ms | 823.06 ms | +13.33 ms (+1.65%) |

`non_blocking=True` removes direct H2D synchronization but gives no confirmed end-to-end gain for this real packed workload. This differs from the Smoke result (+3.1%) and is why the two datasets are reported separately.

## MoT optimization experiments

All variants reproduced the baseline loss sequence and approximately 32169 MiB peak allocation.

1. One `index_copy` merge per layer reduced `IndexPutBackward0` from 778 calls / 469.40 ms to 682 / 55.36 ms, but introduced 48 `IndexCopyBackward0` calls / 399.43 ms. MoT MLP forward CPU time increased from 181.71 to 191.61 ms across two steps. Rejected.
2. Concatenate expert outputs and restore token order with one inverse permutation reduced `IndexPutBackward0` to 682 / 60.09 ms, but `IndexBackward0` became 632 calls / 530.39 ms. Backward increased from 776.93 to 797.46 ms/step. Rejected.
3. A custom merge autograd function reduced `IndexPutBackward0` to 682 / 53.77 ms, but its own 48 `_MergeExpertOutputsBackward` calls cost 396.62 ms. Backward was 766.40 ms/step, within run-to-run Trace noise. Rejected.

Commits `26f7521` and `c20a403` preserve the first two independent experiments. The final code restores the original MoT placement behavior; no misleading routing rewrite remains enabled.

## Selected optimization: device-native flex-mask metadata

`create_sparse_mask` previously built three CPU tensors during every forward and then copied them to CUDA. Commit `faf1baa` constructs the integer metadata directly on the target device.

| Attention-mask metric, per step | Baseline | Optimized | Change |
| --- | ---: | ---: | ---: |
| CPU time | 8.46 ms | 8.01 ms | -0.45 ms (-5.3%) |
| Synchronizations | 9 | 8 | -1 |
| Synchronization time | 1.32 ms | 0.08 ms | -1.24 ms |
| Memcpy calls | 17 | 16 | -1 |
| CUDA kernel time | 0.19 ms | 0.20 ms | effectively unchanged |

The optimized four-step loss sequence exactly matched baseline: `(1.5764,17.8056)`, `(1.8074,25.8162)`, `(1.5007,18.0254)`, `(2.1314,7.7496)`. Peak allocation remained 32169.01 MiB.

Three profiler-free optimized runs produced medians of 873.38, 787.40, and 809.73 ms; their median is 809.73 ms versus the prior real-data non-blocking baseline median of 823.06 ms (1.62% faster). The first optimized run is an outlier and the expected local saving is under 1 ms, so the end-to-end difference is **indicative, not statistically attributable** to this change. The defensible result is the direct removal of one synchronization and one copy per step.

## Reproduction

```bash
# Baseline real-data trace (checkout 4e069c7)
BAGEL_DATA_ROOT=/home/why/BAGEL/data/bagel_real_profile \
DATASET_CONFIG=data/configs/single_a100_real_profile.yaml \
TOTAL_STEPS=4 RUN_NAME=real-multimodal-baseline \
bash scripts/train_single_a100_smoke.sh \
  --profile_dir /home/why/BAGEL/traces/real_multimodal_baseline \
  --profile_wait 1 --profile_warmup 1 --profile_active 2 \
  --save_final_checkpoint False

python scripts/analyze_multimodal_trace.py \
  /home/why/BAGEL/traces/real_multimodal_baseline/*.pt.trace.json --top-ops 35

# Final optimized trace (checkout faf1baa)
BAGEL_DATA_ROOT=/home/why/BAGEL/data/bagel_real_profile \
DATASET_CONFIG=data/configs/single_a100_real_profile.yaml \
TOTAL_STEPS=4 RUN_NAME=real-mask-device-native \
bash scripts/train_single_a100_smoke.sh \
  --profile_dir /home/why/BAGEL/traces/real_mask_device_native \
  --profile_wait 1 --profile_warmup 1 --profile_active 2 \
  --save_final_checkpoint False
```
