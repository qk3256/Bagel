# Single-A100 optimization phase summary

Date: 2026-11-07

## Smoke experiments: generic mechanisms

The controlled Smoke workload remains the reference for the original three-way tensor-placement experiment:

| Variant | Median Step Time | Relative result | Loss | Peak memory |
| --- | ---: | ---: | --- | ---: |
| Baseline blocking H2D | 806.45 ms | reference | identical | 32171.88 MiB |
| `non_blocking=True` | 781.44 ms | 3.1% faster | identical | 32171.88 MiB |
| GPU device-native mask tensor | 803.23 ms | 2.8% slower than non-blocking | identical | 32171.88 MiB |

These results isolate generic synchronization behavior but do not represent semantic multimodal data. Dataset construction details are in `experiments/data/smoke_dataset_audit.md`; the complete three-way interpretation is in `experiments/optimizations/three_way_comparison.md`.

## Real-data experiments: multimodal paths

The real subset contains 256 COCO Captions, 256 VQA v2, and 256 MagicBrush samples in BAGEL native loaders. It is semantic data and exercises understanding, QA, and editing paths in the same packed workload.

Main findings:

- MoT MLP output placement is responsible for 415.72 ms of the 469.40 ms exact `IndexPutBackward0` total over two profiled steps.
- 469.40 ms is a two-step total; the correct per-step number is 234.70 ms. The engine wrapper overlaps and is not additive.
- Three semantically equivalent MoT merge strategies merely transferred cost to `IndexCopyBackward0`, `IndexBackward0`, or a custom backward. None is retained.
- Token assembly, MoT index construction, DataLoader wait, and H2D are small relative to forward/backward. ViT, MoT, and backward are the actual critical paths.
- Real-data non-blocking H2D was 1.65% slower at the median of three balanced runs, so the Smoke gain cannot be generalized.
- The retained mask change removes one synchronization and one copy per step and reduces the measured mask range by 5.3%, with exact loss and memory parity. Its whole-step benefit is below reliable attribution at this sample size.

Detailed evidence is in `experiments/multimodal/real_profile_20261107.md`, and dataset provenance/format is in `experiments/data/real_profile_dataset.md`.

## Follow-up single-card optimization result

The remaining single-card audit is documented in `experiments/optimizations/single_card_remaining_20261107.md`.

- Strict-equivalence synchronization changes remove four model-forward synchronizations per step while preserving the complete loss sequence and 32168.88 MiB peak memory. Their three-run median is 819.67 ms versus 823.06 ms previously; the 0.41% difference is within noise.
- Optional PyTorch fused AdamW reduces the optimizer from 469 to 68 kernels/step, lowers its CUDA time from 78.52 to 30.50 ms/step, and produces a three-run median of 763.40 ms with 27384.89 MiB peak memory.
- Fused AdamW is deterministic but differs from foreach AdamW after the first update because of floating-point operation order. It remains opt-in until a longer convergence run is available.
- Native/Triton RMSNorm, Triton LayerNorm, Triton RoPE, and compiled RoPE were tested or microbenchmarked and rejected for numerical or performance reasons.

## Relevant commits

| Commit | Purpose | Final status |
| --- | --- | --- |
| `1995742` | Prepare compact real multimodal dataset and native loader config | retained |
| `1331ae7` | Add token/H2D markers and exact index audit | retained |
| `057313b` | Attribute backward nodes and summarize paired runs | retained |
| `26f7521` | MoT `index_copy` merge experiment | preserved as negative experiment |
| `c20a403` | MoT inverse-permutation experiment | preserved as negative experiment |
| `faf1baa` | Restore original MoT behavior; build mask metadata on device | retained final implementation |
| `e93b20c` / `b13b682` | Add opt-in fused AdamW while preserving the default foreach path | retained, opt-in |
| `8d6ea33` | Move ViT scalar and MSE target indexes to CPU metadata | retained |
| `805a773` | Keep single-rank statistics and normalization counts on CPU | retained |
| `2ceda53` | Cache exact FP32 timestep frequencies outside FSDP casts | retained |

## Recommended next work

1. Treat CPU-side autograd launch overhead in the 24-layer MoT stack as the main systems problem, but optimize it through a fused expert operation with a fused backward rather than another composition of PyTorch index primitives.
2. Measure a longer real-data run (at least 5 repetitions x 50 measured steps) before claiming an end-to-end mask speedup.
3. Investigate the remaining eight per-step `create_block_mask` synchronizations. A cache keyed by `(sample_lens, split_lens, attn_modes)` is promising only if real training shows meaningful mask-pattern reuse.
4. Profile ViT `.item()` synchronization and VAE/ViT overlap separately; these are more substantial than text token assembly.
