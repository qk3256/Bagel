# Single-A100 end-to-end training smoke run

Date: 2026-11-07

Commit under test: `3cf3147`

## Scope

Two optimizer steps using all three native multimodal datasets and the reduced
1.39B-parameter BAGEL composition. The run includes VAE encoding, SigLIP,
Qwen2MoT, Flex Attention, CE and MSE losses, backward, gradient clipping,
AdamW, scheduler, EMA update, and final checkpoint serialization.

## Result

```text
step 0: mse=2.9803 ce=19.6786 steps/s=0.05 tokens/s=0.09k MFU=0.1%
step 1: mse=2.8814 ce=19.8857 steps/s=1.18 tokens/s=1.87k MFU=2.4%
```

- Step 0 includes initial Flex/Inductor compilation and took approximately 20
  seconds.
- Step 1 took approximately 0.85 seconds and processed 1,590 tokens.
- Peak allocated CUDA memory reported by W&B: 32,171.88 MiB.
- Peak reserved CUDA memory reported by W&B: 33,802 MiB.
- The process exited normally after the final checkpoint.

## Artifacts on node161

- Log: `/home/why/BAGEL/results/single_a100_smoke/log.txt`
- Offline W&B run:
  `/home/why/BAGEL/wandb/offline-run-20261107_042452-single-a100-smoke-run0`
- Checkpoint: `/home/why/BAGEL/results/single_a100_smoke/checkpoints/0000002`
- Checkpoint size: approximately 21 GiB total
  - model: 5,557,204,624 bytes
  - EMA: 5,557,204,624 bytes
  - optimizer: 11,110,495,988 bytes

The checkpoint and W&B run are intentionally ignored by Git. They remain on
the experiment node for correctness comparisons.

## Findings

- The upstream training command requires the repository root in `PYTHONPATH`;
  otherwise direct execution of `train/pretrain_unified_navit.py` cannot import
  the sibling `data` package. The committed launch script sets it explicitly.
- PyTorch automatically changes `FULL_SHARD` to `NO_SHARD` at world size one.
- The 40GB A100 has enough capacity for this configuration, with roughly 6GB
  of reserved-memory headroom during the measured run.
