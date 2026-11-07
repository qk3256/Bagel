# Optimization 2: device-native mask and timestep tensors

Date: 2026-11-07

## Change

- Build `full_and_noise_seq_id`, `noise_seq_id`, and `document_id` directly on
  the target device in `create_sparse_mask`.
- Build timestep frequencies with a device-local `torch.arange`.

This removes the pattern “create CPU Tensor, then call `.to(device)`” without
changing mask values, dtypes, or timestep formulas.

## Correctness

The fixed-seed first-step losses remained identical:

```text
baseline:      mse=2.9803 ce=19.6786
device-native: mse=2.9803 ce=19.6786
```

Peak allocated and reserved memory remained 32,171.88 MiB and 33,788 MiB.

## Profile comparison

| Metric, two active steps | Non-blocking H2D | Device-native | Change |
| --- | ---: | ---: | ---: |
| Mean profiler step | 1,349.60ms | 1,342.72ms | +0.5% |
| Pageable H2D copies | 28 / 0.056ms | 20 / 0.043ms | count -29% |
| All stream sync | 36 / 178.97ms | 28 / 178.89ms | count -22% |

Trace:
`/home/why/BAGEL/traces/device_tensor_20261107/node161_297010.1794026363298392395.pt.trace.json`

## Conclusion

The intended eight pageable copies were eliminated (six from sparse-mask
construction and two from timestep-frequency construction). The measured step
change is within run-to-run noise, because 12 Flex Attention internal copies
and eight profiler-bookkeeping copies still provide a later host wait point.

This change is retained as a correctness-preserving cleanup, but is not claimed
as an end-to-end performance win. A profiler-free repeated-step benchmark is
required to separate application behavior from profiler synchronization.
