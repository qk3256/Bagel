# Three-way synchronization optimization comparison

Date: 2026-11-07

## End-to-end results

Each variant ran three independent 12-step jobs. The first two steps of each
job were excluded as compile/warmup, and the median of the remaining ten steps
was calculated. Input order, model initialization, optimizer, and random seeds
were fixed.

| Variant | Run-level medians | Median step | Median tokens/s | Peak allocated / reserved |
| --- | --- | ---: | ---: | ---: |
| Baseline (`67ce81e`) | 790.53 / 806.45 / 813.01ms | 806.45ms | 1,685 | 32,171.88 / 33,788MiB |
| Pinned H2D non-blocking (`116c208`) | 847.46 / 781.44 / 778.22ms | 781.44ms | 1,875 | 32,171.88 / 33,788MiB |
| Device-native mask/timestep (`5269e4f`) | 833.33 / 800.00 / 803.23ms | 803.23ms | 1,790 | 32,171.88 / 33,788MiB |

Relative to Baseline, non-blocking H2D improves median step time by 3.1%.
Device-native small Tensor construction gives back most of that gain and is
2.8% slower than non-blocking alone.

## Trace results

| Two active profiler steps | Baseline | Non-blocking | Device-native |
| --- | ---: | ---: | ---: |
| Mean profiler step | 1,336.26ms | 1,349.60ms | 1,342.72ms |
| `cudaStreamSynchronize` count | 66 | 36 | 28 |
| `cudaStreamSynchronize` time | 181.38ms | 178.97ms | 178.89ms |
| Pinned H2D | 30 / 0.310ms | 30 / 0.349ms | 30 / 0.348ms |
| Pageable H2D | 28 / 0.055ms | 28 / 0.056ms | 20 / 0.043ms |
| D2H | 8 / 0.019ms | 8 / 0.018ms | 8 / 0.018ms |

Profiler step time is dominated by tracing overhead and later synchronization
points, so it does not expose the small end-to-end non-blocking gain. It is
still useful for proving which synchronization sites were removed.

## Numerical correctness

All nine runs produced the same 12 logged MSE values and the same 12 logged CE
values at four-decimal precision. The first two values were:

```text
step 0: mse=2.9803 ce=19.6786
step 1: mse=2.8814 ce=19.8857
```

Memory was unchanged across variants. The retained production change is
non-blocking pinned H2D only; device-native small Tensor construction is kept
as a documented negative experiment and rolled back from the final code.
