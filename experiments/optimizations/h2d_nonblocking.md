# Optimization 1: non-blocking pinned H2D transfers

Date: 2026-11-07

## Change

`SimpleCustomBatch.cuda()` now passes `non_blocking=True` to every tensor copy.
The existing DataLoader already uses `pin_memory=True`, so the main batch
payload can be enqueued without synchronizing the host on each copy.

## Correctness

The fixed-seed first-step losses matched the baseline exactly at displayed
precision:

```text
baseline:    mse=2.9803 ce=19.6786
nonblocking: mse=2.9803 ce=19.6786
```

Peak allocated and reserved memory were unchanged at 32,171.88 MiB and 33,788
MiB in the profiler runs.

## Profile comparison

| Metric, two active steps | Baseline | Non-blocking | Change |
| --- | ---: | ---: | ---: |
| Mean profiler step | 1,336.26ms | 1,349.60ms | -1.0% |
| Pinned H2D copies | 30 / 0.31ms | 30 / 0.35ms | same payload |
| Pinned-H2D-owned stream sync | 30 / 180.92ms | 0 / 0ms | eliminated |
| All stream sync | 66 / 181.38ms | 36 / 178.97ms | count -45% |

Trace:
`/home/why/BAGEL/traces/h2d_nonblocking_20261107/node161_295217.1794026059761257882.pt.trace.json`

## Conclusion

The change successfully removes all host synchronization directly owned by
pinned batch copies, but it does not improve end-to-end step time by itself.
The next pageable H2D operations become the waiting point and inherit almost
the same synchronization time. Call-stack analysis attributes these copies to:

- Flex Attention dense-mask construction: 12 copies.
- BAGEL `create_sparse_mask`: 6 copies.
- timestep-frequency construction: 2 copies.
- profiler bookkeeping: 8 copies.

The non-blocking change is retained because it is correct and removes an
avoidable serialization point. Device-native mask/timestep tensor creation is
the next isolated experiment.
