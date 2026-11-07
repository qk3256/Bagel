"""Aggregate BAGEL record_function ranges from a PyTorch profiler trace.

CUDA work is attributed by matching a GPU event's correlation id to the CUDA
runtime call that launched it, then locating the BAGEL range containing that
runtime call. CUDA durations are sums of event durations, not overlap-aware
wall-clock durations.
"""

import argparse
import collections
import json
import re
from pathlib import Path


MIB = 1024 * 1024
TOKEN_COUNTS_RE = re.compile(
    r"bagel::token_counts::seq=(\d+)::text=(\d+)::vit=(\d+)::vae=(\d+)"
)


def contains(interval, event):
    return (
        interval["pid"] == event["pid"]
        and (
            interval["tid"] == event["tid"]
            or interval["name"] == "bagel::backward"
        )
        and interval["ts"] <= event["ts"] < interval["ts"] + interval["dur"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--top-ops", type=int, default=0, help="Show the top inclusive CPU ops per range."
    )
    args = parser.parse_args()

    with args.trace.open(encoding="utf-8") as stream:
        events = json.load(stream)["traceEvents"]

    token_count_events = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") == "user_annotation"
        and event.get("name", "").startswith("bagel::token_counts::")
    ]
    ranges = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") == "user_annotation"
        and event.get("name", "").startswith("bagel::")
        and not event.get("name", "").startswith("bagel::token_counts::")
    ]
    steps = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") == "user_annotation"
        and event.get("name", "").startswith("ProfilerStep#")
    ]

    stats = collections.defaultdict(
        lambda: {
            "calls": 0,
            "cpu_us": 0.0,
            "cuda_us": 0.0,
            "kernels": 0,
            "memcpy_us": 0.0,
            "memcpys": 0,
            "sync_us": 0.0,
            "syncs": 0,
            "positive_alloc_bytes": 0,
            "max_transient_bytes": 0,
            "peak_allocated_bytes": 0,
            "cpu_ops": collections.defaultdict(lambda: [0.0, 0]),
        }
    )
    for interval in ranges:
        row = stats[interval["name"]]
        row["calls"] += 1
        row["cpu_us"] += interval.get("dur", 0.0)

    ranges_by_thread = collections.defaultdict(list)
    for interval in ranges:
        ranges_by_thread[(interval["pid"], interval["tid"])].append(interval)
    cross_thread_ranges = [
        interval for interval in ranges if interval["name"] == "bagel::backward"
    ]

    def candidate_ranges(event):
        return ranges_by_thread.get((event["pid"], event["tid"]), []) + [
            interval
            for interval in cross_thread_ranges
            if interval["tid"] != event["tid"]
        ]

    runtime_by_correlation = collections.defaultdict(list)
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "cuda_runtime":
            continue
        correlation = event.get("args", {}).get("correlation")
        if correlation is not None:
            runtime_by_correlation[correlation].append(event)
        if event.get("name") in {"cudaStreamSynchronize", "cudaDeviceSynchronize"}:
            for interval in candidate_ranges(event):
                if contains(interval, event):
                    row = stats[interval["name"]]
                    row["sync_us"] += event.get("dur", 0.0)
                    row["syncs"] += 1

    for gpu_event in events:
        if gpu_event.get("ph") != "X" or gpu_event.get("cat") not in {
            "kernel",
            "gpu_memcpy",
            "gpu_memset",
        }:
            continue
        correlation = gpu_event.get("args", {}).get("correlation")
        for runtime_event in runtime_by_correlation.get(correlation, []):
            for interval in candidate_ranges(runtime_event):
                if not contains(interval, runtime_event):
                    continue
                row = stats[interval["name"]]
                if gpu_event["cat"] == "kernel":
                    row["cuda_us"] += gpu_event.get("dur", 0.0)
                    row["kernels"] += 1
                elif gpu_event["cat"] == "gpu_memcpy":
                    row["memcpy_us"] += gpu_event.get("dur", 0.0)
                    row["memcpys"] += 1

    memory_events = [
        event
        for event in events
        if event.get("name") == "[memory]"
        and event.get("args", {}).get("Device Type") == 1
        and event.get("args", {}).get("Device Id") == 0
    ]
    for interval in ranges:
        inside = [event for event in memory_events if contains(interval, event)]
        if not inside:
            continue
        first_args = inside[0]["args"]
        allocated_at_entry = first_args["Total Allocated"] - first_args["Bytes"]
        transient = max(
            0,
            max(event["args"]["Total Allocated"] for event in inside)
            - allocated_at_entry,
        )
        row = stats[interval["name"]]
        row["positive_alloc_bytes"] += sum(
            max(0, event["args"]["Bytes"]) for event in inside
        )
        row["max_transient_bytes"] = max(row["max_transient_bytes"], transient)
        row["peak_allocated_bytes"] = max(
            row["peak_allocated_bytes"],
            max(event["args"]["Total Allocated"] for event in inside),
        )

    if args.top_ops:
        for event in events:
            if event.get("ph") != "X" or event.get("cat") != "cpu_op":
                continue
            for interval in candidate_ranges(event):
                if contains(interval, event):
                    op = stats[interval["name"]]["cpu_ops"][event["name"]]
                    op[0] += event.get("dur", 0.0)
                    op[1] += 1

    def union_duration_us(selected_events):
        by_thread = collections.defaultdict(list)
        for event in selected_events:
            by_thread[(event["pid"], event["tid"])].append(
                (event["ts"], event["ts"] + event.get("dur", 0.0))
            )
        total = 0.0
        for intervals in by_thread.values():
            end = None
            for start, stop in sorted(intervals):
                if end is None or start > end:
                    total += stop - start
                    end = stop
                elif stop > end:
                    total += stop - end
                    end = stop
        return total

    audited_op_names = (
        "IndexPutBackward0",
        "autograd::engine::evaluate_function: IndexPutBackward0",
        "aten::index_put_",
        "aten::_index_put_impl_",
        "aten::index",
    )
    audited_ops = {}
    for op_name in audited_op_names:
        selected = [
            event
            for event in events
            if event.get("ph") == "X"
            and event.get("cat") == "cpu_op"
            and event.get("name") == op_name
        ]
        audited_ops[op_name] = {
            "calls": len(selected),
            "inclusive_total_ms": sum(
                event.get("dur", 0.0) for event in selected
            ) / 1000,
            "same_name_union_ms": union_duration_us(selected) / 1000,
        }

    token_distributions = []
    for event in token_count_events:
        match = TOKEN_COUNTS_RE.fullmatch(event["name"])
        if match:
            sequence, text_tokens, vit_tokens, vae_tokens = map(int, match.groups())
            token_distributions.append(
                {
                    "sequence": sequence,
                    "text": text_tokens,
                    "vit": vit_tokens,
                    "vae": vae_tokens,
                    "mot_und": text_tokens + vit_tokens,
                    "mot_gen": vae_tokens,
                }
            )

    result = {
        "profiled_steps": len(steps),
        "mean_step_cpu_ms": (
            sum(step.get("dur", 0.0) for step in steps) / len(steps) / 1000
            if steps
            else 0.0
        ),
        "token_distributions": token_distributions,
        "index_audit": audited_ops,
        "ranges": {},
    }
    for name, row in sorted(stats.items()):
        result["ranges"][name] = {
            "calls": row["calls"],
            "cpu_total_ms": row["cpu_us"] / 1000,
            "cpu_per_call_ms": row["cpu_us"] / row["calls"] / 1000,
            "cuda_kernel_total_ms": row["cuda_us"] / 1000,
            "kernel_count": row["kernels"],
            "cuda_memcpy_total_ms": row["memcpy_us"] / 1000,
            "memcpy_count": row["memcpys"],
            "sync_total_ms": row["sync_us"] / 1000,
            "sync_count": row["syncs"],
            "positive_alloc_mib": row["positive_alloc_bytes"] / MIB,
            "max_transient_mib": row["max_transient_bytes"] / MIB,
            "peak_allocated_mib": row["peak_allocated_bytes"] / MIB,
            "top_cpu_ops": [
                {
                    "name": op_name,
                    "cpu_total_ms": op_values[0] / 1000,
                    "calls": op_values[1],
                }
                for op_name, op_values in sorted(
                    row["cpu_ops"].items(), key=lambda item: item[1][0], reverse=True
                )[: args.top_ops]
            ],
        }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print(f"profiled steps: {result['profiled_steps']}")
    print(f"mean profiler step CPU wall: {result['mean_step_cpu_ms']:.3f} ms")
    print("token distributions:")
    for row in result["token_distributions"]:
        print(
            "  "
            + ", ".join(f"{key}={value}" for key, value in row.items())
        )
    print("index audit (inclusive wrappers overlap; do not add them):")
    for name, row in result["index_audit"].items():
        print(
            f"  {name}: calls={row['calls']}, "
            f"inclusive_total_ms={row['inclusive_total_ms']:.3f}, "
            f"same_name_union_ms={row['same_name_union_ms']:.3f}"
        )
    header = (
        "range,calls,cpu_total_ms,cpu_per_call_ms,cuda_kernel_ms,kernels,"
        "memcpy_ms,memcpys,sync_ms,syncs,positive_alloc_mib,"
        "max_transient_mib,peak_allocated_mib"
    )
    print(header)
    for name, row in result["ranges"].items():
        print(
            f"{name},{row['calls']},{row['cpu_total_ms']:.3f},"
            f"{row['cpu_per_call_ms']:.3f},{row['cuda_kernel_total_ms']:.3f},"
            f"{row['kernel_count']},{row['cuda_memcpy_total_ms']:.3f},"
            f"{row['memcpy_count']},{row['sync_total_ms']:.3f},"
            f"{row['sync_count']},{row['positive_alloc_mib']:.3f},"
            f"{row['max_transient_mib']:.3f},{row['peak_allocated_mib']:.3f}"
        )
        for op in row["top_cpu_ops"]:
            print(
                f"  cpu_op,{op['cpu_total_ms']:.3f} ms,{op['calls']} calls,"
                f"{op['name']}"
            )


if __name__ == "__main__":
    main()
