"""Print compact operator, transfer, and synchronization summaries from a trace."""

import argparse
import collections
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    with args.trace.open(encoding="utf-8") as stream:
        events = json.load(stream)["traceEvents"]

    totals = collections.defaultdict(lambda: [0.0, 0])
    by_external_id = collections.defaultdict(list)
    for event in events:
        if event.get("ph") != "X":
            continue
        category = event.get("cat", "")
        name = event.get("name", "")
        duration = event.get("dur", 0.0)
        totals[(category, name)][0] += duration
        totals[(category, name)][1] += 1
        external_id = event.get("args", {}).get("External id")
        if external_id is not None:
            by_external_id[external_id].append((category, name))

    for category in ("kernel", "gpu_memcpy", "cuda_runtime", "cpu_op"):
        print(f"\nTOP {category}")
        rows = [
            (duration, count, name)
            for (event_category, name), (duration, count) in totals.items()
            if event_category == category
        ]
        for duration, count, name in sorted(rows, reverse=True)[: args.top]:
            print(f"{duration / 1000:10.3f} ms {count:7d}  {name}")

    synchronization = collections.defaultdict(lambda: [0.0, 0])
    sync_names = {"cudaStreamSynchronize", "cudaDeviceSynchronize"}
    for event in events:
        if event.get("ph") != "X" or event.get("name") not in sync_names:
            continue
        external_id = event.get("args", {}).get("External id")
        owners = [
            f"{category}:{name}"
            for category, name in by_external_id.get(external_id, [])
            if name not in sync_names
        ]
        owner = " | ".join(owners[:4]) or "UNKNOWN"
        synchronization[owner][0] += event.get("dur", 0.0)
        synchronization[owner][1] += 1

    print("\nSYNCHRONIZATION OWNERS")
    for owner, (duration, count) in sorted(
        synchronization.items(), key=lambda item: item[1][0], reverse=True
    ):
        print(f"{duration / 1000:10.3f} ms {count:7d}  {owner}")


if __name__ == "__main__":
    main()
