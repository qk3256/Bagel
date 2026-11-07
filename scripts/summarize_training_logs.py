"""Summarize profiler-free BAGEL console logs without double-counting logger echoes."""

import argparse
import json
import re
import statistics
from pathlib import Path


STEP_RE = re.compile(
    r"\(step=(\d+)\).*?Train Loss mse: ([0-9.]+), "
    r"Train Loss ce: ([0-9.]+), Train Steps/Sec: ([0-9.]+), "
    r"Tokens/Sec: ([0-9.]+)k"
)
MEM_RE = re.compile(r"wandb:\s+mem_allocated\s+([0-9.]+)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--discard", type=int, default=2)
    args = parser.parse_args()

    summaries = []
    loss_sequences = []
    for path in args.logs:
        text = path.read_text(encoding="utf-8", errors="replace")
        steps = {}
        for match in STEP_RE.finditer(text):
            step, mse, ce, steps_per_sec, tokens_k = match.groups()
            steps[int(step)] = {
                "mse": float(mse),
                "ce": float(ce),
                "steps_per_sec": float(steps_per_sec),
                "tokens_per_sec": float(tokens_k) * 1000,
            }
        selected = [steps[index] for index in sorted(steps) if index >= args.discard]
        if not selected:
            raise RuntimeError(f"No post-warmup steps found in {path}")
        memory_matches = MEM_RE.findall(text)
        summary = {
            "log": str(path),
            "measured_steps": len(selected),
            "median_step_ms": statistics.median(
                1000 / row["steps_per_sec"] for row in selected
            ),
            "median_tokens_per_sec": statistics.median(
                row["tokens_per_sec"] for row in selected
            ),
            "peak_allocated_mib": (
                float(memory_matches[-1]) if memory_matches else None
            ),
        }
        summaries.append(summary)
        loss_sequences.append(
            [(steps[index]["mse"], steps[index]["ce"]) for index in sorted(steps)]
        )

    print(
        json.dumps(
            {
                "discarded_steps": args.discard,
                "loss_sequences_equal": all(
                    sequence == loss_sequences[0] for sequence in loss_sequences[1:]
                ),
                "runs": summaries,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
