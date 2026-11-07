"""Build a tiny, reproducible multimodal dataset through BAGEL's native formats."""

import argparse
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image, ImageEnhance, ImageOps


def image_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def write_parquet_shards(rows, output_dir, prefix):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    midpoint = len(rows) // 2
    for shard_index, shard_rows in enumerate((rows[:midpoint], rows[midpoint:])):
        path = (output_dir / f"{prefix}-{shard_index:05d}.parquet").resolve()
        pq.write_table(pa.Table.from_pylist(shard_rows), path, row_group_size=2)
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--download-root", type=Path, default=Path.home() / ".cache/bagel-data"
    )
    args = parser.parse_args()

    source_path = hf_hub_download(
        "uoft-cs/cifar10",
        "plain_text/test-00000-of-00001.parquet",
        repo_type="dataset",
        cache_dir=args.download_root / "hf",
    )
    source_rows = pq.read_table(source_path, columns=["img", "label"]).slice(0, 8)
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    samples = []
    for row in source_rows.to_pylist():
        image = Image.open(io.BytesIO(row["img"]["bytes"])).convert("RGB")
        label = class_names[row["label"]]
        image = image.resize((256, 256))
        samples.append((image, label))

    t2i_rows = [
        {
            "image": image_bytes(image),
            "captions": json.dumps(
                {"short": f"a photograph of a {label}", "label": label}
            ),
        }
        for image, label in samples
    ]
    write_parquet_shards(t2i_rows, args.output / "t2i", "part")

    edit_rows = []
    for image, label in samples:
        edited = ImageEnhance.Contrast(ImageOps.grayscale(image).convert("RGB")).enhance(1.4)
        edit_rows.append(
            {
                "image_list": [image_bytes(image), image_bytes(edited)],
                "instruction_list": [[f"convert the {label} image to high-contrast grayscale"]],
            }
        )
    edit_paths = write_parquet_shards(
        edit_rows, args.output / "editing/seedxedit_multi", "part"
    )
    info_path = args.output / "editing/parquet_info/seedxedit_multi_nas.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info = {str(path): {"num_row_groups": 2} for path in edit_paths}
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    image_dir = args.output / "vlm/images"
    image_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output / "vlm/llava_ov_si.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for index, (image, label) in enumerate(samples):
            filename = f"cifar10-{index:03d}.png"
            image.save(image_dir / filename)
            record = {
                "image": filename,
                "conversations": [
                    {"from": "human", "value": "<image> What is shown in this image?"},
                    {"from": "gpt", "value": f"The image shows a {label}."},
                ],
            }
            stream.write(json.dumps(record) + "\n")

    print(f"DATA_ROOT={args.output.resolve()}")
    print("t2i_rows=8 edit_rows=8 vlm_rows=8 parquet_shards_per_group=2")


if __name__ == "__main__":
    main()
