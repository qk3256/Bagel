"""Prepare small real COCO/VQA/MagicBrush subsets in BAGEL native formats."""

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import random
import zipfile
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image


COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
VQA_QUESTIONS_URL = (
    "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/"
    "v2_Questions_Train_mscoco.zip"
)
VQA_ANNOTATIONS_URL = (
    "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/"
    "v2_Annotations_Train_mscoco.zip"
)
MAGICBRUSH_REPO = "osunlp/MagicBrush"


def download_file(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        return path
    partial = path.with_suffix(path.suffix + ".partial")
    with requests.get(url, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        with partial.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    stream.write(chunk)
    partial.replace(path)
    return path


def read_zipped_json(path, member):
    with zipfile.ZipFile(path) as archive:
        with archive.open(member) as stream:
            return json.load(stream)


def write_parquet_shards(rows, output_dir, prefix, num_shards=4):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for shard_index in range(num_shards):
        shard_rows = rows[shard_index::num_shards]
        path = (output_dir / f"{prefix}-{shard_index:05d}.parquet").resolve()
        pq.write_table(pa.Table.from_pylist(shard_rows), path, row_group_size=16)
        paths.append(path)
    return paths


def image_payload(value):
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return bytes(value["bytes"])
        if value.get("path") and Path(value["path"]).is_file():
            return Path(value["path"]).read_bytes()
    raise ValueError(f"Unsupported Hugging Face image value: {type(value)!r}")


def validate_image(payload):
    with Image.open(io.BytesIO(payload)) as image:
        image.verify()


def load_magicbrush_rows(cache_dir, count):
    endpoint = os.environ.get("HF_ENDPOINT")
    api = HfApi(endpoint=endpoint) if endpoint else HfApi()
    revision = api.repo_info(MAGICBRUSH_REPO, repo_type="dataset").sha
    files = sorted(
        path
        for path in api.list_repo_files(
            MAGICBRUSH_REPO, repo_type="dataset", revision=revision
        )
        if path.startswith("data/dev-") and path.endswith(".parquet")
    )
    if not files:
        raise RuntimeError("MagicBrush dev Parquet shards were not found")

    rows = []
    downloaded = []
    for repo_path in files:
        local_path = hf_hub_download(
            MAGICBRUSH_REPO,
            repo_path,
            repo_type="dataset",
            revision=revision,
            cache_dir=cache_dir / "huggingface",
        )
        downloaded.append(repo_path)
        parquet = pq.ParquetFile(local_path)
        for batch in parquet.iter_batches(batch_size=16):
            for source in batch.to_pylist():
                source_bytes = image_payload(source["source_img"])
                target_bytes = image_payload(source["target_img"])
                validate_image(source_bytes)
                validate_image(target_bytes)
                instruction = str(source.get("instruction") or "").strip()
                if not instruction:
                    continue
                if hashlib.sha256(source_bytes).digest() == hashlib.sha256(
                    target_bytes
                ).digest():
                    continue
                rows.append(
                    {
                        "image_list": [source_bytes, target_bytes],
                        "instruction_list": [[instruction]],
                    }
                )
                if len(rows) >= count:
                    return rows, revision, downloaded
    raise RuntimeError(f"MagicBrush only yielded {len(rows)} rows, need {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--download-root", type=Path, default=Path.home() / ".cache/bagel-real-profile"
    )
    parser.add_argument("--samples-per-dataset", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--download-workers", type=int, default=16)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    args.download_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    coco_zip = download_file(
        COCO_ANNOTATIONS_URL, args.download_root / "annotations_trainval2017.zip"
    )
    vqa_questions_zip = download_file(
        VQA_QUESTIONS_URL, args.download_root / "v2_questions_train.zip"
    )
    vqa_annotations_zip = download_file(
        VQA_ANNOTATIONS_URL, args.download_root / "v2_annotations_train.zip"
    )

    coco = read_zipped_json(coco_zip, "annotations/captions_train2017.json")
    captions_by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        captions_by_image[annotation["image_id"]].append(annotation["caption"])

    questions = read_zipped_json(
        vqa_questions_zip, "v2_OpenEnded_mscoco_train2014_questions.json"
    )["questions"]
    annotations = read_zipped_json(
        vqa_annotations_zip, "v2_mscoco_train2014_annotations.json"
    )["annotations"]
    annotation_by_question = {
        annotation["question_id"]: annotation for annotation in annotations
    }
    questions_by_image = defaultdict(list)
    for question in questions:
        if (
            question["image_id"] in captions_by_image
            and question["question_id"] in annotation_by_question
        ):
            questions_by_image[question["image_id"]].append(question)

    eligible_image_ids = sorted(questions_by_image)
    if len(eligible_image_ids) < args.samples_per_dataset:
        raise RuntimeError(
            f"Only {len(eligible_image_ids)} shared COCO/VQA images are available"
        )
    selected_image_ids = rng.sample(eligible_image_ids, args.samples_per_dataset)

    coco_dir = args.output / "shared/coco"
    coco_dir.mkdir(parents=True, exist_ok=True)

    def fetch_coco(image_id):
        filename = f"{image_id:012d}.jpg"
        url = f"http://images.cocodataset.org/train2017/{filename}"
        path = download_file(url, coco_dir / filename)
        validate_image(path.read_bytes())
        return image_id, filename, path

    downloaded_images = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.download_workers
    ) as pool:
        for image_id, filename, path in pool.map(fetch_coco, selected_image_ids):
            downloaded_images[image_id] = (filename, path)

    vlm_dir = args.output / "vlm"
    vlm_dir.mkdir(parents=True, exist_ok=True)
    caption_path = vlm_dir / "coco_caption.jsonl"
    vqa_path = vlm_dir / "vqa_v2.jsonl"
    selected_vqa_question_ids = []
    with caption_path.open("w", encoding="utf-8") as caption_stream, vqa_path.open(
        "w", encoding="utf-8"
    ) as vqa_stream:
        for image_id in selected_image_ids:
            filename, _ = downloaded_images[image_id]
            caption = rng.choice(captions_by_image[image_id])
            caption_record = {
                "image": filename,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\nDescribe this image in one sentence.",
                    },
                    {"from": "gpt", "value": caption},
                ],
            }
            caption_stream.write(json.dumps(caption_record) + "\n")

            question = rng.choice(questions_by_image[image_id])
            answer = annotation_by_question[question["question_id"]][
                "multiple_choice_answer"
            ]
            selected_vqa_question_ids.append(question["question_id"])
            vqa_record = {
                "image": filename,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{question['question']}",
                    },
                    {"from": "gpt", "value": answer},
                ],
            }
            vqa_stream.write(json.dumps(vqa_record) + "\n")

    magicbrush_rows, magicbrush_revision, magicbrush_files = load_magicbrush_rows(
        args.download_root, args.samples_per_dataset
    )
    edit_paths = write_parquet_shards(
        magicbrush_rows, args.output / "editing/magicbrush", "part"
    )
    parquet_info = {
        str(path): {"num_row_groups": pq.ParquetFile(path).num_row_groups}
        for path in edit_paths
    }
    info_path = args.output / "editing/parquet_info/magicbrush.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(parquet_info, indent=2), encoding="utf-8")

    manifest = {
        "seed": args.seed,
        "samples_per_dataset": args.samples_per_dataset,
        "coco_image_ids": selected_image_ids,
        "vqa_question_ids": selected_vqa_question_ids,
        "magicbrush_revision": magicbrush_revision,
        "magicbrush_source_files": magicbrush_files,
        "sources": {
            "coco_annotations": COCO_ANNOTATIONS_URL,
            "vqa_questions": VQA_QUESTIONS_URL,
            "vqa_annotations": VQA_ANNOTATIONS_URL,
            "magicbrush": f"https://huggingface.co/datasets/{MAGICBRUSH_REPO}",
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    image_bytes_total = sum(path.stat().st_size for _, path in downloaded_images.values())
    edit_bytes_total = sum(path.stat().st_size for path in edit_paths)
    print(f"DATA_ROOT={args.output.resolve()}")
    print(
        f"coco_caption_rows={args.samples_per_dataset} "
        f"vqa_rows={args.samples_per_dataset} "
        f"magicbrush_rows={len(magicbrush_rows)}"
    )
    print(
        f"downloaded_coco_images_mib={image_bytes_total / 1024**2:.1f} "
        f"converted_magicbrush_mib={edit_bytes_total / 1024**2:.1f}"
    )


if __name__ == "__main__":
    main()
