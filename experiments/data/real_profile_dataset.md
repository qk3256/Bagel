# Real multimodal profiling dataset

Date: 2026-11-07

## Scope

The profiling subset contains 256 records from each requested task. COCO Captions and VQA v2 share the same 256 COCO train2017 images so that image downloads are not duplicated. MagicBrush uses real source/target image pairs and real edit instructions.

| BAGEL task | Source | Records | Native input |
| --- | --- | ---: | --- |
| `vlm_sft/coco_caption_real` | COCO Captions train2017 | 256 | JSONL conversation + JPEG path |
| `vlm_sft/vqa_v2_real` | VQA v2 train split | 256 | JSONL conversation + the same JPEG path |
| `unified_edit/magicbrush_real` | `osunlp/MagicBrush` dev shards | 256 | Parquet `image_list=[source,target]`, `instruction_list=[[text]]` |

The prepared output occupies 471 MiB: 38.9 MiB of shared COCO JPEGs and 431.4 MiB of converted MagicBrush Parquet. The retained download cache occupies about 1001 MiB, including the 241 MiB COCO annotation archive, 20.7 MiB VQA annotation archive, 6.9 MiB VQA question archive, and pinned MagicBrush source shards.

MagicBrush is pinned to revision `6c254560879df7851233ea0fd2ae2a62764d0ef2`. The run consumed the first two dev shards needed to obtain 256 valid rows.

## Actual examples

- COCO Caption: `000000520229.jpg`; prompt `Describe this image in one sentence.`; answer `A dying flower is stuck to an onion.`
- VQA v2, sharing that COCO image: question `Is the flower in a vase?`; answer `yes`.
- MagicBrush: instruction `edit some mountains in the background`; real 500 x 500 source and 1024 x 1024 target images.

These are semantic training examples from public datasets, not the synthetic color/shape templates in the Smoke dataset. The fixed seed makes record selection reproducible, but this subset is intended for systems profiling rather than model-quality evaluation.

## BAGEL loader mapping and batch composition

`data/dataset_info.py` maps the two VLM datasets to the shared COCO directory and maps MagicBrush to four generated Parquet shards. `data/configs/single_a100_real_profile.yaml` makes both `unified_edit` and `vlm_sft` mandatory with equal top-level weight. The packer fills a variable-length packed sequence subject to `max_num_tokens=2048`; therefore a batch is not a fixed number of records. The profiled active steps contained:

| Step | Total tokens | Text | ViT | VAE | MoT understanding | MoT generation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1197 | 61 | 624 | 512 | 685 | 512 |
| 2 | 1777 | 49 | 704 | 1024 | 753 | 1024 |

The first active step contains one 512-token generation path plus VLM content; the second contains 1024 generation tokens plus VLM content. This confirms that the Trace covers both expert paths rather than a text-only or format-only batch.

## Reproduction

```bash
/home/why/miniconda3/envs/bagel/bin/python scripts/prepare_real_profile_data.py \
  --output /home/why/BAGEL/data/bagel_real_profile \
  --samples-per-dataset 256 \
  --seed 20261107

BAGEL_DATA_ROOT=/home/why/BAGEL/data/bagel_real_profile \
DATASET_CONFIG=data/configs/single_a100_real_profile.yaml \
TOTAL_STEPS=4 RUN_NAME=real-loader-smoke \
bash scripts/train_single_a100_smoke.sh --save_final_checkpoint False
```

The loader smoke run completed with deterministic losses and 32168.74 MiB peak allocated memory.
