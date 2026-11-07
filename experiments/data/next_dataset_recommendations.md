# Next-stage representative datasets (2026-11-07)

## Recommendation

Use a two-tier plan rather than jumping directly from the tiny smoke set to a
multi-million-pair crawl:

1. **Representative profiling set:** COCO 2017 Captions + VQA v2 + MagicBrush.
   This is roughly 36 GB of image payload, provides real photographs, human
   captions/questions/edit instructions, and is small enough to stage and
   convert on the current machine.
2. **Scale-up set:** add SEED-Data-Edit Part 2/3 for editing and, only after the
   loaders and performance experiments stabilize, Conceptual Captions 3M for
   broader T2I coverage.

This order fixes the current smoke set's lack of semantic and shape diversity
without immediately adding URL-crawl failures, hundreds of GB of storage, or
weak supervision.

## Dataset comparison

| Task | Dataset | Supervision and scale | Source format | Download/storage cost | Quality caveat | BAGEL adaptation |
|---|---|---|---|---|---|---|
| T2I / caption | COCO 2017 Captions | 118,287 train + 5,000 val real images; about five human captions/image | JPEG images + COCO JSON | 18 GB train + 1 GB val + 241 MB annotations | Captions are good but short and object-centric | Convert each image to one Parquet row: binary `image`, JSON-string `captions` dictionary |
| Image QA | VQA v2 train+val | 443,757 train + 214,354 val human questions; 10 answers/question | COCO image id + question/annotation JSON | Reuse the COCO image pool; metadata is small relative to images | Answers are often short; question distribution is benchmark-like | Write JSONL with `image` and `conversations`; use `multiple_choice_answer` as the GPT target |
| Editing, first representative set | MagicBrush train+dev | 8,807 train + 528 dev edit turns; 4,778 sessions; human instructions | HF Parquet: `source_img`, `instruction`, `target_img`, `turn_index` | 16.47 GB download / 16.46 GB materialized dataset | Real COCO source images, but targets were produced through a human-guided DALL-E 2 editing process | Group ordered turns into `image_list` and nested `instruction_list`, then write BAGEL edit Parquet |
| Editing, scale-up | SEED-Data-Edit Part 2/3 | Part 2: 52K real-world pairs; Part 3: 95K pairs / 21K human-annotated multi-turn rounds, up to 5 turns | Archives + tabular path/instruction metadata | 49.2 GB download plus extraction space | CC-BY-NC-4.0; mixed provenance, and some Part-3 sources are JourneyDB rather than photographs | Closest match to upstream `seedxedit_multi`: group turns directly into `image_list` / `instruction_list` |
| T2I, high-quality small alternative | Flickr30k | 31,783 real images, 158,915 human captions | Images + caption text/XML | Moderate, but image access/licensing is less frictionless than COCO | Smaller and human/activity biased | Same T2I Parquet converter as COCO |
| T2I, later scale | Conceptual Captions 3M | 3,318,333 train + 15,840 val image-URL/caption pairs | TSV of URL + automatically filtered/transformed alt-text | Metadata is small; fetching surviving images is typically hundreds of GB and retry-heavy | Not human captions; URL rot and domain/licensing checks are material | Download/filter first, then shard successful image bytes and captions into Parquet |

Official references:

- [COCO official download manifest](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm)
- [VQA v2 official download and schema](https://visualqa.org/download)
- [MagicBrush project](https://osu-nlp-group.github.io/MagicBrush/) and [official HF dataset card](https://huggingface.co/datasets/osunlp/MagicBrush)
- [SEED-Data-Edit Part 2/3 official dataset card](https://huggingface.co/datasets/AILab-CVC/SEED-Data-Edit-Part2-3)
- [Flickr30k Entities official repository](https://github.com/BryanPlummer/flickr30k_entities)
- [Conceptual Captions official page](https://ai.google.com/research/ConceptualCaptions/download)

## Why COCO can serve both T2I and VQA

VQA v2's real-image train/val splits use 82,783 COCO train2014 images and
40,504 COCO val2014 images. COCO 2017 uses the same image pool with a different
118K/5K train/val split. A converter can therefore index the 2017 JPEGs by the
numeric COCO image id and satisfy VQA references without storing a second image
copy. Keep task-specific train/validation membership in metadata rather than
inferring it from the directory name.

## Exact BAGEL target formats

### T2I

`T2IIterableDataset` expects Parquet files with:

```text
image: binary JPEG/PNG bytes
captions: JSON string, for example
          {"human_0": "A dog catches a frisbee.", "human_1": "..."}
```

The loader randomly selects one value from the caption dictionary. Preserve all
human captions rather than expanding the same image into five duplicate rows.

### VLM / QA

`SftJSONLIterableDataset` expects an image directory and JSONL records such as:

```json
{"image":"000000123456.jpg","conversations":[
  {"from":"human","value":"<image>\nWhat color is the bus?"},
  {"from":"gpt","value":"yellow"}
]}
```

Join VQA questions and annotations on `question_id`, map `image_id` to the COCO
JPEG, and use the normalized `multiple_choice_answer` target. Retain the ten raw
answers in converter-side metadata for audits, but the current loader consumes
one target string.

### Editing

`UnifiedEditIterableDataset` expects each Parquet row to contain an ordered
session:

```text
image_list:       [initial_image_bytes, target_turn_1_bytes, ...]
instruction_list: [[instruction_variant_1, ...], [turn_2_variant, ...], ...]
```

For MagicBrush, group by edit session/image id and sort by `turn_index`. Verify
that each turn's `source_img` equals the previous turn's `target_img`; fall back
to a two-image row when continuity does not hold. SEED Part 3 maps naturally to
the multi-turn representation, while Part 2 maps to two-image rows.

The edit loader also needs a `parquet_info` JSON mapping every Parquet path to
its `num_row_groups`. Generate it after final sharding so paths and row-group
counts cannot drift.

## Proposed staged subsets

### Stage A: loader correctness and realistic profiling

- COCO: 10K train images, all captions.
- VQA v2: all questions whose image is in the same 10K image subset.
- MagicBrush: full train+dev (9,335 available turns).
- Keep validation images/sessions disjoint.
- Profile at least three token/shape buckets instead of forcing every image to
  256x256: short edge buckets around 256, 384, and 512 where memory permits.

This is enough to invalidate smoke-only assumptions about repeated masks,
uniform image shapes, templated language, and fixed one-turn edits.

### Stage B: sustained single-GPU run

- Full COCO 2017 caption set and VQA v2 train/val metadata.
- SEED Part 2/3, sampled initially to balance single-turn and multi-turn edits.
- Record both record-count ratios and post-packing token ratios; equal dataset
  weights do not imply equal token or GPU-time contribution.

### Stage C: scale

- Conceptual Captions only after implementing URL validation, content hashing,
  failure accounting, license/domain policy, and deterministic Parquet shards.
- Do not use its requested-row count as the effective dataset size; report only
  successfully downloaded and decoded images.

## Conversion validation gates

Before using any converted set for training:

1. Decode 100 random images from every Parquet shard.
2. Render 50 random image/text or source/instruction/target examples for human
   inspection.
3. Enforce edit-session continuity and non-identical source/target hashes.
4. Run the native BAGEL loader for at least 1,000 yielded records with zero
   silent drops.
5. Report task record ratio, token ratio, resolution distribution, caption or
   answer length, and edit-turn count.
6. Keep source license/provenance fields in a sidecar manifest even when the
   current loader does not consume them.
