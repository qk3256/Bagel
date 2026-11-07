# Smoke dataset audit

Date: 2026-11-07

## Source and construction

`scripts/prepare_smoke_data.py` downloads the first eight records from the
public `uoft-cs/cifar10` test Parquet. The source images are real CIFAR-10
examples with dataset class labels, but their native 32x32 pixels are upscaled
to 256x256 for the VAE path and transformed to 224x224 by the ViT loader.

### T2I

Each image is encoded as PNG bytes in Parquet. Two captions are synthesized
from the CIFAR class label:

```json
{"short": "a photograph of a cat", "label": "cat"}
```

At load time BAGEL randomly selects one value from this caption dictionary.
The image is real and the class-level text is usually aligned, but neither text
is a human-written visual description. It omits appearance, scene, composition,
and relationships, so this is primarily format/performance data.

### Unified editing

For every source image, the script creates a deterministic target by converting
the image to grayscale and applying 1.4x contrast. The Parquet record stores:

```text
input:       original upscaled CIFAR image
target:      high-contrast grayscale version of the same image
instruction: "convert the cat image to high-contrast grayscale"
```

For the first sample, input channel means are
`[110.03, 110.77, 104.37]`; target channel means are
`[108.83, 108.83, 108.83]`, and the pixel difference spans the full image.
The input-target-instruction relationship is semantically correct, but it is a
synthetic single-operation edit rather than a natural editing dataset.

### VLM

The same PNG is referenced from LLaVA-style JSONL, with templated QA derived
from the class label:

```json
{
  "image": "cifar10-000.png",
  "conversations": [
    {"from": "human", "value": "<image> What is shown in this image?"},
    {"from": "gpt", "value": "The image shows a cat."}
  ]
}
```

The image and class target are real dataset supervision. The question and
sentence-form answer are synthetic templates and do not test detailed visual
understanding, grounding, OCR, or reasoning.

## Sampling and batch composition

All three groups have weight 1 and `is_mandatory: true`. `PackedDataset` first
adds one record from every mandatory group, then samples one additional record
with equal one-third probability. With the current 1,024 expected-token and
2,048 maximum-token settings, every inspected batch contained four dataset
records: one from each task plus one duplicated task.

In a fixed-seed audit of 20 batches (80 records):

| Batch composition | Batches |
| --- | ---: |
| 1 T2I, 1 edit, 2 VLM | 11 |
| 2 T2I, 1 edit, 1 VLM | 7 |
| 1 T2I, 2 edit, 1 VLM | 2 |

Aggregate record counts were 27 T2I, 22 edit, and 31 VLM. The expected
long-run record ratio is 1:1:1, but token ratio is not: edit records are longer
because they carry source VAE tokens, source ViT tokens, and target VAE tokens.
Observed packed sequence lengths ranged from 1,070 to 1,848 in this audit.

## Fitness assessment

| Property | Assessment |
| --- | --- |
| Native BAGEL schemas and loaders | Representative |
| Image decode/transform/H2D path | Representative at small resolution |
| VAE + ViT + text token assembly | Representative |
| Mask and MoT modality routing | Representative |
| Natural caption quality | Not representative |
| Natural editing diversity | Not representative |
| VLM reasoning/grounding quality | Not representative |
| Dataset scale and I/O pressure | Not representative |

The dataset is suitable for deterministic correctness and performance work on
the multimodal execution path. It must not be used to draw model-quality or
training-convergence conclusions.
