# Native multimodal smoke-data validation

Date: 2026-11-07

The upstream toy-data URL returned a JavaScript verification page instead of
the documented zip. `scripts/prepare_smoke_data.py` therefore builds a small,
reproducible dataset from the public CIFAR-10 test Parquet hosted by
`uoft-cs/cifar10`.

The generated assets exercise BAGEL's native loaders:

- T2I: two Parquet shards with encoded PNG bytes and JSON captions.
- Unified editing: two Parquet shards with input/target image byte lists and
  instruction lists.
- VLM: eight PNG files plus LLaVA-style conversation JSONL.

All generated data stays under the ignored `data/bagel_example/` directory.

## PackedDataset validation

The validation used Flex packing, 256-pixel VAE images, 224-pixel ViT images,
an expected batch size of 1,024 tokens, and a maximum of 2,048 tokens.

```text
PACK_OK
sequence_length 1852
sample_lens [265, 527, 275, 785, 196]
packed_text_ids (60,)
packed_position_ids (1852,)
padded_images (5, 3, 256, 256)
packed_vae_token_indexes (1280,)
packed_vit_tokens (512, 588)
packed_vit_token_indexes (512,)
mse_loss_indexes (768,)
ce_loss_indexes (7,)
```

The packed batch contained T2I, unified-editing, and VLM records together, so
both MSE and CE losses are active and the VAE, ViT, text, token-index, and mask
construction paths are represented.

## Dependency/configuration findings

- Upstream `requirements.txt` omits `pandas`, although both Parquet loaders call
  `pyarrow.Table.to_pandas()`. The environment script now installs
  `pandas==2.0.3`.
- `SftJSONLIterableDataset` requires a `FrameSampler` constructor argument even
  for image-only records. The smoke config therefore includes minimal frame
  sampler settings.
