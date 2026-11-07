# Model asset compatibility validation

Date: 2026-11-07

Command:

```bash
/home/why/miniconda3/envs/bagel/bin/python \
  scripts/validate_model_assets.py \
  --models-dir /home/why/BAGEL/models
```

Result:

```text
LLM_OK layers=24 parameters=988071680
VIT_OK source_layers=27 loaded_layers=26 parameters=397747744
VAE_OK downsample=8 parameters=83819683
```

The public Qwen checkpoint does not contain BAGEL's generation expert or QK
normalization parameters. They are newly initialized during
`from_pretrained`, then the official `init_moe()` path copies the initial MoT
expert weights. This is expected for the reduced pretraining baseline, not a
fine-tuning-equivalence claim.

The public SigLIP checkpoint loads through BAGEL's custom vision class. The
default `vit_select_layer=-2` rule reduces the source model from 27 to 26
loaded transformer layers. The BAGEL VAE checkpoint loads with a spatial
downsample factor of 8.
