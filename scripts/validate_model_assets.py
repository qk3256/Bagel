"""Validate the model assets used by the single-A100 pretraining baseline."""

import argparse
import gc
from pathlib import Path

from modeling.autoencoder import load_ae
from modeling.bagel import (
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    llm_path = args.models_dir / "Qwen2.5-0.5B-Instruct"
    vit_path = args.models_dir / "siglip-so400m-patch14-384"
    vae_path = args.models_dir / "BAGEL-7B-MoT" / "ae.safetensors"

    llm_config = Qwen2Config.from_pretrained(llm_path)
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.freeze_und = False
    llm = Qwen2ForCausalLM.from_pretrained(llm_path, config=llm_config)
    llm.init_moe()
    print(
        "LLM_OK",
        f"layers={llm_config.num_hidden_layers}",
        f"parameters={parameter_count(llm)}",
    )
    del llm
    gc.collect()

    vit_config = SiglipVisionConfig.from_pretrained(vit_path)
    source_layers = vit_config.num_hidden_layers
    vit_config.num_hidden_layers = source_layers - 1
    vit_config.rope = False
    vit = SiglipVisionModel.from_pretrained(vit_path, config=vit_config)
    print(
        "VIT_OK",
        f"source_layers={source_layers}",
        f"loaded_layers={vit_config.num_hidden_layers}",
        f"parameters={parameter_count(vit)}",
    )
    del vit
    gc.collect()

    vae, vae_config = load_ae(local_path=vae_path)
    print(
        "VAE_OK",
        f"downsample={vae_config.downsample}",
        f"parameters={parameter_count(vae)}",
    )


if __name__ == "__main__":
    main()
