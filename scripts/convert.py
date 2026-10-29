"""Convert the released Idol weights (speridlabs/idol) into the files ComfyUI's stock loaders take.

    python scripts/convert.py                                   # downloads speridlabs/idol, writes ./out/
    python scripts/convert.py --src /path/to/idol --out /ComfyUI/models [--dtype fp16|bf16|fp32]

Writes
    <out>/diffusion_models/idol_<dtype>.safetensors     the denoiser, for Load Diffusion Model
    <out>/vae/idol_decoder_<dtype>.safetensors          octree + gaussian decoders in one file, for Load VAE

Idol is the TripoSplat architecture, which ComfyUI implements natively (comfy/ldm/triposplat), so the weights load
unchanged. Only the file layout differs:
  * denoiser.safetensors carries `pos_pe`, the latent tokens' fixed Sobol positions. ComfyUI recomputes that buffer
    (SobolEngine, seed 123); the script checks the two are identical and drops it.
  * decoder.safetensors (octree) and decoder_gs.safetensors (gaussian) become one file under the `octree.` and
    `gs.` prefixes ComfyUI detects the TripoSplat gaussian decoder by.
BiRefNet, DINOv3 ViT-H+/16 and the FLUX.2 VAE are not converted: ComfyUI loads the published files (see README).
Needs torch, safetensors and (to download) huggingface_hub.
"""

import argparse
import json
import os

import torch
from safetensors.torch import load_file, save_file

DEFAULT_REPO = "speridlabs/idol"
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

# What ComfyUI's TripoSplat detection assumes (comfy/ldm/triposplat): a checkpoint with other sizes would not load.
COMFY_DENOISER = {"q_token_length": 8192, "in_channels": 16, "cam_channels": 5, "model_channels": 1024, "cond_channels": 1280,
                  "cond2_channels": 128, "num_blocks": 24, "num_refiner_blocks": 2, "num_heads": 16, "mlp_ratio": 4}
COMFY_OCTREE = {"model_channels": 1024, "cond_channels": 16, "num_blocks": 4, "num_heads": 16, "mlp_ratio": 4}
COMFY_GS = {"in_channels": 3, "model_channels": 1024, "cond_channels": 16, "num_blocks": 16, "num_heads": 16, "mlp_ratio": 4}
COMFY_REP = {"num_gaussians": 32, "perturbe_size": 1.5, "3d_filter_kernel_size": 0.0009, "scaling_bias": 0.004,
             "opacity_bias": 0.1, "lr": {"_xyz": 1.0, "_features_dc": 1.0, "_opacity": 1.0, "_scaling": 1.0, "_rotation": 0.1}}


def fetch(src, name):
    if os.path.isdir(src):
        return os.path.join(src, name)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(src, name)


def check_config(config):
    def same(section, expected):
        got = {k: config[section]["args"].get(k) for k in expected}
        if got != expected:
            raise SystemExit(f"{section}: config {got} differs from what ComfyUI's TripoSplat code assumes {expected}")
    same("denoiser", COMFY_DENOISER)
    same("decoder", COMFY_OCTREE)
    same("decoder_gs", COMFY_GS)
    rep = config["decoder_gs"]["args"]["representation_config"]
    if {k: rep.get(k) for k in COMFY_REP} != COMFY_REP:
        raise SystemExit(f"decoder_gs.representation_config {rep} differs from ComfyUI's defaults {COMFY_REP}")
    if config["denoiser"]["name"] != "LatentSeqMMFlowModel":
        raise SystemExit(f"unexpected denoiser {config['denoiser']['name']}")


def cast(sd, dtype):
    return {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=DEFAULT_REPO, help="Hugging Face repo id or a local folder with the released files")
    ap.add_argument("--out", default="out", help="output root; files go to <out>/diffusion_models and <out>/vae")
    ap.add_argument("--dtype", default="fp16", choices=DTYPES)
    args = ap.parse_args()
    dtype = DTYPES[args.dtype]

    with open(fetch(args.src, "config.json")) as f:
        config = json.load(f)
    check_config(config)
    metadata = {"format": "pt", "idol_source": args.src, "idol_converter": "ComfyUI-Idol scripts/convert.py"}

    # denoiser: drop pos_pe after checking it is the buffer ComfyUI recomputes
    sd = load_file(fetch(args.src, "denoiser.safetensors"))
    pos_pe = sd.pop("pos_pe")
    sobol = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=123).draw(pos_pe.shape[1]).unsqueeze(0)
    if not torch.allclose(pos_pe.float(), sobol, atol=1e-6):
        raise SystemExit("denoiser pos_pe is not the Sobol(seed=123) sequence ComfyUI uses; this checkpoint needs a code change")
    os.makedirs(os.path.join(args.out, "diffusion_models"), exist_ok=True)
    path = os.path.join(args.out, "diffusion_models", f"idol_{args.dtype}.safetensors")
    save_file(cast(sd, dtype), path, metadata=metadata)
    print(f"wrote {path}: {len(sd)} tensors, {sum(v.numel() for v in sd.values()) / 1e6:.1f}M parameters")

    # decoders: one file, prefixed as ComfyUI's OctreeGaussianDecoder names its submodules
    sd = {f"octree.{k}": v for k, v in load_file(fetch(args.src, "decoder.safetensors")).items()}
    sd.update({f"gs.{k}": v for k, v in load_file(fetch(args.src, "decoder_gs.safetensors")).items()})
    if "gs.base_offset_scale" not in sd or "octree.out_proj.weight" not in sd:
        raise SystemExit("decoder files do not look like the Idol octree + gaussian decoders")
    os.makedirs(os.path.join(args.out, "vae"), exist_ok=True)
    path = os.path.join(args.out, "vae", f"idol_decoder_{args.dtype}.safetensors")
    save_file(cast(sd, dtype), path, metadata=metadata)
    print(f"wrote {path}: {len(sd)} tensors, {sum(v.numel() for v in sd.values()) / 1e6:.1f}M parameters")


if __name__ == "__main__":
    main()
