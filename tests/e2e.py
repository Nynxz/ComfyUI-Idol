"""Headless end-to-end run through the stock ComfyUI nodes plus the two Idol nodes (the graph of the example workflow).

    <comfy>/.venv/bin/python tests/e2e.py --comfy <ComfyUI> --image photo.png --out <dir> [--mode synthetic|real]
        [--unet idol_fp16.safetensors] [--decoder idol_decoder_fp16.safetensors] [--clip-vision dino_v3_vit_h.safetensors]
        [--flux2-vae flux2-vae.safetensors] [--bg birefnet.safetensors] [--steps 50] [--cfg 4.0] [--seed 42] [--cpu]

Model names are relative to the configured model folders (extra_model_paths.yaml is honoured). Writes <name>.ply,
<name>_input.png (what the model saw), <name>_view.png (the splat rendered from the camera Idol predicted for the
input) and <name>_turntable.png (eight views), and prints timings and the camera.
"""

import argparse
import importlib
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="synthetic", choices=["synthetic", "real"])
    ap.add_argument("--unet", default="idol_fp16.safetensors")
    ap.add_argument("--decoder", default="idol_decoder_fp16.safetensors")
    ap.add_argument("--clip-vision", default="dino_v3_vit_h.safetensors")
    ap.add_argument("--flux2-vae", default="flux2-vae.safetensors")
    ap.add_argument("--bg", default="birefnet.safetensors")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=4.0, help="ComfyUI cfg = 1 + Idol's cfg_strength (reference 3.0)")
    ap.add_argument("--sampler", default="euler")
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-gaussians", type=int, default=262144)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    pack_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, args.comfy)
    sys.argv = sys.argv[:1] + (["--cpu"] if args.cpu else [])
    import comfy.options
    comfy.options.enable_args_parsing()
    from utils.extra_config import load_extra_path_config
    extra = os.path.join(args.comfy, "extra_model_paths.yaml")
    if os.path.isfile(extra):
        load_extra_path_config(extra)

    import torch

    sys.path.insert(0, os.path.dirname(pack_root))
    pack = importlib.import_module(os.path.basename(pack_root))
    import comfy.model_management
    import comfy_extras.nodes_bg_removal as bg_nodes
    import comfy_extras.nodes_gaussian_splat as splat_nodes
    import comfy_extras.nodes_triposplat as ts
    import nodes

    IdolPreprocessImage, IdolCamera = pack.idol.nodes.IdolPreprocessImage, pack.idol.nodes.IdolCamera
    with torch.inference_mode():  # as ComfyUI's executor runs nodes
        run(args, nodes, bg_nodes, splat_nodes, ts, IdolPreprocessImage, IdolCamera, comfy.model_management)


def run(args, nodes, bg_nodes, splat_nodes, ts, IdolPreprocessImage, IdolCamera, model_management):
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    os.makedirs(args.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.image))[0]
    save = lambda t, fn: Image.fromarray((t.cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)).save(os.path.join(args.out, fn))

    pil = ImageOps.exif_transpose(Image.open(args.image))
    mask = None
    if pil.mode == "RGBA" and np.array(pil.getchannel("A")).min() < 255:
        mask = torch.from_numpy(np.array(pil.getchannel("A")).astype(np.float32) / 255)[None]
    image = torch.from_numpy(np.array(pil.convert("RGB")).astype(np.float32) / 255)[None]

    t0 = time.time()
    bg = None if mask is not None else bg_nodes.LoadBackgroundRemovalModel.execute(args.bg).args[0]
    prepared, _alpha = IdolPreprocessImage.execute(image, {"mode": args.mode}, erode_radius=1, size=1024, mask=mask, bg_removal_model=bg).args
    save(prepared[0], f"{name}_input.png")
    t_pre = time.time() - t0

    (clip_vision,) = nodes.CLIPVisionLoader().load_clip(args.clip_vision)
    (flux2_vae,) = nodes.VAELoader().load_vae(args.flux2_vae)
    t0 = time.time()
    positive, negative, latent = ts.TripoSplatConditioning.execute(clip_vision, flux2_vae, prepared).args
    t_cond = time.time() - t0

    (model,) = nodes.UNETLoader().load_unet(args.unet, "default")
    t0 = time.time()
    (samples,) = nodes.KSampler().sample(model, args.seed, args.steps, args.cfg, args.sampler, args.scheduler,
                                         positive, negative, latent, denoise=1.0)
    torch.cuda.synchronize() if torch.cuda.is_available() and not args.cpu else None
    t_sample = time.time() - t0

    (decoder,) = nodes.VAELoader().load_vae(args.decoder)
    t0 = time.time()
    (splat,) = ts.VAEDecodeTripoSplat.execute(samples, decoder, args.num_gaussians, args.seed).args
    t_decode = time.time() - t0

    camera_info, yaw, pitch, distance, fov = IdolCamera.execute(samples).args
    (file3d,) = splat_nodes.SplatToFile3D.execute(splat, "ply").args
    with open(os.path.join(args.out, f"{name}.ply"), "wb") as f:
        f.write(file3d.get_bytes())

    render = lambda frames, cam: splat_nodes.RenderSplat.execute(splat, 768, 768, frames, 1.0, 2.0, 0.0, 0.0, "#000000", "color", camera_info=cam).args[0]
    view = render(1, camera_info)
    save(view[0], f"{name}_view.png")
    turn = render(8, camera_info)
    sheet = torch.cat([torch.cat(list(turn[i * 4:(i + 1) * 4]), dim=1) for i in range(2)], dim=0)
    save(sheet, f"{name}_turntable.png")

    n = int(splat.counts[0]) if splat.counts is not None else splat.positions.shape[1]
    print(f"{name}: preprocess {t_pre:.1f}s, conditioning {t_cond:.1f}s, sampling {t_sample:.1f}s "
          f"({args.steps} steps {args.sampler}/{args.scheduler} cfg {args.cfg}), decode {t_decode:.1f}s, {n} gaussians")
    print(f"predicted camera: yaw {yaw:.1f} deg, pitch {pitch:.1f} deg, distance {distance:.3f}, fov {fov:.2f} deg")
    print("wrote", ", ".join(f"{name}{s}" for s in [".ply", "_input.png", "_view.png", "_turntable.png"]), "in", args.out)
    model_management.unload_all_models()


if __name__ == "__main__":
    main()
