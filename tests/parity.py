"""Numerical parity of the ComfyUI path against the reference Idol implementation (idol-sdk).

    <comfy>/.venv/bin/python tests/parity.py --comfy <ComfyUI> --ref <idol-sdk> \
        --weights <dir with config.json + the released safetensors> \
        --converted <dir holding diffusion_models/idol_fp16.safetensors and vae/idol_decoder_fp16.safetensors> [--cuda]

On random inputs it compares the denoiser's velocity (latent + camera), the octree decoder's logits, the gaussian
decoder's activated gaussians and ComfyUI's SPLAT frame transform, the image preprocessing (on a demo image with a
synthetic alpha) and the camera decode. The reference runs in fp32. The ComfyUI side runs twice: from the released
fp32 weights (implementation parity) and from the converted fp16 file (what users run).
"""

import argparse
import importlib
import json
import math
import os
import sys
import time
import types


def setup(comfy, ref, cuda):
    pack_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, comfy)
    sys.path.insert(0, ref)
    sys.path.insert(0, os.path.dirname(pack_root))
    sys.argv = sys.argv[:1] + ([] if cuda else ["--cpu"])
    import comfy.options
    comfy.options.enable_args_parsing()
    # the reference only needs TensorDict as a dict with a batch_size, and plyfile for save_ply (not used here)
    td = types.ModuleType("tensordict")

    class TensorDict(dict):
        def __init__(self, data=None, batch_size=None, device=None):
            super().__init__(data or {})
            self.batch_size = batch_size

    td.TensorDict = TensorDict
    sys.modules.setdefault("tensordict", td)
    ply = types.ModuleType("plyfile")
    ply.PlyData = ply.PlyElement = None
    sys.modules.setdefault("plyfile", ply)
    return importlib.import_module(os.path.basename(pack_root))


def report(name, a, b, tol):
    a, b = a.float().flatten(), b.float().flatten()
    rel = float((a - b).norm() / b.norm().clamp_min(1e-12))
    cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    ok = rel <= tol
    print(f"  {'ok ' if ok else 'BAD'} {name:<34} rel err {rel:.2e}  max abs {float((a - b).abs().max()):.2e}  cos {cos:.6f}")
    return ok


def fp32(model):
    model.float()
    model.dtype = torch.float32  # the reference casts its trunk activations to self.dtype
    return model.eval().requires_grad_(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--converted", required=True)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--image", default=None, help="an RGB image for the preprocessing check (default: a demo image from --ref)")
    args = ap.parse_args()

    global torch
    pack = setup(args.comfy, args.ref, args.cuda)
    import comfy.model_management
    import comfy.sd
    import numpy as np
    import torch
    from comfy.ldm.triposplat.gaussian import build_gaussian_models
    from idol.gaussian import quaternion_multiply
    from idol.models import GaussianFixedlenDecoder, LatentSeqMMFlowModel, OctreeProbabilityFixedlenDecoder
    from PIL import Image, ImageDraw
    from safetensors.torch import load_file

    import idol.camera as ref_camera
    import idol.preprocess as ref_preprocess

    device = comfy.model_management.get_torch_device()
    TensorDict = sys.modules["tensordict"].TensorDict
    with open(os.path.join(args.weights, "config.json")) as f:
        config = json.load(f)
    g = torch.Generator().manual_seed(0)
    ok = True

    # ---- denoiser -------------------------------------------------------------------------------------------------
    print("denoiser (velocity of latent + camera)")
    ref = fp32(LatentSeqMMFlowModel(**config["denoiser"]["args"]))
    ref.load_state_dict(load_file(os.path.join(args.weights, "denoiser.safetensors")), strict=True)
    ref = fp32(ref).to(device)
    z = torch.randn(1, 8192, 16, generator=g)
    cam = torch.randn(1, 1, 5, generator=g)
    feat1 = torch.nn.functional.layer_norm(torch.randn(1, 4101, 1280, generator=g), (1280,))
    ref_img = torch.randn(1, 128, 64, 64, generator=g)  # the FLUX.2 VAE latent of a 1024 image, as ComfyUI passes it
    feat2 = torch.cat([torch.zeros(1, 5, 128), ref_img.flatten(2).transpose(1, 2)], 1)
    t = torch.tensor([700.0])
    with torch.no_grad():
        out = ref(TensorDict({"latent": z.to(device), "camera": cam.to(device)}, batch_size=1), t.to(device),
                  TensorDict({"feature1": feat1.to(device), "feature2": feat2.to(device)}, batch_size=1))
    ref_latent, ref_cam = out["latent"].cpu(), out["camera"].cpu()
    del ref
    for label, path, dtype, tol in [("released fp32 weights", os.path.join(args.weights, "denoiser.safetensors"), torch.float32, 1e-4),
                                    ("converted idol_fp16", os.path.join(args.converted, "diffusion_models", "idol_fp16.safetensors"),
                                     None, 5e-2)]:
        patcher = comfy.sd.load_diffusion_model(path, model_options={"dtype": dtype} if dtype else {})
        comfy.model_management.load_models_gpu([patcher])
        dm = patcher.model.diffusion_model
        dt = next(dm.parameters()).dtype
        t0 = time.time()
        with torch.no_grad():
            port_latent, port_cam = dm([z.to(device, dt), cam.to(device, dt)], t.to(device, dt), context=feat1.to(device, dt),
                                       ref_latents=[ref_img.to(device, dt)])
        print(f"  {label}: ComfyUI model dtype {dt}, forward {time.time() - t0:.1f}s")
        ok &= report(f"latent velocity ({label})", port_latent.cpu(), ref_latent, tol)
        ok &= report(f"camera velocity ({label})", port_cam.cpu(), ref_cam, tol)
        del patcher, dm
        comfy.model_management.unload_all_models()

    # ---- decoders -------------------------------------------------------------------------------------------------
    print("octree + gaussian decoders")
    ref_oct = fp32(OctreeProbabilityFixedlenDecoder(**config["decoder"]["args"]))
    ref_oct.load_state_dict(load_file(os.path.join(args.weights, "decoder.safetensors")), strict=True)
    ref_oct = fp32(ref_oct).to(device)
    ref_gs = fp32(GaussianFixedlenDecoder(**config["decoder_gs"]["args"]))
    ref_gs.load_state_dict(load_file(os.path.join(args.weights, "decoder_gs.safetensors")), strict=True)
    ref_gs = fp32(ref_gs).to(device)
    latent = torch.randn(1, 8192, 16, generator=g)
    coords = (torch.randint(0, 4, (1, 64, 3), generator=g).float() + 0.5) / 4  # level-2 cell centres
    level = torch.tensor([8])
    points = torch.rand(1, 256, 3, generator=g)
    with torch.no_grad():
        ref_logits = ref_oct(coords.to(device), level.to(device), latent.to(device)).cpu()
        ref_feats = ref_gs(points.to(device), latent.to(device))
        ref_g = ref_gs.to_representation(points.to(device), ref_feats)[0]
    merged = {f"octree.{k}": v for k, v in load_file(os.path.join(args.weights, "decoder.safetensors")).items()}
    merged.update({f"gs.{k}": v for k, v in load_file(os.path.join(args.weights, "decoder_gs.safetensors")).items()})
    # From fp32 weights the decoders differ from the reference by ~1e-4 to 3e-3: ComfyUI's fused RMSNorm uses eps 1e-6
    # where the reference uses F.normalize (eps ~0), and the cross-attention keys from the 16-channel latent have small
    # norms. With eps 0 the two match exactly; it is well below the fp16 rounding of the shipped files.
    for label, sd, tol in [("released fp32 weights", merged, 5e-3),
                           ("converted idol_decoder_fp16", load_file(os.path.join(args.converted, "vae", "idol_decoder_fp16.safetensors")), 5e-2)]:
        vae = comfy.sd.VAE(sd=sd, dtype=torch.float32 if not args.cuda else None)
        comfy.model_management.load_models_gpu([vae.patcher])
        fsm, dt = vae.first_stage_model, vae.vae_dtype
        with torch.no_grad():
            logits = fsm.octree(coords.to(vae.device, dt), level.to(vae.device), latent.to(vae.device, dt))["logits"]
            pred = fsm.gs(x={"points": points.to(vae.device, dt)}, cond=latent.to(vae.device, dt))
            pg = build_gaussian_models(fsm.gs, {"points": points.to(vae.device, dt)}, pred)[0]
            positions, _scales, rotations, opacities, sh = pg.render_tensors()
        print(f"  {label}: ComfyUI VAE dtype {dt}")
        ok &= report(f"octree logits ({label})", logits.cpu(), ref_logits, tol)
        ok &= report(f"gaussian raw features ({label})", pred["features"].cpu(), ref_feats.cpu(), tol)
        ok &= report(f"xyz ({label})", pg.get_xyz.cpu(), ref_g.xyz.cpu(), tol)
        ok &= report(f"features_dc ({label})", pg._features_dc.cpu(), ref_g.features_dc.cpu(), tol)
        ok &= report(f"scaling ({label})", pg.get_scaling.cpu(), ref_g.scaling.cpu(), tol)
        ok &= report(f"opacity ({label})", pg.get_opacity.cpu(), ref_g.opacity.cpu(), tol)
        rot = torch.nn.functional.normalize(pg._rotation + pg.rots_bias[None].to(pg._rotation), dim=-1)
        ok &= report(f"rotation ({label})", rot.cpu(), ref_g.rotation.cpu(), tol)
        # SPLAT frame: what save_ply writes (z-up -> y-up) must equal ComfyUI's render_tensors
        xyz = ref_g.xyz.cpu().numpy()
        ply_xyz = np.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], 1)
        ply_rot = quaternion_multiply(np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]), ref_g.rotation.cpu().numpy())
        ok &= report(f"PLY positions ({label})", positions.cpu(), torch.from_numpy(ply_xyz), tol)
        pr, rr = rotations.cpu(), torch.from_numpy(ply_rot).float()
        sign = torch.where((pr * rr).sum(-1, keepdim=True) < 0, -1.0, 1.0)  # q and -q are the same rotation
        ok &= report(f"PLY rotations ({label})", pr * sign, rr, tol)
        ok &= report(f"PLY opacity/sh ({label})", torch.cat([opacities.cpu().flatten(), sh.cpu().flatten()]),
                     torch.cat([ref_g.opacity.cpu().flatten(), ref_g.features_dc.cpu().flatten()]), tol)
        del vae
        comfy.model_management.unload_all_models()

    # ---- preprocessing --------------------------------------------------------------------------------------------
    print("preprocessing (PIL reference vs tensor port)")
    image_path = args.image or os.path.join(args.ref, "demo", "examples", "fig-fox.webp")
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    alpha = Image.new("L", (w, h), 0)
    ImageDraw.Draw(alpha).ellipse([w * 0.15, h * 0.1, w * 0.8, h * 0.95], fill=255)  # off-centre object, exercises the crop
    alpha = alpha.resize((w // 3, h // 3)).resize((w, h), Image.BILINEAR)  # soft edge

    class FakeMatting:
        def remove_background(self, im):
            im = im.copy()
            im.putalpha(alpha.resize(im.size, Image.BILINEAR))
            return im

    for mode, margin in ref_preprocess.CROP_MARGIN.items():
        ref_out = ref_preprocess.preprocess_image(img, FakeMatting(), image_size=1024, crop_margin=margin)
        ref_t = torch.from_numpy(np.array(ref_out)).float() / 255
        port_img, _alpha = pack.idol.preprocess.preprocess(torch.from_numpy(np.array(img)).float() / 255,
                                                               torch.from_numpy(np.array(alpha)).float() / 255, 1024, margin, 1)
        diff = (port_img - ref_t).abs()
        print(f"  mode {mode} (margin {margin}): mean abs diff {float(diff.mean()) * 255:.3f}/255, "
              f"pixels differing by > 2/255: {float((diff.max(-1).values > 2 / 255).float().mean()) * 100:.2f}%")
        ok &= float(diff.mean()) * 255 < 0.5

    # ---- camera ---------------------------------------------------------------------------------------------------
    print("camera decode")
    worst = 0.0
    for i in range(200):
        c = torch.randn(5, generator=g)
        c[3] = abs(c[3]) * (0.0 if i % 50 == 0 else 1.0)  # a few orthographic cases
        r = ref_camera.decode_camera(c)
        d = pack.idol.camera.decode_camera(c)
        yaw, pitch = pack.idol.camera.orbit_angles(d)
        worst = max(worst, abs(r["radius"] - d["radius"]) / r["radius"], abs(r["fov"] - d["fov"]),
                    abs(math.degrees(r["pitch"]) - pitch), abs(((yaw + math.degrees(r["yaw"]) - 180) + 180) % 360 - 180))
        # camera_info: position maps back to radius * direction in the object frame, and the camera looks at the target
        info = pack.idol.camera.camera_info(d)
        p = info["position"]
        obj = torch.tensor([p["x"], -p["z"], p["y"]])
        worst = max(worst, float((obj - torch.tensor(d["direction"]) * d["radius"]).abs().max()) / d["radius"])
        q = info["quaternion"]
        qw, qx, qy, qz = q["w"], q["x"], q["y"], q["z"]
        local_z = torch.tensor([2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx * qx + qy * qy)])
        eye = torch.tensor([p["x"], p["y"], p["z"]])
        worst = max(worst, float((local_z - eye / eye.norm()).abs().max()))
    print(f"  worst deviation across 200 random cameras: {worst:.2e} (yaw is 180 - Idol's yaw, pitch equal)")
    ok &= worst < 1e-4

    print("PARITY OK" if ok else "PARITY FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
