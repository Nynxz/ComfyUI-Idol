"""Idol's input preprocessing on ComfyUI tensors: a port of idol-sdk's preprocess.py.

Short side to `size`, erode the alpha, square crop of `crop_margin` x the object's bounding box, resize to
`size` x `size`, composite on black. Lanczos resampling through PIL, as the reference (comfy.utils.lanczos).
"""

import comfy.utils
import torch
import torch.nn.functional as F

# crop side = object bounding box x margin, so the object spans 1 / margin of the image
CROP_MARGIN = {"synthetic": 1.2, "real": 1.7}


def preprocess(image: torch.Tensor, alpha: torch.Tensor, size: int = 1024, crop_margin: float = 1.2,
               erode_radius: int = 1):
    """image (H, W, 3) and alpha (H, W) in [0, 1] -> (image (size, size, 3) over black, alpha (size, size))."""
    image, alpha = image.detach(), alpha.detach()
    rgba = torch.cat([image[..., :3].clamp(0, 1).movedim(-1, 0), alpha.clamp(0, 1)[None]], 0)[None]  # (1, 4, H, W)
    h, w = rgba.shape[-2:]
    s = size / min(w, h)
    rgba = comfy.utils.common_upscale(rgba, max(1, round(w * s)), max(1, round(h * s)), "lanczos", "disabled").clamp(0, 1)

    a = rgba[:, 3:4]
    if erode_radius > 0:  # a min filter over a (2r + 1) window: no background-coloured fringe survives
        a = -F.max_pool2d(-a, 2 * erode_radius + 1, stride=1, padding=erode_radius)
    rgba = torch.cat([rgba[:, :3], a], 1)

    ys, xs = torch.nonzero(a[0, 0] > 0, as_tuple=True)
    if xs.numel() == 0:
        raise ValueError("Idol Preprocess Image: no object found in the image (the mask is empty)")
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) / 2 * crop_margin
    left, top, right, bottom = int(cx - half), int(cy - half), int(cx + half), int(cy + half)

    H, W = rgba.shape[-2:]
    crop = rgba.new_zeros((1, 4, max(1, bottom - top), max(1, right - left)))  # outside the image stays 0, as PIL.crop
    sx0, sy0, sx1, sy1 = max(left, 0), max(top, 0), min(right, W), min(bottom, H)
    if sx1 > sx0 and sy1 > sy0:
        crop[:, :, sy0 - top:sy1 - top, sx0 - left:sx1 - left] = rgba[:, :, sy0:sy1, sx0:sx1]
    crop = comfy.utils.common_upscale(crop, size, size, "lanczos", "disabled").clamp(0, 1)
    out = crop[:, :3] * crop[:, 3:4]  # composite over black
    return out[0].movedim(0, -1), crop[0, 3]
