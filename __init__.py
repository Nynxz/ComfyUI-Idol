"""ComfyUI-Idol: native ComfyUI support for Idol (speridlabs/idol), a single image to 3D gaussian splat model.

Idol is the TripoSplat architecture retrained, and ComfyUI implements that natively, so with the weights in
ComfyUI's layout (scripts/convert.py, or the files on nynxz/Idol) the stock nodes run it: Load Diffusion Model,
Load VAE, Load CLIP Vision, TripoSplat Conditioning, KSampler, TripoSplat Decode, Save Splat. The nodes here add
what Idol's own pipeline has on top: Idol Preprocess Image (its crop modes, background removal in place), Idol
Camera (the input camera the model predicts next to the object) and Idol Prune Splat (drops the near-transparent
gaussians that make viewers and exports hazy).
"""

from comfy_api.latest import ComfyExtension
from typing_extensions import override

from .idol.nodes import NODES


class IdolExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return NODES


async def comfy_entrypoint():
    return IdolExtension()
