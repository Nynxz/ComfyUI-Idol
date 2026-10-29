"""Idol nodes. The model itself runs through ComfyUI's stock TripoSplat nodes (see the README); these two cover
the parts of Idol's pipeline those nodes do not have."""

import logging

import comfy.utils
import torch
import torch.nn.functional as F
from comfy_api.latest import IO

from . import camera as cam
from .preprocess import CROP_MARGIN, preprocess

logger = logging.getLogger(__name__)


class IdolPreprocessImage(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IdolPreprocessImage",
            display_name="Idol Preprocess Image",
            category="3d/idol",
            search_aliases=["idol crop", "idol background removal", "image to splat preprocess"],
            description="Prepares an image for Idol the way its own pipeline does: the object is cut out, the alpha "
                        "eroded by a pixel, a square of mode x the object's bounding box is cropped around it and "
                        "composited on black at 1024 x 1024. Connect the object's mask, or a background removal model "
                        "(BiRefNet) to predict one.",
            inputs=[
                IO.Image.Input("image"),
                IO.DynamicCombo.Input("mode", options=[
                    IO.DynamicCombo.Option("synthetic", []),
                    IO.DynamicCombo.Option("real", []),
                    IO.DynamicCombo.Option("custom", [
                        IO.Float.Input("crop_margin", default=1.2, min=1.0, max=4.0, step=0.05,
                                       tooltip="Crop side as a multiple of the object's bounding box."),
                    ]),
                ], tooltip="synthetic: renders and generated images of one object, the object fills 1/1.2 of the "
                           "frame (Idol's default). real: photographs, a looser 1/1.7. custom: set the margin."),
                IO.Mask.Input("mask", optional=True,
                              tooltip="The object's alpha (1 = object). From Remove Background, or Load Image's mask "
                                      "through Invert Mask for a transparent PNG. Leave empty to use bg_removal_model."),
                IO.BackgroundRemoval.Input("bg_removal_model", optional=True,
                                           tooltip="Used when no mask is connected (Load Background Removal Model, "
                                                   "birefnet.safetensors: the matting model Idol ships)."),
                IO.Int.Input("erode_radius", default=1, min=0, max=16, advanced=True,
                             tooltip="Erode the alpha by this many pixels before cropping (Idol: 1)."),
                IO.Int.Input("size", default=1024, min=256, max=4096, step=16, advanced=True,
                             tooltip="Output size. Idol is trained at 1024; other sizes run but are off-distribution."),
            ],
            outputs=[
                IO.Image.Output(display_name="image", tooltip="What the model sees: the object on black, 1024 x 1024."),
                IO.Mask.Output(display_name="mask", tooltip="The object's alpha in the output frame."),
            ],
        )

    @classmethod
    def execute(cls, image, mode, erode_radius, size, mask=None, bg_removal_model=None) -> IO.NodeOutput:
        if mask is None:
            if bg_removal_model is None:
                raise ValueError("Idol Preprocess Image: connect the object's mask or a background removal model")
            with torch.no_grad():
                mask = bg_removal_model.encode_image(image)
        margin = mode["crop_margin"] if mode["mode"] == "custom" else CROP_MARGIN[mode["mode"]]
        size = max(16, (int(size) // 16) * 16)  # the DINOv3 patch and the FLUX.2 VAE stride
        if mask.ndim == 2:
            mask = mask[None]
        if mask.shape[0] != image.shape[0]:
            mask = comfy.utils.repeat_to_batch_size(mask, image.shape[0])
        if tuple(mask.shape[1:]) != tuple(image.shape[1:3]):
            mask = F.interpolate(mask[:, None].float(), size=tuple(image.shape[1:3]), mode="bilinear", align_corners=False)[:, 0]
        mask = mask.to(image.device, image.dtype)
        images, alphas = zip(*(preprocess(image[i], mask[i], size, margin, erode_radius) for i in range(image.shape[0])))
        return IO.NodeOutput(torch.stack(images), torch.stack(alphas))


class IdolCamera(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IdolCamera",
            display_name="Idol Camera",
            category="3d/idol",
            search_aliases=["idol predicted camera", "splat input view", "input camera"],
            description="The camera Idol predicted for the input image: the denoiser estimates the viewpoint next to "
                        "the object. Feed camera_info to Render Splat to re-render the input view (a quick check of "
                        "the result), or start a turntable there; yaw/pitch/distance/fov plug into Create Camera Info.",
            inputs=[
                IO.Latent.Input("samples", tooltip="The KSampler output (its latent carries the camera token)."),
            ],
            outputs=[
                IO.Load3DCamera.Output(display_name="camera_info"),
                IO.Float.Output(display_name="yaw", tooltip="Degrees, Create Camera Info's orbit convention."),
                IO.Float.Output(display_name="pitch", tooltip="Degrees, positive = above the object."),
                IO.Float.Output(display_name="distance", tooltip="Camera distance from the object's centre; the object spans about one unit."),
                IO.Float.Output(display_name="fov", tooltip="Vertical field of view in degrees."),
            ],
        )

    @classmethod
    def execute(cls, samples) -> IO.NodeOutput:
        s = samples["samples"]
        if not getattr(s, "is_nested", False) or len(s.unbind()) < 2:
            raise ValueError("Idol Camera: this latent has no camera token. Connect the KSampler output of a graph "
                             "that starts from TripoSplat Conditioning.")
        camera = s.unbind()[1]
        if camera.shape[0] > 1:
            logger.warning("Idol Camera: batch of %d, using the first item", camera.shape[0])
        decoded = cam.decode_camera(camera[0, 0])
        yaw, pitch = cam.orbit_angles(decoded)
        return IO.NodeOutput(cam.camera_info(decoded), yaw, pitch, decoded["radius"], decoded["fov"])


NODES = [IdolPreprocessImage, IdolCamera]
