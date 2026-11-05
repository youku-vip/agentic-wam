import math
import random
from collections.abc import Mapping

import torch
import torch.nn as nn
import torchvision.transforms as TF
import torchvision.transforms.functional as F


class ToTensor(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor):
        assert x.dtype == torch.uint8
        x = x.to(torch.float32) / 255.0
        return x


class LetterboxResize(nn.Module):
    """Fit a CHW image or TCHW clip into (height, width), then centre-pad.

    One scale is used for both axes. Padding is constant (black by default,
    before the dataset's [-1, 1] normalization); odd padding goes bottom/right.
    """

    def __init__(self, size, interpolation=TF.InterpolationMode.BILINEAR,
                 antialias=True, fill=0.0, scale_up=True):
        super().__init__()
        if isinstance(size, int) or len(size) != 2:
            raise ValueError("LetterboxResize requires size=[height, width]")
        if any(isinstance(v, bool) or int(v) != v or v <= 0 for v in size):
            raise ValueError("LetterboxResize size must contain two positive integers")
        if not math.isfinite(fill):
            raise ValueError("LetterboxResize fill must be finite")
        if not isinstance(scale_up, bool):
            raise ValueError("LetterboxResize scale_up must be a boolean")
        self.size = tuple(int(v) for v in size)
        self.interpolation = interpolation
        self.antialias = antialias
        self.fill = float(fill)
        self.scale_up = scale_up

    def forward(self, image):
        if not isinstance(image, torch.Tensor) or image.ndim not in (3, 4):
            raise ValueError("LetterboxResize expects a CHW image or TCHW clip tensor")
        height, width = image.shape[-2:]
        if height <= 0 or width <= 0:
            raise ValueError("LetterboxResize cannot resize an empty image")
        target_h, target_w = self.size
        scale = min(target_h / height, target_w / width)
        if not self.scale_up:
            scale = min(scale, 1.0)
        resized_h = max(1, min(target_h, int(height * scale + 0.5)))
        resized_w = max(1, min(target_w, int(width * scale + 0.5)))
        image = F.resize(image, [resized_h, resized_w],
                         interpolation=self.interpolation, antialias=self.antialias)
        top = (target_h - resized_h) // 2
        left = (target_w - resized_w) // 2
        return F.pad(image, [left, top, target_w - resized_w - left,
                             target_h - resized_h - top], fill=self.fill)


def configure_image_resize(transforms, mode, scale_up=True):
    """Replace explicit H/W Resize operations only when letterbox is selected."""
    if mode not in ("stretch", "letterbox"):
        raise ValueError(f"image_resize_mode must be stretch|letterbox, got {mode!r}")
    if mode == "stretch":
        return transforms
    if isinstance(transforms, Mapping):
        return {key: configure_image_resize(value, mode, scale_up) for key, value in transforms.items()}
    if transforms is None:
        raise ValueError("letterbox requires explicit train/val image transforms")
    result, found = [], False
    for transform in transforms:
        if isinstance(transform, TF.Resize):
            if transform.max_size is not None:
                raise ValueError("letterbox requires Resize(size=[height, width], max_size=None)")
            transform = LetterboxResize(transform.size, interpolation=transform.interpolation,
                                        antialias=transform.antialias, scale_up=scale_up)
        elif isinstance(transform, LetterboxResize):
            transform = LetterboxResize(transform.size, interpolation=transform.interpolation,
                                        antialias=transform.antialias, fill=transform.fill, scale_up=scale_up)
        found = found or isinstance(transform, LetterboxResize)
        result.append(transform)
    if not found:
        raise ValueError("letterbox requires Resize(size=[height, width]) in each image transform list")
    return result


def validate_video_image_size(size, shape_meta=None):
    """Validate the common RGB size used by Qwen patch/merge and video patches."""
    if isinstance(size, (int, str)) or len(size) != 2:
        raise ValueError("video_size must be [height, width]")
    if any(isinstance(v, bool) or int(v) != v or v <= 0 or int(v) % 32 for v in size):
        raise ValueError("video_size height/width must be positive multiples of 32")
    size = tuple(int(v) for v in size)
    if shape_meta is not None:
        images = shape_meta["images"]
        if not images:
            raise ValueError("shape_meta.images must contain at least one camera")
        for image in images:
            shape = tuple(image["shape"])
            if shape != (3, *size):
                raise ValueError(f"Camera {image['key']!r} shape={shape} disagrees with video_size={size}")
    return size


def resolve_eval_image_preproc(image_resize_mode, requested=""):
    """Keep deployment geometry aligned with its configured training recipe."""
    if image_resize_mode not in ("stretch", "letterbox"):
        raise ValueError(f"Unknown image_resize_mode: {image_resize_mode!r}")
    requested = str(requested or "").strip().lower()
    if requested in ("", "auto"):
        requested = "letterbox" if image_resize_mode == "letterbox" else "parity"
    if requested not in ("legacy", "parity", "parity_crop", "letterbox"):
        raise ValueError("AGENTICWAM_EVAL_PREPROC must be auto|legacy|parity|parity_crop|letterbox")
    if (requested == "letterbox") != (image_resize_mode == "letterbox"):
        raise ValueError(
            "Deployment preprocessing disagrees with processor.image_resize_mode; "
            "use the training image_resize_mode and AGENTICWAM_EVAL_PREPROC=auto"
        )
    return requested


class Pad(nn.Module):
    def __init__(self, padding, fill=0, padding_mode='constant'):
        super().__init__()
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode
        self.pad = TF.Pad(padding=tuple(padding), fill=fill, padding_mode=padding_mode)
    
    def forward(self, x: torch.Tensor):
        assert x.ndim == 4, "Can only pad tensor of 4 dims."
        return self.pad(x)


class ClipColorCast(nn.Module):
    """Per-clip, per-channel multiplicative colour cast.

    Models RoboTwin's `random_light` exactly: `_base_task.py` fixes a WHITE ambient term
    (`set_ambient_light([0.5, 0.5, 0.5])`) and then draws each directional/point light's RGB
    intensity as `[rand(), rand(), rand()]` ~ U[0,1]^3. Rendered radiance is therefore the albedo
    scaled INDEPENDENTLY PER CHANNEL -- a multiplicative cast, not an HSV rotation.

    This is why ColorJitter runs with `hue=0` in the RoboTwin configs: a hue rotation moves colour
    IDENTITY (red -> orange), which RoboTwin never does, and most RoboTwin training frames carry
    an instruction naming the target object's colour ("red bottle", "bottle with red cap"), so hue
    jitter would desynchronise the language<->appearance binding it is supposed to ground.

    One draw per call, so all T frames of a clip share the cast (the video-prediction target stays
    consistent with the observation); the processor calls the transform list once per camera, so
    cameras get independent draws. Input/output: [T,C,H,W] float in [0,1].
    """

    def __init__(self, low: float = 0.5, high: float = 1.0, p: float = 0.7):
        super().__init__()
        assert 0.0 < low <= high, f"need 0 < low <= high, got {low}, {high}"
        assert 0.0 <= p <= 1.0, f"p must be in [0,1], got {p}"
        self.low, self.high, self.p = float(low), float(high), float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"expected [T,C,H,W], got {tuple(x.shape)}"
        if random.random() > self.p:
            return x
        gain = torch.empty(1, x.shape[1], 1, 1, dtype=x.dtype, device=x.device).uniform_(
            self.low, self.high
        )
        return (x * gain).clamp_(0.0, 1.0)

    def extra_repr(self) -> str:
        return f"low={self.low}, high={self.high}, p={self.p}"
