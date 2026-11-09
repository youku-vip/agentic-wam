"""Scoped, diagnostic VAE decoding for deployment image reviews only."""
from contextlib import contextmanager
from types import MethodType

import torch


def tensor_summary(value):
    x = value.detach().float()
    finite = torch.isfinite(x)
    selected = x[finite]
    return dict(shape=list(x.shape), dtype=str(value.dtype),
                elements=x.numel(), nan_count=int(torch.isnan(x).sum().item()),
                posinf_count=int(torch.isposinf(x).sum().item()), neginf_count=int(torch.isneginf(x).sum().item()),
                finite_min=float(selected.min().item()) if selected.numel() else None,
                finite_max=float(selected.max().item()) if selected.numel() else None)


@contextmanager
def temporary_float32(module):
    """Restore the exact original storages/dtypes, including mixed-dtype buffers, after decoding."""
    originals = []
    seen = set()
    try:
        for value in [*module.parameters(), *module.buffers()]:
            if value.is_floating_point() and id(value) not in seen and value.dtype != torch.float32:
                seen.add(id(value))
                originals.append((value, value.data))
                value.data = value.data.to(dtype=torch.float32)
        yield
    finally:
        for value, original in reversed(originals):
            value.data = original


@torch.no_grad()
def _review_decode(model, latents):
    precision = model._review_vae_decode_precision
    diagnostic = dict(precision=precision, latents=tensor_summary(latents),
                      scope='pixel export only; encoder and policy weights restored after decode')
    model._review_video_decode_diagnostics = diagnostic
    try:
        if precision == 'native':
            pixels = model._review_original_decode_video_latents(latents)
        else:
            nc = int(model.n_cameras)
            if latents.ndim != 5 or latents.shape[2] % nc:
                raise ValueError(f'expected camera-major [B,C,nc*T,H,W] latents, got {tuple(latents.shape)}')
            batch, channels, time, height, width = latents.shape
            per_view = (latents.reshape(batch, channels, nc, time//nc, height, width)
                        .permute(0, 2, 1, 3, 4, 5).reshape(batch*nc, channels, time//nc, height, width))
            # The inference method has an outer bf16 autocast. Disable it HERE as well as
            # casting weights, otherwise convolutions would still run in reduced precision.
            with torch.autocast(device_type=latents.device.type, enabled=False), temporary_float32(model.vae):
                # One camera at a time also reduces the decoder's peak activation memory.
                decoded = [model.vae.single_decode(view.unsqueeze(0).float()) for view in per_view]
            pixels = torch.cat(decoded, dim=0)
            output_channels, frames, out_h, out_w = pixels.shape[1:]
            pixels = (pixels.reshape(batch, nc, output_channels, frames, out_h, out_w)
                      .permute(0, 2, 1, 3, 4, 5).reshape(batch, output_channels, nc*frames, out_h, out_w))
        diagnostic['pixels'] = tensor_summary(pixels)
        diagnostic['status'] = ('nonfinite' if any(diagnostic['pixels'][key]
                                                  for key in ('nan_count', 'posinf_count', 'neginf_count')) else 'finite')
        return pixels
    except Exception as exc:
        diagnostic.update(status='error', error=f'{type(exc).__name__}: {exc}')
        raise


def install_review_vae_decode(model, precision='fp32'):
    """Override only the image decoder; denoising, state encoding and action decoding stay intact."""
    if precision not in ('fp32', 'native'):
        raise ValueError('review VAE decode precision must be fp32 or native')
    if not hasattr(model, '_review_original_decode_video_latents'):
        model._review_original_decode_video_latents = model._decode_video_latents
        model._decode_video_latents = MethodType(_review_decode, model)
    model._review_vae_decode_precision = precision
    model._review_video_decode_diagnostics = None
