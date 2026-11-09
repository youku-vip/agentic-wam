"""CPU-only checks of scoped decoder precision; no checkpoint or CUDA required."""
import torch
import pytest

from agenticwam.utils.review_vae_decode import install_review_vae_decode


class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones((), dtype=torch.bfloat16), requires_grad=False)
        self.register_buffer('mixed_buffer', torch.tensor(.123456789, dtype=torch.float64))
        self.calls = []
        self.fail = False

    def single_decode(self, value):
        self.calls.append((value.dtype, self.weight.dtype, torch.is_autocast_enabled('cpu'), len(value)))
        if self.fail:
            raise RuntimeError('decoder failure')
        return value[:, :3] * self.weight


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vae = VAE()
        self.n_cameras = 3
        self.policy_weight = torch.nn.Parameter(torch.tensor(2., dtype=torch.bfloat16))

    def _decode_video_latents(self, value):
        return value[:, :3]


def state(module):
    return {name: (value.dtype, value.data_ptr(), value.detach().clone())
            for name, value in [*module.named_parameters(), *module.named_buffers()]}


def assert_restored(model, before):
    after = state(model)
    assert before.keys() == after.keys()
    for key in before:
        assert before[key][:2] == after[key][:2]
        assert torch.equal(before[key][2], after[key][2])


def test_fp32_disables_enclosing_autocast_restores_original_storages_and_camera_layout():
    model = Model()
    before = state(model)
    latents = torch.arange(2*6*9*2*3).reshape(2, 6, 9, 2, 3).float()
    install_review_vae_decode(model, 'fp32')
    with torch.autocast('cpu', dtype=torch.bfloat16):
        result = model._decode_video_latents(latents)
    assert torch.equal(result, latents[:, :3])
    assert model.vae.calls == [(torch.float32, torch.float32, False, 1)]*6
    assert_restored(model, before)
    assert model._review_video_decode_diagnostics['status'] == 'finite'


def test_exception_restores_vae_and_policy_parameters():
    model = Model()
    before = state(model)
    model.vae.fail = True
    install_review_vae_decode(model)
    with pytest.raises(RuntimeError, match='decoder failure'):
        model._decode_video_latents(torch.zeros(1, 6, 9, 2, 3))
    assert_restored(model, before)
    assert model._review_video_decode_diagnostics['status'] == 'error'


def test_native_mode_and_reinstallation_preserve_original_method():
    model = Model()
    original = model._decode_video_latents
    before = state(model)
    install_review_vae_decode(model)
    install_review_vae_decode(model, 'native')
    value = torch.ones(1, 6, 9, 2, 3)
    assert torch.equal(model._decode_video_latents(value), original(value))
    assert not model.vae.calls
    assert model._review_original_decode_video_latents == original
    assert_restored(model, before)


def test_nonfinite_latents_are_reported_without_silent_replacement():
    model = Model()
    install_review_vae_decode(model)
    value = torch.ones(1, 6, 9, 2, 3)
    value[0, 0, 0, 0, 0] = float('nan')
    result = model._decode_video_latents(value)
    diagnostic = model._review_video_decode_diagnostics
    assert torch.isnan(result[0, 0, 0, 0, 0])
    assert diagnostic['latents']['nan_count'] == 1 and diagnostic['pixels']['nan_count'] == 1
    assert diagnostic['status'] == 'nonfinite'
