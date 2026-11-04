"""Bounded TorchCodec handles owned by a DataLoader worker's main thread.

Never retain native decoder handles in the parent process: a later DataLoader
fork must not inherit live FFmpeg decoder threads/handles from this cache.
"""
from collections import OrderedDict
import operator
import os
from pathlib import Path
import threading

import torch

_PID = os.getpid()
_DECODERS = OrderedDict()
_COUNTS = {"hits": 0, "misses": 0, "evictions": 0}


def _reset_for_process():
    global _PID
    if _PID != os.getpid():
        _DECODERS.clear()
        _COUNTS.update(hits=0, misses=0, evictions=0)
        _PID = os.getpid()


def clear_decoder_cache():
    _reset_for_process()
    _DECODERS.clear()
    _COUNTS.update(hits=0, misses=0, evictions=0)


def decoder_cache_info():
    _reset_for_process()
    return {"pid": _PID, "size": len(_DECODERS), **_COUNTS}


def discard_video_decoder(video_path):
    _reset_for_process()
    path = str(Path(video_path).absolute())
    for key in list(_DECODERS):
        if key[0] == path:
            del _DECODERS[key]


def get_video_decoder(video_path, *, device="cpu", cache_size=0, num_threads=None):
    from torchcodec.decoders import VideoDecoder

    cache_size = operator.index(cache_size)
    if cache_size < 0:
        raise ValueError("decoder cache size must be nonnegative")
    if num_threads is not None:
        num_threads = operator.index(num_threads)
        if num_threads < 0:
            raise ValueError("decoder thread count must be nonnegative")
    kwargs = {"device": device, "seek_mode": "approximate"}
    if num_threads is not None:
        kwargs["num_ffmpeg_threads"] = num_threads
    # Calls from the parent or an auxiliary thread keep the original fresh-decoder
    # behavior, even when cache_size is configured. CUDA decoding is not cached.
    if (cache_size == 0 or torch.utils.data.get_worker_info() is None
            or threading.current_thread() is not threading.main_thread()
            or torch.device(device).type != "cpu"):
        return VideoDecoder(video_path, **kwargs)

    _reset_for_process()
    path = Path(video_path).absolute()
    stat = path.stat()
    key = (str(path), str(device), num_threads, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    # A replaced/modified file must never use its previous decoder state.
    for old in list(_DECODERS):
        if old[0] == str(path) and old != key:
            del _DECODERS[old]
    while len(_DECODERS) > cache_size:
        _DECODERS.popitem(last=False)
        _COUNTS["evictions"] += 1
    if key in _DECODERS:
        _COUNTS["hits"] += 1
        _DECODERS.move_to_end(key)
        return _DECODERS[key]
    _COUNTS["misses"] += 1
    while len(_DECODERS) >= cache_size:
        _DECODERS.popitem(last=False)
        _COUNTS["evictions"] += 1
    decoder = VideoDecoder(video_path, **kwargs)
    _DECODERS[key] = decoder
    return decoder
