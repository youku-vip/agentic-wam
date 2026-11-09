"""Read the same per-prompt Cosmos embedding files used by RobotVideoDataset."""
from collections import OrderedDict
import hashlib
from pathlib import Path

import torch


class CachedCosmosTextEncoder:
    """Bounded CPU cache; encode() returns [1,L,D], without loading text model weights."""

    def __init__(self, cache_dir, max_entries=8):
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        if not self.cache_dir.is_dir():
            raise FileNotFoundError(f'Cosmos text cache directory missing: {self.cache_dir}')
        self.max_entries = int(max_entries)
        if self.max_entries < 1:
            raise ValueError('max_entries must be positive')
        self._cache = OrderedDict()

    def encode(self, prompt):
        if prompt in self._cache:
            self._cache.move_to_end(prompt)
            return self._cache[prompt]
        digest = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        path = self.cache_dir / f'{digest}.cosmos_reason1.pt'
        if not path.is_file():
            raise FileNotFoundError(f'Missing training prompt embedding: {path}. '
                                    'Use the exact training instruction or precompute this prompt; '
                                    'no substitute embedding is generated.')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        emb = payload.get('prompt_embs') if isinstance(payload, dict) else None
        if not isinstance(emb, torch.Tensor) or not emb.is_floating_point():
            raise ValueError(f'Invalid prompt_embs tensor in {path}')
        if emb.ndim == 2:
            emb = emb.unsqueeze(0)
        if emb.ndim != 3 or emb.shape[0] != 1 or min(emb.shape[1:]) < 1:
            raise ValueError(f'Expected Cosmos embedding [L,D] or [1,L,D], got {tuple(emb.shape)}')
        if not torch.isfinite(emb).all() or not torch.count_nonzero(emb):
            raise ValueError(f'Nonfinite or all-zero Cosmos embedding: {path}')
        emb = emb.detach().contiguous()
        self._cache[prompt] = emb
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        return emb
