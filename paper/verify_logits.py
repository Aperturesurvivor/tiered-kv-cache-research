#!/usr/bin/env python3
"""Verify preserved question logits offline. Requires NumPy; performs no inference."""
import hashlib
import json
from pathlib import Path
import numpy as np

folder = Path(__file__).resolve().parent.parent / 'results/evidence/persistent-probe-0676d46bfe94/logits'
manifest = json.loads((folder / 'payload-manifest.json').read_text())
for item in manifest['files']:
    data = (folder / item['name']).read_bytes()
    assert len(data) == item['size'] and hashlib.sha256(data).hexdigest() == item['sha256']
stock = np.load(folder / 'stock_hf-question-logits.npy', allow_pickle=False)
full = np.load(folder / 'all_kv_retrieval-question-logits.npy', allow_pickle=False)
sparse = np.load(folder / 'rare_context-question-logits.npy', allow_pickle=False)
assert stock.shape == full.shape == sparse.shape
assert np.array_equal(stock, full)
delta = float(np.max(np.abs(sparse - stock)))
assert delta == 22.4765625, delta
print(f'PASS: full equals stock; sparse maximum absolute difference = {delta}; shape = {stock.shape}')
