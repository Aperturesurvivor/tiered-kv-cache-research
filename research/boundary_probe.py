"""Exploratory paired replay of an observed split-block Qwen3 failure.

Execute from the original b5d7283 runtime root, after all GPU jobs are idle and
before archiving its prefixes. This changes only the selected next block; it
does not modify or replace the independent held-out confirmation experiment.
"""
import argparse
from dataclasses import asdict, replace
from functools import partial
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from tiered_kv.config import Config
from tiered_kv.generate import RunEngine, append_jsonl, weights_digest
from tiered_kv.store import atomic_json, file_sha
from tiered_kv.telemetry import DEVICE, ResourceGuard, runtime_identity
from tiered_kv.types import digest
from tiered_kv.validation import implementation_hash, parity


SOURCE = 'b5d72839ec2f59d7d8af3b36c6f7b78b2daf237a'
EXAMPLE = 'niah-4096-0.1-41'


def plan(source_run):
    source_run = Path(source_run)
    complete = json.loads((source_run/'complete.json').read_text())
    manifest = json.loads((source_run/'manifest.json').read_text())
    fixtures = json.loads((source_run/'fixtures.json').read_text())
    rows = [json.loads(line) for line in (source_run/'samples.jsonl').read_text().splitlines()]
    if not complete['weights_unchanged'] or complete['rows'] != len(rows):
        raise ValueError('completed immutable source screen required')
    if manifest['model']['model_id'] not in ('Qwen/Qwen3-0.6B', 'Qwen/Qwen3-1.7B'):
        raise ValueError('this diagnostic is restricted to the two Qwen3 screens')
    if digest(fixtures) != manifest['fixture_hash']:
        raise ValueError('fixture differs from source manifest')
    example = next(e for e in fixtures['examples'] if e['id'] == EXAMPLE)
    baseline = next(r for r in rows if r['example_id'] == EXAMPLE and r['config_id'] == 'rare_fast')
    original = Config(**baseline['policy'])
    if original.fixed_blocks != (2,) or example['gold_blocks'] != [2, 3]:
        raise ValueError('observed split-block failure identity differs')
    if baseline['code_found'] != 0:
        raise ValueError('source does not contain the observed failure')
    neighbor = replace(original, id='rare_plus_next', fixed_blocks=(2, 3))
    differences = {key for key in asdict(original) if asdict(original)[key] != asdict(neighbor)[key]}
    if differences != {'id', 'fixed_blocks'}:
        raise ValueError('unexpected paired policy difference')
    return manifest, example, baseline, (original, neighbor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    manifest, example, baseline, policies = plan(args.source_run)
    if args.plan_only:
        print(json.dumps({'example': EXAMPLE, 'policies': [asdict(p) for p in policies]}))
        return
    if DEVICE != 'cuda' or Path('SOURCE_REVISION').read_text().strip() != SOURCE:
        raise ValueError('run only on the original b5d7283 CUDA source')
    torch.set_num_threads(manifest['runtime']['threads'])
    torch.set_num_interop_threads(manifest['runtime']['interop_threads'])
    if implementation_hash() != manifest['implementation_hash'] or runtime_identity() != manifest['runtime']:
        raise ValueError('source implementation or runtime differs from the original screen')
    prefix = Path('artifacts/scratch')/args.source_run.name/(EXAMPLE+'-prefix')
    metadata = json.loads((prefix/'checkpoint.json').read_text())['metadata']
    spec = manifest['model']
    if metadata['token_hash'] != example['token_hash'] or metadata['model_revision'] != spec['revision']:
        raise ValueError('source prefix identity mismatch')
    identity = {'source_revision': SOURCE, 'source_manifest_sha256': file_sha(args.source_run/'manifest.json'),
                'source_samples_sha256': file_sha(args.source_run/'samples.jsonl'),
                'implementation_hash': implementation_hash(), 'script_sha256': file_sha(__file__),
                'runtime': runtime_identity(), 'example_hash': digest(example),
                'model': spec, 'policies': [asdict(p) for p in policies],
                'scope': 'exploratory observed-failure diagnostic; one adjacent block added; not held-out accuracy'}
    root = Path('results')/('boundary-'+spec['model_id'].replace('/', '--')+'-'+digest(identity)[:12])
    root.mkdir(exist_ok=False)
    atomic_json(root/'manifest.json', identity)
    atomic_json(root/'fixtures.json', {'examples': [example]})
    snapshot = snapshot_download(spec['model_id'], revision=spec['revision'], local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    tokenizer.apply_chat_template = partial(tokenizer.apply_chat_template, enable_thinking=False)
    with ResourceGuard() as guard:
        model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map={'': DEVICE},
            attn_implementation='sdpa').eval().requires_grad_(False)
        if getattr(model.config, '_commit_hash', None) not in (None, spec['revision']):
            raise ValueError('loaded revision mismatch')
        model.config._commit_hash = spec['revision']
        before = weights_digest(model)
        if before != json.loads((args.source_run/'validation.json').read_text())['weights_sha256']:
            raise ValueError('loaded weights differ from source screen')
        atomic_json(root/'validation.json', {'parity': parity(model, 512), 'weights_sha256': before})
        engine = RunEngine(model, tokenizer, root, guard)
        try:
            for policy in policies:
                row = engine.run(policy, example, protocol='query_replay', prefix_info=(prefix, metadata['prefill_s']))
                row.update(policy=asdict(policy), model_id=spec['model_id'], diagnostic=True,
                           gold_assisted=policy.id == 'rare_plus_next')
                if policy.id == 'rare_fast':
                    row['original_tokens_reproduced'] = row.get('token_ids') == baseline['token_ids']
                append_jsonl(root/'samples.jsonl', row)
                print(json.dumps({key: row.get(key) for key in
                    ('config_id', 'status', 'prediction', 'code_found', 'original_tokens_reproduced')}), flush=True)
                if row['status'] != 'ok' or row.get('original_tokens_reproduced') is False:
                    raise RuntimeError('paired replay failed or original output was not reproduced')
        finally:
            engine.adapter.uninstall()
        if weights_digest(model) != before:
            raise RuntimeError('frozen weights changed')
        atomic_json(root/'complete.json', {'rows': 2, 'weights_unchanged': True, 'original_tokens_reproduced': True})
    print('BOUNDARY_DONE', root, flush=True)


if __name__ == '__main__':
    main()
