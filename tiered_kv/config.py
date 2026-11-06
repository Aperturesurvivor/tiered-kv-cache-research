from dataclasses import dataclass, asdict, fields
from pathlib import Path
import yaml
from .types import digest

MODEL_REVISION='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
DATA_REVISION='5e628be450b7e67fb7ae6e201bd6d8f7056f7672'
MODEL_PATH=Path('artifacts/hf/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots')/MODEL_REVISION
DATA_PATH=Path('artifacts/hf/hub/datasets--zai-org--LongBench/snapshots')/DATA_REVISION/'data.zip'


@dataclass(frozen=True)
class Config:
    id: str='default'
    W: int=2000
    C: int=32
    B: int=1024
    R: int=128
    k: int=4
    merge: str='mean_rot'
    full: bool=False
    protocol: str='exact_shared'
    model_revision: str=MODEL_REVISION
    precision: str='float16'
    batch_size: int=1
    rope_variant: str='original'
    retrieval: str='centroid'
    fixed_blocks: tuple[int,...]=()
    sink_tokens: int=0
    buffered_writes: bool=False

    def __post_init__(self):
        for name in ('W','B','R','C','k'):
            value=getattr(self,name)
            if type(value) is not int or value < (1 if name in ('W','B','R') else 0):
                raise ValueError(f'invalid {name}')
        if not self.id or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in self.id):
            raise ValueError('invalid config ID')
        if type(self.full) is not bool or self.merge not in ('mean_rot','mean_pre','first'):
            raise ValueError('invalid full/merge')
        if self.protocol not in ('exact_shared','online_tiered','query_replay'):
            raise ValueError('invalid protocol')
        if self.retrieval not in ('centroid','fixed'):
            raise ValueError('invalid retrieval strategy')
        object.__setattr__(self,'fixed_blocks',tuple(self.fixed_blocks))
        if any(type(b) is not int or b<0 for b in self.fixed_blocks) or len(set(self.fixed_blocks))!=len(self.fixed_blocks):
            raise ValueError('invalid fixed block selection')
        if self.fixed_blocks and self.retrieval!='fixed':
            raise ValueError('fixed blocks require fixed retrieval')
        if len(self.fixed_blocks)>self.k:
            raise ValueError('fixed blocks exceed retrieval budget')
        if type(self.sink_tokens) is not int or self.sink_tokens<0 or type(self.buffered_writes) is not bool:
            raise ValueError('invalid sink/write buffering configuration')
        if len(self.model_revision)!=40 or any(c not in '0123456789abcdef' for c in self.model_revision):
            raise ValueError('model revision must be an immutable full SHA')
        if self.precision!='float16' or self.batch_size!=1 or self.rope_variant!='original':
            raise ValueError('only fp16 / batch 1 / original RoPE supported')

    @property
    def digest(self):
        return digest(asdict(self))


def load_matrix(path):
    doc=yaml.safe_load(Path(path).read_text())
    if set(doc)!={'defaults','configs'} or not isinstance(doc['defaults'],dict) or not isinstance(doc['configs'],list):
        raise ValueError('matrix requires defaults and configs')
    allowed={f.name for f in fields(Config)}-{'id'}
    rows=[]
    for item in doc['configs']:
        if set(item)!={'id','overrides'} or not isinstance(item['overrides'],dict):
            raise ValueError('config requires id and overrides')
        values=doc['defaults']|item['overrides']
        if set(values)-allowed:
            raise ValueError(f'unknown keys: {set(values)-allowed}')
        rows.append(Config(id=item['id'],**values))
    if len({r.id for r in rows})!=len(rows):
        raise ValueError('duplicate config ID')
    return rows


def load_eval(path='configs/eval.yaml'):
    doc=yaml.safe_load(Path(path).read_text())
    if set(doc)!={'pilot','confirm','online-smoke'}:
        raise ValueError('unknown evaluation profile')
    expected={'budgets','depths','seeds','reports_per_band','latency_budgets','latency_trials'}
    for name,profile in doc.items():
        if name!='online-smoke' and set(profile)!=expected:
            raise ValueError('unknown evaluation key')
    if doc['online-smoke']!={'tokens':2048,'checkpoint':1024}:
        raise ValueError('invalid online smoke profile')
    return doc
