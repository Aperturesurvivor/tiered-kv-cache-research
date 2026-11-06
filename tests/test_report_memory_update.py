"""Synthetic corruption checks; these are not measured model results."""
from copy import deepcopy

import pytest

from research.report_memory_update import audit_append, policies, replay_versions, summarize
from test_report_memory import observations, Index


def versioned_observations(attention='sparse',mode='both'):
    manifest,episodes,rows,tok=observations(attention,mode)
    ep=episodes[0]
    versions=[dict(id=f'v{i}',budget=8192,prefix_tokens_count=n,first_suffix='<assistant>',
                   record_spans=[[1000,1001]]) for i,n in enumerate((6000,6100,6200))]
    ep['versions']=versions
    p=next(p for p in policies() if p['attention']==attention and p['memory']==mode)
    manifest['schedule'][0]['policy']=p
    for qi,(q,row) in enumerate(zip(ep['queries'],rows)):
        vi=(0,1,2,2,2)[qi];n=versions[vi]['prefix_tokens_count']
        q.update(version=vi,record=0)
        row.update(policy=p,example_id=f'episode-q{qi}',archive_version=vi,archive_tokens=n,
                   purpose='natural_model_memory_archive_update',policy_call_wall_s=2.5)
        if row['policy_config']:row['policy_config']['id']=p['id']
        if attention=='stock':
            for work in [row,*[rr[k] for rr in row['rounds'] for k in ('work_before_prompt','cumulative_work')]]:
                evaluated=work['model_token_evaluations']
                positions=2*(n*evaluated+evaluated*(evaluated+1)//2)
                work['layer_summed_valid_KV_positions']=positions
                work['logical_attention_head_pairs']=positions*4
                work['max_active_KV_entries']=n+evaluated if evaluated else 0
    return manifest,episodes,rows,tok


@pytest.mark.parametrize('attention,mode',[('stock','notes'),('sparse','none'),('sparse','both')])
def test_versioned_replay_keeps_wrong_notes_and_uses_each_visible_index(attention,mode):
    args=versioned_observations(attention,mode);seen=[]
    def factory(v,root):
        seen.append(v['id']);return Index()
    report=replay_versions(*args,{0},index_factory=factory)
    assert report['observed_rows']==5 and report['tokenizer_verified']
    assert seen==(['v0','v1','v2'] if attention=='sparse' else [])
    assert report['memory_trace'][2]['saved'] is False
    assert report['memory_trace'][3]['saved_incorrect']==(mode!='none')


@pytest.mark.parametrize('mutation',['wrong_version','hidden_future_extent','repaired_note','underreported_time'])
def test_rejects_corrupt_versioned_trace(mutation):
    args=versioned_observations();row=args[2][2]
    if mutation=='wrong_version':row['archive_version']=1
    elif mutation=='hidden_future_extent':row['archive_tokens']+=128
    elif mutation=='repaired_note':args[2][3]['memory_after']['entries'][-1]['answer']='42'
    else:row['policy_call_wall_s']=.01
    with pytest.raises(ValueError):
        replay_versions(*args,{0},index_factory=lambda v,root:Index())


def receipt():
    model=dict(revision='pinned',config=dict(num_hidden_layers=2,num_key_value_heads=2,
        hidden_size=64,num_attention_heads=4))
    old,n=148,129;layer_bytes=2*4*2*16
    value=dict(identity=dict(old_positions=old,new_positions=old+n,append_tokens=n,chunk=128,
        model_revision='pinned',**{k:'a'*64 for k in ('parent_checkpoint_sha256','parent_tokens_sha256','tokens_sha256','token_hash')}),
        prefix_audit=[dict(layer=i,prefix_positions=old,kv_bit_equal=True) for i in range(2)],
        append_model_tokens=n,append_forward_calls=2,append_layer_kv_positions=2*(n*old+n*(n+1)//2),
        logical_append_write_bytes=n*layer_bytes,logical_tensor_read_bytes_including_audit=3*old*layer_bytes,
        cuda_peak_allocated_bytes=1000000,**{k:.1 for k in ('checksum_mount_s','dense_init_s','append_s','checkpoint_s','prefix_audit_s')})
    return model,value


def test_append_accounting_requires_every_layer_and_every_ancestor():
    model,value=receipt();audit_append(value,model,148,277)
    for field in ('append_model_tokens','append_layer_kv_positions','logical_append_write_bytes',
                  'logical_tensor_read_bytes_including_audit'):
        broken=deepcopy(value);broken[field]-=1
        with pytest.raises(ValueError):audit_append(broken,model,148,277)
    broken=deepcopy(value);broken['prefix_audit'].pop()
    with pytest.raises(ValueError):audit_append(broken,model,148,277)


def test_report_keeps_hot_and_aged_questions_separate_and_charges_preparation():
    episodes=[dict(id=f'e{i}',budget=8192 if i<2 else 32768) for i in range(4)]
    kinds=['learn_old','hot_update','aged_update','historical','added_entity']
    rows=[];archives={}
    for ep in episodes:
        name=ep['id']+'.json';archives[name]=[dict(version_setup_wall_s=2.) for _ in range(3)]
        for p in policies():
            for qi,kind in enumerate(kinds):
                rows.append(dict(episode_id=ep['id'],policy=p,query_index=qi,task=kind,
                    exact_correct=not (p['attention']=='sparse' and qi==2),request_s=1.,
                    policy_call_wall_s=2.,generated_tokens=20,model_token_evaluations=100,
                    cuda_peak_allocated_bytes=1000000,archive_evidence_file=name))
    report=summarize(episodes,rows,archives)
    hot=[r for r in report['quality'] if r['policy']=='sparse-update-both-r2-cap128' and r['kind']=='hot_update']
    aged=[r for r in report['quality'] if r['policy']=='sparse-update-both-r2-cap128' and r['kind']=='aged_update']
    assert all(r['correct']==2 for r in hot) and all(r['correct']==0 for r in aged)
    assert all(r['matching_stock']=='stock-update-notes-r2-cap128' for r in hot+aged)
    assert all(r['whole_serving_calls_s']==10 and r['shared_preparation_observed_s']==6
               and r['charged_episode_s']==16 for r in report['episode_totals'])
