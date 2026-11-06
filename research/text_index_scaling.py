"""Synthetic disk-index scaling, not model quality or context extrapolation."""
import json
from pathlib import Path
import platform
import time
import warnings
import numpy as np
from research.semantic_index import DiskTextIndex
from research.resources import HostObservation
from tiered_kv.store import atomic_json,file_sha
from tiered_kv.types import digest


def main():
    identity={'source_sha256':file_sha('research/semantic_index.py'),'script_sha256':file_sha(__file__),
        'platform':platform.platform(),'budgets':[4096,32768,262144,1048576,4194304],
        'block_size':128,'vector_dimensions':384,'scope':'synthetic index data structure only; no model inference',
        'query':'What is the access code for cabinet_TARGET?','trials':7,'cache':'warm; no OS eviction'}
    root=Path('artifacts/text-index-scaling')/digest(identity)[:12]
    root.mkdir(parents=True,exist_ok=False)
    output=Path('results/evidence')/('text-index-scaling-'+digest(identity)[:12]+'.json')
    index=DiskTextIndex(root/'metadata.sqlite3',create=True)
    rng=np.random.default_rng(139)
    count=0
    results=[]
    try:
        for tokens in identity['budgets']:
            start=time.perf_counter()
            for offset in range(count,tokens//128,64):
                rows=[(i,i*128,(i+1)*128,f'The access code for cabinet_{i} is {10000000+i}. This archive contains records of cabinets and blue folders. '+('The archive stores relevant records. '*16))
                      for i in range(offset,min(offset+64,tokens//128))]
                vectors=rng.normal(size=(len(rows),384)).astype(np.float32)
                vectors/=np.linalg.norm(vectors,axis=1,keepdims=True)
                index.add(rows,vectors)
            count=tokens//128
            build_s=time.perf_counter()-start
            query=f'What is the access code for cabinet_{count-1}?'
            vector=np.frombuffer(index.db.execute('SELECT vector FROM chunks WHERE id=?',(count-1,)).fetchone()[0],dtype='<f4')
            check=np.stack([np.frombuffer(r[0],dtype='<f4') for r in index.db.execute('SELECT vector FROM chunks LIMIT 64')])
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                actual=check@vector
            reference=np.einsum('ij,j->i',check.astype(np.float64),vector.astype(np.float64),optimize=False)
            np.testing.assert_allclose(actual,reference,atol=1e-6,rtol=1e-5)
            numeric_check={'finite':bool(np.isfinite(actual).all()),'max_abs_error':float(abs(actual-reference).max()),
                           'numpy':np.__version__,'warnings':[str(w.message) for w in captured]}
            for method in ('lexical','lexical_rare','semantic'):
                times=[]
                # One untimed warm lookup, followed by seven complete queries.
                index.search(query,vector,4,method)
                with HostObservation() as host:
                    for trial in range(7):
                        start=time.perf_counter()
                        selected=index.search(query,vector,4,method)
                        times.append((time.perf_counter()-start)*1000)
                row={'tokens_represented':tokens,'blocks':count,'method':method,'latency_ms':times,
                     'selected':selected,'database_bytes':(root/'metadata.sqlite3').stat().st_size,
                     'incremental_build_s':build_s,'lookup_details':index.last_search,'numeric_check':numeric_check,**host.metrics}
                results.append(row)
                atomic_json(output,{'identity':identity,'rows':results,'status':'running'})
                print(tokens,method,round(float(np.median(times)),3),flush=True)
    finally:
        index.close()
    atomic_json(output,{'identity':identity,'rows':results,'status':'complete'})
    print(output,flush=True)


if __name__=='__main__':
    main()
