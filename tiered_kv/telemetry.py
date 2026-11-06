import importlib.metadata
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import threading
import time
import psutil
import torch
from .store import atomic_json

GIB=1024**3


def resolve_device():
    requested=os.environ.get('TIERED_DEVICE','').strip().lower()
    if requested=='cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('TIERED_DEVICE=cuda but CUDA unavailable: timed CPU fallback prohibited')
        return 'cuda'
    if requested=='mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError('TIERED_DEVICE=mps but MPS unavailable: timed CPU fallback prohibited')
        return 'mps'
    if requested:
        raise RuntimeError("TIERED_DEVICE must be 'cuda' or 'mps', got %r"%requested)
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    raise RuntimeError('no CUDA or MPS accelerator: timed CPU fallback prohibited')


DEVICE=resolve_device()


def runtime_identity():
    """Bind correctness evidence and resumed timings to the actual runtime."""
    identity={'device':DEVICE,'os':platform.platform(),'machine':platform.machine(),
              'python':platform.python_version(),'torch':str(torch.__version__),
              'transformers':importlib.metadata.version('transformers'),
              'numpy':importlib.metadata.version('numpy'),
              'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads()}
    if DEVICE=='cuda':
        identity.update(gpu=torch.cuda.get_device_name(),
            capability=list(torch.cuda.get_device_capability()),cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
            tf32=torch.backends.cuda.matmul.allow_tf32,
            fp16_reduced=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            math_sdp_reduced=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
            flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
            mem_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
            math_sdp=torch.backends.cuda.math_sdp_enabled(),
            cudnn_sdp=torch.backends.cuda.cudnn_sdp_enabled())
    return identity


def device_synchronize():
    if DEVICE=='cuda':
        torch.cuda.synchronize()
    elif DEVICE=='mps':
        torch.mps.synchronize()
    else:
        raise RuntimeError('no accelerator device: timed CPU fallback prohibited')


def device_empty_cache():
    if DEVICE=='cuda':
        torch.cuda.empty_cache()
    elif DEVICE=='mps':
        torch.mps.empty_cache()


def accel_allocated():
    if DEVICE=='cuda':
        return torch.cuda.memory_allocated()
    if DEVICE=='mps' and torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory()
    return 0


def accel_reserved():
    if DEVICE=='cuda':
        return torch.cuda.memory_reserved()
    if DEVICE=='mps' and torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory()
    return 0


class ResourceAbort(RuntimeError):
    pass


def disk_preflight(path='.'):
    result=subprocess.run(['/bin/df','-h',str(path)],capture_output=True,text=True,check=True)
    if shutil.disk_usage(path).free<5*10**9:
        raise ResourceAbort('free disk below 5 GB reserve: '+result.stdout.strip())
    return result.stdout.strip()


def pressure_level():
    # kern.memorystatus_vm_pressure_level is macOS-only. On other platforms the
    # guard runs psutil-only (swap growth / RSS / disk); pressure reads 0 so the
    # pressure-warning abort never fires there, everything else still does.
    if platform.system()!='Darwin':
        return 0
    proc=subprocess.run(['/usr/sbin/sysctl','-n','kern.memorystatus_vm_pressure_level'],capture_output=True,text=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip())
    return int(proc.stdout)


def sample_resources(path='.'):
    pressure=pressure_level()
    return {'time':time.time(),'rss':psutil.Process().memory_info().rss,
            'available':psutil.virtual_memory().available,'swap_used':psutil.swap_memory().used,
            'disk_free':shutil.disk_usage(path).free,'pressure':pressure,
            'mps_allocated':accel_allocated(),  # accelerator memory; key name kept stable for the results schema
            'mps_driver':accel_reserved()}


def evaluate_limits(sample,baseline,consecutive):
    consecutive=consecutive+1 if sample['pressure']>=2 else 0
    if consecutive>=3:
        raise ResourceAbort('three consecutive one-second memory pressure warnings')
    if sample['swap_used']-baseline['swap_used']>512*1024**2:
        raise ResourceAbort('swap growth exceeds 512 MiB')
    if sample['rss']>8*GIB:
        raise ResourceAbort('process RSS exceeds 8 GiB')
    if sample['disk_free']<5*10**9:
        raise ResourceAbort('free disk below 5 GB reserve')
    return consecutive


class ResourceGuard:
    """A one-second sampler; telemetry failure also stops work (fail closed)."""
    def __init__(self,path='.',sink=None):
        disk_preflight(path)
        self.path,self.sink=path,sink
        self.baseline=sample_resources(path)
        self.last=self.baseline
        self.peaks=self.baseline.copy()
        self.min_available=self.baseline['available']
        self.error=None
        self.consecutive=0
        self.stop_event=threading.Event()
        self.thread=threading.Thread(target=self._monitor,daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def _monitor(self):
        while not self.stop_event.wait(1):
            try:
                self.record(sample_resources(self.path))
            except Exception as exc:
                self.error=ResourceAbort(f'resource monitor: {exc}')
                return

    def record(self,sample):
        self.last=sample
        self.consecutive=evaluate_limits(sample,self.baseline,self.consecutive)
        for key in ('rss','mps_allocated','mps_driver','swap_used','pressure'):
            self.peaks[key]=max(self.peaks[key],sample[key])
        self.min_available=min(self.min_available,sample['available'])
        if self.sink:
            self.sink(sample)

    def check(self):
        if self.error:
            raise self.error

    def __exit__(self,*args):
        self.stop_event.set()
        self.thread.join(timeout=3)
        if args[0] is None:
            self.record(sample_resources(self.path))
            self.check()

    def metrics(self):
        return {**{'peak_'+key:self.peaks[key] for key in ('rss','mps_allocated','mps_driver')},
                'system_min_available':self.min_available,'swap_delta':self.peaks['swap_used']-self.baseline['swap_used'],
                'pressure_max':self.peaks['pressure']}


def doctor(output='results/doctor.json'):
    result={'gate':1,'python':platform.python_version(),'machine':platform.machine(),'macOS':platform.mac_ver()[0],
            'device':DEVICE,'cuda_available':torch.cuda.is_available(),
            'mps_built':torch.backends.mps.is_built(),'mps_available':torch.backends.mps.is_available(),'checks':{},'errors':[]}
    def check(name,fn):
        try:
            result['checks'][name]=fn()
        except Exception as exc:
            result['errors'].append(f'{name}: {type(exc).__name__}: {exc}')
    def assert_environment():
        if platform.python_version_tuple()[:2] not in (('3','12'),('3','13')):
            raise RuntimeError('requires Python 3.12 or 3.13')
        if DEVICE=='mps':
            if platform.machine()!='arm64':
                raise RuntimeError('MPS requires ARM64')
            if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') not in (None,'0'):
                raise RuntimeError('MPS CPU fallback must be disabled')
        required={'HF_HOME':str(Path('artifacts/hf').resolve()),'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'1'}
        if platform.system()=='Darwin':
            required['VECLIB_MAXIMUM_THREADS']='1'
        for key,value in required.items():
            if os.environ.get(key)!=value:
                raise RuntimeError(f'{key} must be {value}')
        return True
    def pins():
        versions={}
        for line in Path('requirements.in').read_text().splitlines():
            if '==' in line and not line.startswith('#'):
                name,version=line.split('==')
                versions[name]=importlib.metadata.version(name)
                if versions[name]!=version:
                    raise RuntimeError(f'{name}: {versions[name]} != {version}')
        return versions
    def pip_check():
        import sys
        proc=subprocess.run([sys.executable,'-m','pip','check','--disable-pip-version-check'],capture_output=True,text=True)
        ours={line.split('==')[0].lower().replace('-','_') for line in Path('requirements.in').read_text().splitlines() if '==' in line and not line.startswith('#')}
        # Lines look like: "<requirer> <ver> has requirement <req>, but you have <x>."
        # Only fail when the requirer is one of OUR packages; a third-party
        # package's unsatisfied requirement on our (validated) pins is not our problem.
        bad=[]
        for line in proc.stdout.splitlines():
            m=re.match(r'(\S+)\s+\S+\s+has requirement',line)
            if m and m.group(1).lower().replace('-','_') in ours:
                bad.append(line)
        if bad:
            raise RuntimeError('pip check conflicts in our packages: '+'; '.join(bad[:5]))
        return 'ok (third-party conflicts ignored)' if proc.returncode else proc.stdout.strip()
    def accel_op():
        q=torch.randn(1,4,2,16,device=DEVICE,dtype=torch.float16)
        linear=torch.nn.Linear(16,16,device=DEVICE,dtype=torch.float16)
        out=torch.nn.functional.scaled_dot_product_attention(q,q,q,dropout_p=0.)
        out=linear(out)
        device_synchronize()
        if not bool(torch.isfinite(out).all()):
            raise RuntimeError('nonfinite fp16 operation')
        return list(out.shape)
    def native_accel():
        if DEVICE=='mps':
            import ctypes
            metal=ctypes.CDLL('/System/Library/Frameworks/Metal.framework/Metal')
            metal.MTLCreateSystemDefaultDevice.restype=ctypes.c_void_p
            if not metal.MTLCreateSystemDefaultDevice():
                raise RuntimeError('native Metal returns no default GPU device')
            return True
        if DEVICE=='cuda':
            if torch.cuda.device_count()<1:
                raise RuntimeError('CUDA reports no devices')
            return torch.cuda.get_device_name(0)
        raise RuntimeError('no accelerator device')
    check('environment',assert_environment)
    check('pins',pins)
    check('pip_check',pip_check)
    check('disk',disk_preflight)
    check('system_memory',lambda:psutil.virtual_memory()._asdict())
    check('process_memory',lambda:psutil.Process().memory_info()._asdict())
    check('swap_memory',lambda:psutil.swap_memory()._asdict())
    check('pressure_level',pressure_level)
    check('native_accel_device',native_accel)
    check('accel_fp16_attention_linear',accel_op)
    result['status']='passed' if not result['errors'] and DEVICE in ('cuda','mps') else 'failed'
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    atomic_json(output,result)
    return result
