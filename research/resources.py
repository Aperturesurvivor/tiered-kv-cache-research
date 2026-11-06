"""Policy-local host observations, kept separate from cumulative run peaks."""
import threading
import psutil


def logical_store_reads(store):
    """Tensor-read bytes across an overlay's entire immutable-parent chain.

These counters exclude checksum scans and do not distinguish page-cache hits
from physical disk reads. Each object owns only its local file-read counter.
"""
    total = 0
    while store is not None:
        total += store.read_bytes
        store = store.prefix
    return total


class HostObservation:
    def __init__(self,scope='whole policy call, including initialization, query/decode and checkpoint; stock includes prefill'):
        self.scope=scope
        self.process=psutil.Process()
        self.stop=threading.Event()
        self.samples=[]
        self.error=None
        self.metrics={}

    def sample(self):
        self.samples.append(self.process.memory_info().rss)

    def monitor(self):
        while not self.stop.wait(.1):
            try:
                self.sample()
            except Exception as exc:
                self.error=exc
                return

    def io(self):
        return self.process.io_counters()._asdict() if hasattr(self.process,'io_counters') else {}

    def __enter__(self):
        self.sample()
        self.start_io=self.io()
        self.start_memory=psutil.virtual_memory()._asdict()
        self.thread=threading.Thread(target=self.monitor,daemon=True)
        self.thread.start()
        return self

    def __exit__(self,*args):
        self.stop.set()
        self.thread.join(timeout=3)
        self.sample()
        end_io=self.io()
        self.metrics={'host_rss_start_bytes':self.samples[0],
            'host_rss_end_bytes':self.samples[-1],
            'host_rss_sampled_peak_bytes':max(self.samples),
            'host_rss_samples':len(self.samples),'host_sampling_interval_s':.1,
            'host_observation_scope':self.scope,
            'process_io_delta':{k:end_io[k]-v for k,v in self.start_io.items()},
            'system_memory_start':self.start_memory,
            'system_memory_end':psutil.virtual_memory()._asdict()}
        if self.error and args[0] is None:
            raise RuntimeError('host observation failed') from self.error
