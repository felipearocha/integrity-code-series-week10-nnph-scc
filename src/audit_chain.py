"""Immutable SHA-256 hash-linked audit chain — Week 10."""
import hashlib,json,time
from dataclasses import dataclass,field,asdict
from typing import List

@dataclass
class AuditEntry:
    index:int; timestamp:str; run_id:str; inputs:dict; outputs:dict
    model_version:str="week10_v1.0"; prev_hash:str="0"*64
    entry_hash:str=field(default="",init=False)
    def __post_init__(self): self.entry_hash=self._compute_hash()
    def _compute_hash(self):
        payload=json.dumps({k:v for k,v in asdict(self).items() if k!="entry_hash"},sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()
    def verify(self): return self.entry_hash==self._compute_hash()

class AuditChain:
    def __init__(self): self._e:List[AuditEntry]=[]
    def append(self,run_id,inputs,outputs,mv="week10_v1.0"):
        def _s(d):
            import numpy as np; out={}
            for k,v in d.items():
                if isinstance(v,np.ndarray): out[k]=v.tolist()
                elif isinstance(v,(np.integer,)): out[k]=int(v)
                elif isinstance(v,(np.floating,)): out[k]=float(v)
                elif isinstance(v,dict): out[k]=_s(v)
                else: out[k]=v
            return out
        ts=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
        prev=self._e[-1].entry_hash if self._e else "0"*64
        e=AuditEntry(len(self._e),ts,run_id,_s(inputs),_s(outputs),mv,prev)
        self._e.append(e); return e
    def verify_chain(self):
        for k,e in enumerate(self._e):
            if not e.verify(): return False
            if k>0 and e.prev_hash!=self._e[k-1].entry_hash: return False
        return True
    def to_json(self): return json.dumps([asdict(e) for e in self._e],indent=2,sort_keys=True)
    def __len__(self): return len(self._e)
    def __getitem__(self,i): return self._e[i]

_CHAIN=AuditChain()
def log_run(run_id,inputs,outputs): return _CHAIN.append(run_id,inputs,outputs)
def get_chain(): return _CHAIN
