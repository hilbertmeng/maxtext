"""Atomic snapshots from separate G-only backward and primal-only forward runs.

Pairing proves matching declared identities and row coverage, not numerical
fidelity of the independently compiled forward. Native gradient/forward and
H^T G reconstruction gates remain mandatory at the caller/global reviewer.
"""
import hashlib
import gzip
import json
from pathlib import Path
import shutil
import tempfile
import threading
import numpy as np
from wo_boundary_capture import load_boundary


class SplitWriter:
    def __init__(self, directory, step, context, layers=range(6), disk_reserve=1<<30, allow_identical_forward_repeats=False):
        if not isinstance(context,dict) or not all(isinstance(context.get(k),str) and context[k] for k in ('run','state','input','rng')):
            raise ValueError('Explicit run/state/input/RNG identities required')
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.step=int(step);self.context=dict(context);self.layers=set(layers)
        self.allow_repeats=bool(allow_identical_forward_repeats);self.identical_forward_repeats=0
        self.reserve=int(disk_reserve)
        if self.reserve<0:raise ValueError('Negative disk reserve')
        self.lock=threading.Lock();self.errors=[]

    def forward(self, layer, a, o, h, rows, valid):
        self._record('primal',layer,dict(AV=a,localO=o,H=h),rows,valid)

    def backward(self, layer, g, rows, valid):
        self._record('cotangent',layer,dict(G=g),rows,valid)

    def _record(self, phase, layer, arrays, rows, valid):
        try:
            layer=int(np.asarray(layer))
            if layer not in self.layers:return
            arrays={k:np.asarray(v) for k,v in arrays.items()}
            rows,valid=np.asarray(rows),np.asarray(valid)
            if rows.ndim!=2 or not np.issubdtype(rows.dtype,np.integer) or valid.dtype!=np.bool_ or valid.shape!=rows.shape:
                raise ValueError('Invalid row grid or valid mask')
            ids=rows[:,0]
            if len(set(ids.tolist()))!=len(ids) or not np.array_equal(rows,np.broadcast_to(ids[:,None],rows.shape)):
                raise ValueError('Duplicate or nonconstant global rows')
            if phase=='primal' and (arrays['H'].ndim!=4 or any(v.shape!=arrays['H'].shape for v in arrays.values())):
                raise ValueError('AV/O/H shape mismatch')
            if phase=='cotangent' and arrays['G'].ndim!=3:raise ValueError('G must be BTD')
            if any(v.shape[:2]!=rows.shape or not np.isfinite(v).all() for v in arrays.values()):
                raise ValueError('Boundary shape mismatch or nonfinite tensor')
            arrays.update(global_rows=ids,valid=valid)
            key=f'layer-{layer:02d}-rows-'+hashlib.sha256(ids.tobytes()).hexdigest()[:16]
            parent=self.directory/phase;parent.mkdir(exist_ok=True)
            with self.lock:
                target=parent/key
                if target.exists():
                    if phase=='primal' and self.allow_repeats:
                        old=json.loads((target/'complete.json').read_text())
                        if old['context']==self.context and old['step']==self.step and all(
                            name in old['files'] and old['files'][name]['shape']==list(a.shape) and old['files'][name]['dtype']==str(a.dtype) and old['files'][name]['sha256']==hashlib.sha256(a.tobytes()).hexdigest()
                            for name,a in arrays.items()):
                            self.identical_forward_repeats+=1
                            return
                    raise ValueError('Duplicate snapshot or differing remat repeat')
                if shutil.disk_usage(parent).free<sum(v.nbytes for v in arrays.values())+self.reserve:
                    raise RuntimeError('Insufficient disk for split snapshot')
                temp=Path(tempfile.mkdtemp(prefix='.snapshot-',dir=parent))
                try:
                    meta=dict(step=self.step,layer=layer,phase=phase,context=self.context,global_rows=ids.tolist(),files={})
                    for name,a in arrays.items():
                        bf16=str(a.dtype)=='bfloat16'
                        np.save(temp/(name+'.npy'),a.view(np.uint16) if bf16 else a,allow_pickle=False)
                        meta['files'][name]=dict(shape=list(a.shape),dtype=str(a.dtype),bf16_bits=bf16,sha256=hashlib.sha256(a.tobytes()).hexdigest())
                    (temp/'complete.json').write_text(json.dumps(meta,indent=2)+'\n')
                    temp.rename(target)
                except BaseException:
                    shutil.rmtree(temp)
                    raise
        except Exception as exc:
            with self.lock:self.errors.append(repr(exc))
            raise


def paired_snapshots(directory, step, context, layers, expected_rows):
    """Yield validated raw pairs; reject absent, unmatched or overlapping shards."""
    directory=Path(directory);layers=set(layers);expected_rows=sorted(expected_rows)
    paths={phase:{p.name:p for p in (directory/phase).glob('layer-*')} for phase in ('primal','cotangent')}
    if not paths['primal'] or paths['primal'].keys()!=paths['cotangent'].keys():
        raise ValueError('Missing or unmatched split snapshots')
    seen={i:[] for i in layers};pairs=[]
    for key,p in sorted(paths['primal'].items()):
        pm,a=load_boundary(p);gm,g=load_boundary(paths['cotangent'][key])
        for phase,meta in [('primal',pm),('cotangent',gm)]:
            if meta['step']!=step or meta['context']!=context or meta['phase']!=phase:
                raise ValueError('Snapshot identity mismatch')
        layer=pm['layer']
        if layer not in layers or layer!=gm['layer'] or pm['global_rows']!=gm['global_rows']:
            raise ValueError('Layer/row identity mismatch')
        if pm['global_rows']!=a['global_rows'].tolist() or gm['global_rows']!=g['global_rows'].tolist():
            raise ValueError('Manifest rows disagree with stored rows')
        if not np.array_equal(a['global_rows'],g['global_rows']) or not np.array_equal(a['valid'],g['valid']):
            raise ValueError('Different row grids or target masks across executions')
        if a['H'].shape[:2]!=g['G'].shape[:2]:raise ValueError('Different boundary batch/time shapes')
        seen[layer].extend(a['global_rows'].tolist());a.update(G=g['G']);pairs.append((pm,a))
    if any(sorted(v)!=expected_rows for v in seen.values()):
        raise ValueError('Missing or overlapping row coverage')
    return pairs


def save_failed_hg_diagnostic(output, pairs, layers, expected_rows, chunk_tokens=256,
                              disk_reserve=1<<30):
    """Retain ONLY local H^T G matrices after a failed snapshot fidelity gate.

    Caller must first use paired_snapshots to verify context, checksums and masks.
    Matrices require a four-worker sum before comparison with replicated native
    W_O. They are never scientific AV/O partials or a completed capture report.
    """
    if chunk_tokens <= 0 or disk_reserve < 0:
        raise ValueError('Invalid diagnostic resource bound')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    records=[]
    for layer in layers:
        selected=[(m,v) for m,v in pairs if m['layer']==layer]
        if not selected:raise ValueError('Missing diagnostic layer')
        rows=np.concatenate([np.asarray(v['global_rows']) for _,v in selected])
        if len(set(rows.tolist()))!=len(rows) or set(rows.tolist())!=set(expected_rows):
            raise ValueError('Missing or duplicate diagnostic rows')
        total=None
        for _,v in selected:
            h,g=v['H'],v['G']
            if h.ndim!=4 or g.ndim!=3 or h.shape[:2]!=g.shape[:2]:
                raise ValueError('Diagnostic H/G shape mismatch')
            shape=(*h.shape[2:],g.shape[-1])
            if total is None:
                needed=int(np.prod(shape))*8
                if shutil.disk_usage(output).free < disk_reserve+needed:
                    raise OSError('Diagnostic disk reserve')
                total=np.zeros(shape,np.float64)
            if total.shape!=shape:raise ValueError('Diagnostic partition shape mismatch')
            for row in range(h.shape[0]):
                for start in range(0,h.shape[1],chunk_tokens):
                    hh=np.asarray(h[row,start:start+chunk_tokens],np.float64)
                    gg=np.asarray(g[row,start:start+chunk_tokens],np.float64)
                    if not np.isfinite(hh).all() or not np.isfinite(gg).all():
                        raise ValueError('Nonfinite diagnostic H/G')
                    total += (hh.reshape(len(hh),-1).T @ gg).reshape(shape)
        target=output/f'diagnostic-layer-{layer:02d}-hg.npy'
        with tempfile.NamedTemporaryFile(dir=output,delete=False) as f:
            temp=Path(f.name)
            try:
                np.save(f,total,allow_pickle=False)
                f.flush()
                temp.replace(target)
            finally:
                temp.unlink(missing_ok=True)
        saved=np.load(target,allow_pickle=False,mmap_mode='r')
        if not np.array_equal(saved,total):raise ValueError('Diagnostic readback mismatch')
        records.append(dict(layer=int(layer),file=target.name,global_rows=rows.tolist(),
                            shape=list(total.shape),dtype=str(total.dtype)))
    report=dict(diagnostic_only=True,complete=False,scientific_review_complete=False,
                requires_global_worker_merge=True,scope='fullbatch_H_only',
                interpretation='Failed fidelity data; no AV/O attribution',matrices=records)
    (output/'failed-hg-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def execute_recorded(fn, args, output, phase):
    """Save compiler text from the very compiled object dispatched below.

    Dispatch is not completion: the caller must block and validate the result.
    Text availability/size limits are explicit; missing HLO never implies equality.
    """
    output=Path(output);compiled=fn.lower(*args).compile()
    meta=dict(stage='compiled.as_text',same_compiled_object_dispatch=False,
              execution_complete=False,available=False,saved=False)
    try:
        raw=compiled.as_text().encode()
        meta.update(available=True,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        if len(raw)<=64*(1<<20):
            target=output/(phase+'-optimized-hlo.txt.gz')
            with gzip.open(target,'wb') as handle:handle.write(raw)
            meta.update(saved=True,file=target.name)
        else:meta['omitted_reason']='Compiler text exceeds 64MiB bound'
    except Exception as exc:
        meta['error']=type(exc).__name__+': '+str(exc)
    target=output/(phase+'-compiler-evidence.json')
    target.write_text(json.dumps(meta,indent=2)+'\n')
    result=compiled(*args)
    meta['same_compiled_object_dispatch']=True
    target.write_text(json.dumps(meta,indent=2)+'\n')
    return result
