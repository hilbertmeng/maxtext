"""Identity cotangent tap and atomic, partition-aware frozen-boundary capture.

Experimental instrumentation, NOT proof of native TPU graph fidelity. Callers
must still gate the actual full gradient and every W_O leaf against controls.
The callback writes raw local operands; local matrix work runs after backward.
"""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import threading
import numpy as np


def identity_boundary_tap(writer, layer_limit=None):
    import jax

    @jax.custom_vjp
    def tap(y, av, local_o, h, row_ids, valid, layer):
        return y

    def forward(y, av, local_o, h, row_ids, valid, layer):
        return y, (av, local_o, h, row_ids, valid, layer)

    def backward(saved, g):
        av, local_o, h, row_ids, valid, layer = saved
        # No dummy parameter tree, matrix contractions, dtype conversion, or
        # layer-selection conditional on the real output/cotangent path.
        def capture(_):
            jax.debug.callback(writer, layer, av, local_o, h, g, row_ids, valid,
                               partitioned=True, ordered=False)
            return 0
        if layer_limit is None:
            capture(None)
        else:
            # Select only the side effect; return the true cotangent unchanged.
            jax.lax.cond((layer >= 0) & (layer < layer_limit), capture, lambda _:0, None)
        return g, None, None, None, None, None, None

    tap.defvjp(forward, backward)
    return tap


class BoundaryWriter:
    """One writer per worker/step; reject duplicates and missing row identities."""
    def __init__(self, directory, step, layers=range(6), disk_reserve=1<<30):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.step, self.layers = int(step), set(layers)
        self.disk_reserve = int(disk_reserve)
        self.lock = threading.Lock()
        self.errors = []

    def __call__(self, layer, av, local_o, h, g, row_ids, valid):
        try:
            self._write(int(np.asarray(layer)), av, local_o, h, g, row_ids, valid)
        except Exception as exc:
            with self.lock:
                self.errors.append(repr(exc))
            raise

    def _write(self, layer, av, local_o, h, g, row_ids, valid):
        if layer not in self.layers:
            return
        av, local_o, h, g, row_ids, valid = map(np.asarray, (av, local_o, h, g, row_ids, valid))
        if av.ndim != 4 or av.shape != local_o.shape or av.shape != h.shape:
            raise ValueError('AV/O/H partition shape mismatch')
        b, t = av.shape[:2]
        if g.ndim != 3 or g.shape[:2] != (b,t) or row_ids.shape != (b,t) or valid.shape != (b,t):
            raise ValueError('Boundary partition row/time mismatch')
        rows = row_ids[:,0]
        if not np.issubdtype(rows.dtype, np.integer) or not np.array_equal(row_ids, np.broadcast_to(rows[:,None],(b,t))):
            raise ValueError('Global row identities must be constant across time')
        if len(set(rows.tolist())) != b or not all(np.isfinite(x).all() for x in (av,local_o,h,g)):
            raise ValueError('Duplicate rows or nonfinite boundaries')
        arrays = dict(AV=av, localO=local_o, H=h, G=g, global_rows=rows, valid=valid)
        key = hashlib.sha256(rows.tobytes()).hexdigest()[:16]
        target = self.directory / f'layer-{layer:02d}-rows-{key}'
        with self.lock:
            if target.exists():
                raise ValueError('Duplicate boundary partition; refusing double counting')
            if shutil.disk_usage(self.directory).free < sum(x.nbytes for x in arrays.values()) + self.disk_reserve:
                raise RuntimeError('Insufficient disk for boundary capture')
            temp = Path(tempfile.mkdtemp(prefix='.boundary-',dir=self.directory))
            try:
                meta = dict(step=self.step,layer=layer,files={})
                for name, array in arrays.items():
                    # Preserve BF16 bit patterns: NumPy reload would otherwise
                    # treat ml_dtypes BF16 as opaque void16. Record explicit dtype.
                    bf16 = str(array.dtype)=='bfloat16'
                    storage = array.view(np.uint16) if bf16 else array
                    np.save(temp/(name+'.npy'),storage,allow_pickle=False)
                    meta['files'][name]=dict(shape=list(array.shape),dtype=str(array.dtype),
                        sha256=hashlib.sha256(array.tobytes()).hexdigest(),bf16_bits=bf16)
                (temp/'complete.json').write_text(json.dumps(meta,indent=2)+'\n')
                temp.rename(target)
            except BaseException:
                shutil.rmtree(temp)
                raise


def load_boundary(directory):
    directory = Path(directory)
    meta = json.loads((directory/'complete.json').read_text())
    arrays = {}
    for name, info in meta['files'].items():
        a = np.load(directory/(name+'.npy'),mmap_mode='r',allow_pickle=False)
        if info['bf16_bits']:
            import ml_dtypes
            a = a.view(ml_dtypes.bfloat16)
        if list(a.shape)!=info['shape'] or str(a.dtype)!=info['dtype'] or hashlib.sha256(a.tobytes()).hexdigest()!=info['sha256']:
            raise ValueError('Boundary file identity mismatch')
        arrays[name]=a
    return meta,arrays
