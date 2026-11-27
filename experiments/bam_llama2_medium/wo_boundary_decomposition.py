"""Host-side partition sums from frozen AV/O/H and the same output cotangent.

No model execution or normalization: G must already include the actual global
loss normalization. Merge matrices before computing norms (cross-shard and
cross-token cancellation matters). H is the actual BF16-rounded sum, not A+O
reconstructed in FP32. This separates rounding residual from a branch split.
"""
import numpy as np


def partition_sums(av, local_o, h, g, global_rows, valid, token=1211, sequence=9,
                   chunk_tokens=256):
    av, local_o, h, g = [np.asarray(x, dtype=np.float32) for x in (av, local_o, h, g)]
    rows, valid = np.asarray(global_rows), np.asarray(valid, dtype=bool)
    if av.shape != local_o.shape or av.shape != h.shape or av.ndim != 4:
        raise ValueError('AV/O/H must share [local_batch,time,heads,head_dim]')
    b, t, n, q = av.shape
    if g.ndim != 3 or g.shape[:2] != (b, t) or valid.shape != (b, t):
        raise ValueError('G/valid do not match activation batch and time')
    if rows.shape != (b,) or len(set(rows.tolist())) != b:
        raise ValueError('Unique global row identities are required')
    if chunk_tokens <= 0 or not 0 <= token < t:
        raise ValueError('Invalid chunk size or selected token')
    if not all(np.isfinite(x).all() for x in (av, local_o, h, g)):
        raise ValueError('Nonfinite boundary data')
    d = g.shape[-1]
    # No batch-size scaling here: every shard uses the SAME globally normalized G.
    parts = np.zeros((3, 3, n * q, d), np.float64)
    amplitude = np.zeros((3, 6), np.float64)
    for row, global_row in enumerate(rows):
        for start in range(0, t, chunk_tokens):
            end = min(start + chunk_tokens, t)
            a, o, actual_h = [x[row, start:end].reshape(end-start, n*q)
                              for x in (av, local_o, h)]
            cg = g[row, start:end]
            selected = global_row == sequence
            masks = (np.ones(end-start, bool),
                     np.full(end-start, selected),
                     selected & (np.arange(start, end) == token))
            for scope, mask in enumerate(masks):
                for branch, x in enumerate((a, o, actual_h)):
                    # Convert only a chunk to FP64, avoiding full-boundary copies.
                    parts[scope, branch] += x[mask].astype(np.float64).T @ cg[mask].astype(np.float64)
                mask = mask & valid[row, start:end]
                aa, oo, hh = [x[mask].astype(np.float64) for x in (a, o, actual_h)]
                amplitude[scope] += [np.sum(aa*aa), np.sum(oo*oo), np.sum(hh*hh),
                                     np.sum(aa*oo), np.sum((hh-aa-oo)**2), aa.size]
    return dict(parts=parts.reshape(3, 3, n, q, d), amplitude=amplitude,
                global_rows=rows.copy())


def merge_sums(partitions, expected_rows):
    if not partitions:
        raise ValueError('No partitions')
    rows = np.concatenate([p['global_rows'] for p in partitions])
    if len(set(rows.tolist())) != len(rows) or set(rows.tolist()) != set(expected_rows):
        raise ValueError('Missing or duplicate global rows')
    return dict(parts=sum(p['parts'] for p in partitions),
                amplitude=sum(p['amplitude'] for p in partitions), global_rows=rows)


def describe(sums):
    result = []
    for scope, (parts, amp) in enumerate(zip(sums['parts'], sums['amplitude'])):
        a, o, h = parts
        a_norm, o_norm, h_norm = [float(np.linalg.norm(x)) for x in (a, o, h)]
        total = a + o
        total_sq = float(np.sum(total*total))
        result.append(dict(scope=('fullbatch', 'sequence9', 'token9_1211')[scope],
            AV_gradient_norm=a_norm, O_gradient_norm=o_norm,
            H_gradient_norm=h_norm, sum_gradient_norm=float(np.sqrt(total_sq)),
            gradient_cosine=float(np.sum(a*o)/max(a_norm*o_norm, 1e-30)),
            O_projection_fraction=float(np.sum(o*total)/max(total_sq, 1e-30)),
            rounding_gradient_relative_error=float(np.linalg.norm(h-total)/max(h_norm, 1e-30)),
            sum_forward_relative_squared_error=float(amp[4]/max(amp[2], 1e-30))))
    return result
