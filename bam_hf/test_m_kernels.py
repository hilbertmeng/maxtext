"""Unit test: Triton BAM read/write kernels against the eager reference (strided inputs, fp32/bf16).

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python bam_hf/test_m_kernels.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bam_m_kernels import bam_decode_read, bam_read, bam_write  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def rms(x, eps):
  f = x.float()
  return f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)


def rope(x, pos, max_ts=10000.):
  d = x.shape[-1]
  ts = max_ts ** (2 * torch.arange(d // 2, device=x.device, dtype=torch.float32) / d)
  ang = pos.float()[:, None, None] / ts
  a, b = x.float().chunk(2, -1)
  return torch.cat((a * torch.cos(ang) - b * torch.sin(ang), b * torch.cos(ang) + a * torch.sin(ang)), -1)


def ref_read(m, keys, gates, s, std, pos, eps=1e-4, scale=0.1, div=0.99, qk_cols=72, q_scale=96 ** -0.5):
  out = []
  for a in range(4):
    key = keys[:, min(a, 2)]
    x = (scale * torch.sigmoid(gates[:, a].float()))[..., None] * rms(key, eps)
    x = x + (1.1 * torch.sigmoid(gates[:, 4 + a].float()) / div)[..., None] * s[a].float().T
    y = torch.einsum('tnv,tkv->tnk', x, m.float())
    if a < 2 and qk_cols < y.shape[-1]:
      y = torch.cat((y[..., :qk_cols], rope(std[:, a], pos)), -1) * (q_scale if a == 0 else 1.)
    out.append(y)
  return torch.stack(out, 1)


def ref_write(m, groups, eps=1e-6):
  out = m.float()
  for c, g, a in groups:
    out = out + torch.einsum('tnk,tnv->tkv', torch.sigmoid(g.float())[..., None] * rms(c, eps), rms(a, eps))
  return out


def main():
  torch.manual_seed(0)
  n, k, v = 20, 96, 40
  for t, dtype, tol in ((37, torch.float32, 1e-4), (37, torch.bfloat16, 3e-2),          # decode kernel (T < 64)
                        (300, torch.float32, 1e-4), (300, torch.bfloat16, 3e-2)):       # stacked prefill kernel
    dev = dict(device='cuda', dtype=dtype)
    m = torch.randn(t, k, v, **dev)
    up = torch.randn(t, 3 * n * v + 32, **dev)                    # padded fused output, sliced
    keys = up[:, :3 * n * v].unflatten(-1, (3, n, v))
    xs = torch.randn(t, 1984, **dev)
    gates = xs[:, 1360:1520].unflatten(-1, (8, n))
    s = torch.randn(4, v, n, **dev) * 0.3
    std = xs[:, :2 * n * 24].unflatten(-1, (2, n, 24))
    pos = torch.randint(0, 4096, (t,), device='cuda')
    nop = (m, m, m[:, :, 0], m, m, m[:, :, 0], m, 0, False, 1e-6)
    y = bam_read(m, keys, gates, s, std, pos, 1e-4, 0.1, 0.99, 72, 96 ** -0.5, 10000., True, *nop)[0].float()
    r = ref_read(m, keys, gates, s, std, pos)
    raw = bam_read(m, keys, gates, s, std, pos, 1e-4, 0.1, 0.99, 72, 96 ** -0.5, 10000., False, *nop)[0].float()
    r_raw = ref_read(m, keys, gates, s, std, pos, qk_cols=96, q_scale=1.)
    e_raw = ((raw - r_raw).norm() / r_raw.norm()).item()
    e_read = ((y - r).norm() / r.norm()).item()
    if t < 64:     # decode read with the key up-projection fused
      hid = xs[:, 960:1360]
      w_up = torch.randn(3 * n * v + 32, 400, **dev) * 400 ** -0.5
      b_up = torch.randn(3 * n * v + 32, **dev) * 0.1
      keys_d = (F.linear(F.gelu(hid.float(), approximate='tanh'), w_up.float(), b_up.float()))[:, :3 * n * v]
      pos2 = torch.randint(0, 4096, (1, t), device='cuda').expand(1, t)
      yd = bam_decode_read(hid, w_up, b_up, m, gates, s, std, pos2, 1e-4, 0.1, 0.99, 72, 96 ** -0.5, 10000.,
                           m, m, m[:, :, 0], m, m, m[:, :, 0], m, 0, False, 1e-6)[0].float()
      rd = ref_read(m, keys_d.unflatten(-1, (3, n, v)), gates, s, std, pos2[0])
      e_dec = ((yd - rd).norm() / rd.norm()).item()
      print(f'T={t} {dtype}: decode_read (fused up) rel {e_dec:.1e}')
      assert e_dec < tol, e_dec
    c1 = torch.randn(t, n, k, **dev); g1 = xs[:, 1760:1780]; a1 = torch.randn(t, n * v + 32, **dev)[:, :n * v].unflatten(-1, (n, v))
    c2 = torch.randn(t, 1920, **dev).unflatten(-1, (n, k)); g2 = torch.randn(t, 404, **dev)[:, :n]; a2 = torch.randn(t, n, v, **dev)
    loc = torch.randn(t, n, k, **dev)
    w1 = bam_write(m, c1, c1, g1, a1, c1, g1, a1, False, False, 1e-6).float()
    w2 = bam_write(m, c1, loc, g1, a1, c2, g2, a2, True, True, 1e-6).float()
    r1, r2 = ref_write(m, [(c1, g1, a1)]), ref_write(m, [(c1 + loc, g1, a1), (c2, g2, a2)])
    e1, e2 = ((w1 - r1).norm() / r1.norm()).item(), ((w2 - r2).norm() / r2.norm()).item()
    print(f'T={t} {dtype}: read(rope) rel {e_read:.1e}  read(raw) rel {e_raw:.1e}  write1 rel {e1:.1e}  write2 rel {e2:.1e}')
    assert max(e_read, e_raw, e1, e2) < tol, (e_read, e_raw, e1, e2)
  # Deferred write applied inside the read: equals write-then-read, and returns the written M.
  for t, dtype, tol in ((3, torch.float32, 1e-4), (3, torch.bfloat16, 3e-2), (37, torch.float32, 1e-4),
                        (300, torch.float32, 1e-4), (300, torch.bfloat16, 3e-2)):
    dev = dict(device='cuda', dtype=dtype)
    m = torch.randn(t, k, v, **dev)
    keys = torch.randn(t, 3, n, v, **dev); xs = torch.randn(t, 1984, **dev); gates = xs[:, 1360:1520].unflatten(-1, (8, n))
    s = torch.randn(4, v, n, **dev) * 0.3; std = xs[:, :960].unflatten(-1, (2, n, 24)); pos = torch.arange(t, device='cuda')
    attn, loc = torch.randn(t, n, k, **dev), torch.randn(t, n, k, **dev); g1 = torch.randn(t, n, **dev); a1 = torch.randn(t, n, v, **dev)
    c2 = torch.randn(t, n, k, **dev); g2 = torch.randn(t, n, **dev); a2 = torch.randn(t, n, v, **dev)
    for pend, has_local in ((1, True), (2, False), (2, True)):
      groups = [((attn + loc) if has_local else attn, g1, a1)] + ([(c2, g2, a2)] if pend == 2 else [])
      m_ref = ref_write(m, groups).to(dtype)
      pargs = (attn, loc, g1, a1, c2, g2, a2, pend, has_local, 1e-6)
      y, m_new = bam_read(m, keys, gates, s, std, pos, 1e-4, 0.1, 0.99, 72, 96 ** -0.5, 10000., t < 64, *pargs)
      r = ref_read(m_ref, keys, gates, s, std, pos) if t < 64 else ref_read(m_ref, keys, gates, s, std, pos, qk_cols=96, q_scale=1.)
      e_y = ((y.float() - r).norm() / r.norm()).item(); e_m = ((m_new.float() - m_ref.float()).norm() / m_ref.float().norm()).item()
      msg = f'T={t} {dtype} pending={pend} local={has_local}: read rel {e_y:.1e} M rel {e_m:.1e}'
      if t <= 4:
        hid = xs[:, 960:1360]; w_up = torch.randn(3 * n * v, 400, **dev) * 400 ** -0.5; b_up = torch.randn(3 * n * v, **dev) * .1
        kd = F.linear(F.gelu(hid.float(), approximate='tanh'), w_up.float(), b_up.float()).unflatten(-1, (3, n, v))
        yd, md = bam_decode_read(hid, w_up, b_up, m, gates, s, std, pos[None], 1e-4, 0.1, 0.99, 72, 96 ** -0.5, 10000., *pargs)
        rd = ref_read(m_ref, kd, gates, s, std, pos)
        e_d = ((yd.float() - rd).norm() / rd.norm()).item(); e_dm = ((md.float() - m_ref.float()).norm() / m_ref.float().norm()).item()
        msg += f' | decode_read rel {e_d:.1e} M rel {e_dm:.1e}'
        assert max(e_d, e_dm) < tol, (e_d, e_dm)
      print(msg)
      assert max(e_y, e_m) < tol, (e_y, e_m)
  print('M_KERNELS_OK')


if __name__ == '__main__':
  main()
