"""Read-only own-head Wc spectra for the paired 43k analysis."""
from pathlib import Path
import io, json, os, subprocess, sys, time
import numpy as np
import ml_dtypes
from scipy.linalg import eigh

R = Path(__file__).resolve().parent
os.environ['GOOGLE_AUTH_TOKEN_FOR_TESTING'] = subprocess.check_output(
    ['gcloud', 'auth', 'print-access-token'], text=True).strip()
sys.path.insert(0, json.loads((R / 'prenorm.json').read_text())['worktree'] + '/MaxText')
import head_spectral_diagnostic as hd

for arm in ['prenorm']:
    m = json.loads((R / (arm + '.json')).read_text())
    h = Path(m['local_DATA']) / 'head_spectra'
    h.mkdir(parents=True, exist_ok=True)
    start = time.time()
    flat, identity = hd.restore(m['checkpoints']['43000'], head_only=True)
    assert identity['state_step'] == 43001
    w = np.asarray(flat[('decoder', 'lm_head', 'logits_dense')], ml_dtypes.bfloat16).astype(np.float64)
    gamma = flat[('decoder', 'lm_head', 'decoder_norm', 'scale')]
    beta = flat[('decoder', 'lm_head', 'decoder_norm', 'bias')]
    assert w.shape == (2560, 50304)
    mu = w.mean(1)
    w -= mu[:, None]
    gram = w @ w.T
    del w
    e, u = eigh(gram, driver='evd', check_finite=False)
    assert e.min() > 0
    basis = u[:, :128].astype(np.float32)
    orth = float(np.max(np.abs(basis.T @ basis - np.eye(128))))
    assert orth < 1e-5
    np.save(h / '43000.npy', basis)
    np.savez_compressed(h / '43000.npz', centered_singular=np.sqrt(e), centered_basis=u.astype(np.float32), gamma=gamma, bias=beta, vocabulary_mean=mu)
    b = gamma[:, None] * basis
    b -= b.mean(0)
    control = float(np.sum(b*b) / ((1-1/2560) * np.sum(gamma*gamma)))
    row = dict(arm=arm, **identity, sigma_min=float(np.sqrt(e[0])), sigma_max=float(np.sqrt(e[-1])),
               gamma_RMS=float(np.sqrt(np.mean(gamma*gamma))), bias_RMS=float(np.sqrt(np.mean(beta*beta))),
               gamma_isotropic_control=control, orthogonality_max_error=orth, seconds=time.time()-start)
    (h / '43000.json').write_text(json.dumps(row, indent=2)+'\n')
    subprocess.run(['gsutil', '-q', 'cp', str(h/'43000.npy'), m['own_basis_root']+'/43000.npy'], check=True)
    print('HEAD_SPECTRUM_COMPLETE', json.dumps(row), flush=True)
