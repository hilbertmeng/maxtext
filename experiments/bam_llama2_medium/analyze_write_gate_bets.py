"""Sequence-paired uncertainty and bidirectional response for the sealed probe."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import stats


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    result = np.empty_like(p)
    result[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return result


def analyze(root, out, draws=20000):
    meta = json.loads((root / 'metadata.json').read_text())
    protocol = meta['protocol']
    heads = protocol['heads']
    shifts = protocol['shifts']
    records = [json.loads(p.read_text()) for p in sorted(root.glob('seq_*.json'))]
    ids = [r['sequence'] for r in records]
    assert len(ids) == len(set(ids)) and all(32 <= i < 128 for i in ids), ids
    assert len(records) >= 2, 'Need paired sequences, not individual tokens as replicates'
    n = len(records)
    delta = np.empty((n, len(heads), len(shifts)))
    gradient = np.empty((n, len(heads)))
    selected = np.empty_like(gradient)
    rounded = np.empty_like(delta)
    gate_change = np.empty_like(delta)
    for i, record in enumerate(records):
        rows = {r['id']: r for r in record['rows']}
        grads = {r['id']: r for r in record['gradients']}
        for control in ['baseline', 'zero_shift_control', 'terminal_control']:
            assert abs(rows[control]['delta']) <= 1e-7, (record['sequence'], control, rows[control])
        for j, head in enumerate(heads):
            gradient[i, j] = grads[head['id']]['derivative']
            selected[i, j] = grads[head['id']]['selected']
            for k, shift in enumerate(shifts):
                row = rows[head['id'] + f'_shift{shift}']
                assert row['clipped'] == 0
                delta[i, j, k] = row['delta']
                rounded[i, j, k] = row['rounded_unchanged']
                gate_change[i, j, k] = -row['gate_removed']
    assert np.isfinite(delta).all() and np.isfinite(gradient).all()
    # Resample whole sequences, preserving correlation across heads and arms.
    rng = np.random.default_rng(9202202)
    weights = rng.multinomial(n, np.full(n, 1 / n), size=draws).astype(np.float64) / n
    boot = (weights @ delta.reshape(n, -1)).reshape(draws, len(heads), len(shifts))
    ci = np.quantile(boot, [.025, .975], axis=0)
    gci = np.quantile(weights @ gradient, [.025, .975], axis=0)
    mean = delta.mean(0)
    lower = shifts.index(-1.)
    upper = shifts.index(1.)
    primary = delta[:, :, lower]
    rawp = stats.ttest_1samp(primary, 0, axis=0).pvalue
    rawp = np.where(np.isnan(rawp) & (np.max(np.abs(primary), axis=0) == 0), 1., rawp)
    adjusted = holm(rawp)
    secant = (delta[:, :, upper] - delta[:, :, lower]) / 2
    error_ci = np.quantile(weights @ (gradient - secant), [.025, .975], axis=0)
    responses = []
    for j, head in enumerate(heads):
        labels = []
        for k in [lower, upper]:
            lo, hi = ci[:, j, k]
            labels.append('increase' if lo > 0 else 'decrease' if hi < 0 else 'uncertain')
        shape = {('decrease', 'increase'): 'lower_better',
                 ('increase', 'decrease'): 'higher_better',
                 ('increase', 'increase'): 'both_worse',
                 ('decrease', 'decrease'): 'both_better'}.get(tuple(labels), 'unresolved')
        responses.append(dict(**head, mean_delta=mean[j].tolist(), ci95=ci[:, j].T.tolist(),
                              gradient_mean=float(gradient[:, j].mean()), gradient_ci95=gci[:, j].tolist(),
                              secant_mean=float(secant[:, j].mean()),
                              gradient_minus_secant_ci95=error_ci[:, j].tolist(),
                              gradient_secant_sequence_correlation=float(np.corrcoef(gradient[:, j], secant[:, j])[0, 1]) if np.std(secant[:, j]) > 0 and np.std(gradient[:, j]) > 0 else None,
                              primary_holm_p=float(adjusted[j]),
                              primary_equivalent=bool(np.max(np.abs(ci[:, j, lower])) < 1e-4),
                              shape_pointwise_ci=shape,
                              selected_tokens=int(selected[:, j].sum()),
                              rounded_fraction=(rounded[:, j].sum(0) / max(1, selected[:, j].sum())).tolist(),
                              actual_gate_change_per_selected=(gate_change[:, j].sum(0) / max(1, selected[:, j].sum())).tolist()))
    groups = {}
    for group in protocol['predictions']:
        members = [j for j, h in enumerate(heads) if group in h['groups']]
        gb = boot[:, members, lower].mean(1)
        groups[group] = dict(heads=[heads[j]['id'] for j in members],
                             primary_mean=float(mean[members, lower].mean()),
                             ci95=np.quantile(gb, [.025, .975]).tolist(),
                             mean_increase_count=int((mean[members, lower] > 0).sum()),
                             holm_increase_count=sum(adjusted[j] < .05 and mean[j, lower] > 0 for j in members),
                             holm_decrease_count=sum(adjusted[j] < .05 and mean[j, lower] < 0 for j in members),
                             equivalent_count=sum(responses[j]['primary_equivalent'] for j in members))
    result = dict(n_sequences=n, sequences=ids, complete=(ids == list(range(32, 128))),
                  shifts=shifts, bootstrap_draws=draws, heads=responses, groups=groups,
                  gradient_secant_head_sign_agreement=float(np.mean(np.sign(gradient.mean(0)) == np.sign(secant.mean(0)))),
                  gradient_secant_head_mean_absolute_error=float(np.mean(np.abs(gradient.mean(0) - secant.mean(0)))),
                  uncertainty='95% sequence bootstrap; primary head tests: paired one-sample t with Holm over 29 heads; curve labels exploratory pointwise CIs')
    out.mkdir(parents=True, exist_ok=True)
    # Convert NumPy scalar booleans/integers in summaries without losing floats.
    (out / 'summary.json').write_text(json.dumps(result, indent=2, default=lambda x: x.item()))
    np.savez_compressed(out / 'paired.npz', delta=delta, gradient=gradient, selected=selected)
    plot(result, out)
    print(json.dumps(dict(n=n, complete=result['complete'], groups=groups), indent=2, default=lambda x: x.item()))


def plot(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(6, 5, figsize=(17, 17), sharex=True)
    shifts = result['shifts']
    order = np.argsort(shifts + [0.])
    for ax, h in zip(axes.flat, result['heads']):
        x = np.array(shifts + [0.])[order]
        y = np.array(h['mean_delta'] + [0.])[order] * 1e4
        interval = np.array(h['ci95'] + [[0., 0.]])[order] * 1e4
        ax.errorbar(x, y, yerr=[y - interval[:, 0], interval[:, 1] - y], fmt='o-', capsize=3)
        ax.plot(x, x * h['gradient_mean'] * 1e4, '--', color='tab:orange', label='gradient prediction')
        ax.axhline(0, color='gray', linewidth=.7)
        ax.set_title(h['id'] + ' | ' + h['shape_pointwise_ci'], fontsize=9)
        ax.set_xlabel('alpha'); ax.set_ylabel('delta loss x 10,000')
    for ax in list(axes.flat)[len(result['heads']):]: ax.axis('off')
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"Single-head conditional write-gate responses; n={result['n_sequences']} paired sequences")
    fig.tight_layout(); fig.savefig(out / 'head_responses.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--draws', type=int, default=20000)
    args = parser.parse_args()
    analyze(args.root, args.out, args.draws)
