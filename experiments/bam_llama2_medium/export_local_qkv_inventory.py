"""Export the local-QKV study ledger without importing JAX or executing exp.py.

Usage: python3 experiments/bam_llama2_medium/export_local_qkv_inventory.py
Outputs Markdown on stdout. Runtime hashes describe historical experiments, not
the current implementation; inherited configuration remains authoritative in exp.py.
"""
import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'MaxText/exp.py'
text = SOURCE.read_text()
lines = text.splitlines()
tree = ast.parse(text)
EXTRA = {
    'BamLlama2MediumV2',
    'BamLlama2MediumV2C256ScanAotControl',
    'BamLlama2MediumV2C256ScanAotCleanControl',
    'BamLlama2MediumV2C256FullMPostReadV8PartialRoPESeparateQKPairedInit',
    'BamLlama2XLHead16x128V2C256PartialRoPE',
    'BamLlama2MediumV2C256LocalFetchC8LocalVLLFScan',
    'BamLlama2MediumV2C256LocalFetchC8LocalVSharedRankGateScan',
    'BamLlama2MediumV2C256ScanAotControlLocalQKRank2',
    'BamLlama2MediumV2C256ScanAotControlLocalQKRank2SharedRankGate',
    'BamLlama2MediumV2C256Paired40LocalQKRank2',
    'BamLlama2MediumV2C256Paired40LocalQKRank4',
    'BamLlama2MediumV2C256Paired40LocalQKRank2CurrentControl',
    'BamLlama2MediumV2C256Paired40LocalQKRank2SharedRankGate',
    'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2',
    'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8LocalVLLF',
    'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2SharedRankGate',
    'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2PairedOrthV32SharedRankGate',
    'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2PairedIdentityV32SharedRankGate',
}
selected = [n for n in tree.body if isinstance(n, ast.ClassDef) and (
    n.name.startswith(('BamMediumIndependentLLF', 'BamXLIndependentLLF',
                       'BamMediumPaired40Rank2')) or n.name in EXTRA)]
print('# Local QKV experiment inventory\n')
print('Generated from `MaxText/exp.py` by `export_local_qkv_inventory.py`. '
      'This is a snapshot of the ledger, not proof that every historical branch '
      'has the same runtime semantics today. See [synthesis](local_qkv_routing_review.md).\n')
print(f'{len(selected)} classes; training, reproduction, and speed-only controls are distinguished by their recorded notes.\n')
for n in selected:
    block = lines[n.lineno - 1:n.end_lineno]
    comments = [s.strip()[2:].strip() for s in block if s.strip().startswith('# ')]
    hashes = re.findall(r'(?:code_commit:\s*|^)([0-9a-f]{7,40})(?=[; ,])', '\n'.join(comments), re.M)
    print(f'## {n.name}\n')
    print(f'Parent: {", ".join(ast.unparse(b) for b in n.bases)}. '
          f'[Source](../../MaxText/exp.py#L{n.lineno}). '
          f'Runtime: {", ".join(dict.fromkeys(hashes)) or "see source notes / parent; not inferred"}.\n')
    print('```python')
    for s in n.body:
        if isinstance(s, ast.Assign) and not any(isinstance(t, ast.Name) and t.id in
                                                ('model_name', 'jax_cache_dir') for t in s.targets):
            print(ast.unparse(s))
    print('```\n')
    print('Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):\n')
    for c in comments:
        print(f'- {c}')
    print()
