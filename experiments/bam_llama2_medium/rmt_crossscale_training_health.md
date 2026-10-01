# MediumProp versus XLProp dynamic RMT B training health

Question: why does the XLProp all-M-read-pre-norm dynamic RMT lose its advantage
while MediumProp retains it? Diagnose training health and distinguish an optimizer
problem, saturated/collapsed representations, and output calibration. Do not equate
lower loss than a failed parent with a healthy trajectory.

Training runtime shared562bfb9397c5e5f5314c19b3f109b2feba1a0596:
- Medium RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNorm,
  checkpoint5400 (40% of13500), D1200/head16x75/M48x75,C8,R256.
- XL RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNorm,
  checkpoint17500 (35% of50000), D1920/head20x96/M60x96,C10,R384.
These are near-progress, not exactly matched progress; the early checkpoints have
already been pruned. Historical TB supplies the temporal evidence.

Diagnostic worktree /data0/xd/rmt-crossscale-health, codex/rmt-crossscale-health,
created from the exact training runtime. Main keeps this report only.
Artifacts /data0/xd/bam_diagnostics/rmt-crossscale-health-20261001/.
TPUs owned by this diagnosis: EW4a v6e-1
xd-v6e-1-rmt-health-medium-261001 and xd-v6e-1-rmt-health-xl-261001.
Retained STANDARD/FLEX compilers and ongoing training TPUs are untouched.

Initial evidence: RMT-B / Mudd MHA-gain ratio stays near1.55 on Medium but
falls1.38->.98 on XL. XL BAM catches B because B's extra benefit disappears;
BAM's own ratio also declines. Existing TB shows no renewed late global clipping.
Raw-gradient medians grow.245->.469 on XL but decline.390->.257 on Medium.
XL unembedding static/dynamic RMS rise1.64/1.58 at5000 to7.70/7.14 at17500;
Medium near matched progress is1.73/1.29. XL unembedding gates approach1.
OutputHead explicitly omits vector output RMSNorm for RMT; final full-M norm
precedes both reads, so large readout amplitude can reach logits. It can also be
an innocuous gauge change compensated by logits weights; measure it, do not assume.

Probe runner MaxText/rmt_checkpoint_health.py: read-only parameter restore,
no optimizer updates/saves; same32 fixed-seed, shuffled TruePile4096 sequences
with full hashes and per-sequence outputs. Four sequences additionally produce
per-parameter/per-layer gradients and activation/cotangent scalar taps.
Masked-vocabulary-centered logit RMS, entropy/confidence, temperature-sweep CE
and its scale derivative directly test output calibration, avoiding softmax-invariant
logit offsets. Fixed-position matrix participation rank/top-energy power estimate
and attention entropy/concentration distinguish representation collapse from simple
amplitude growth. Parameter norms and gradient shares distinguish scale growth,
vanishing or dominating parameter groups. Probe-off/on CE and gradient identity
is a required focused CPU gate; actual checkpoint baseline/probe CE is checked too.

Missing training metrics already exposed: summed unembedding RMS; vocabulary-
centered logits RMS/entropy/calibration; grouped gradient shares/update-to-weight
ratios; raw carry M scale and dM/M; matrix rank/coherence; attention entropy and
NoPE/RoPE logit contribution balance. Determine diagnostic value before selecting
a small permanent metric set. No new formal training recipe is prescribed yet.
