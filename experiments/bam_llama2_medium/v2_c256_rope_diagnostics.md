# V2 C256 RoPE and LocalQK diagnostics

Checkpoint: `BamLlama2MediumV2/checkpoints/13250/items`. Pile eval is shuffled;
the structural/gate cohort contains 128 unique 2048-token sequences
(`0b414befd8c15850`), and exact loss ablations use its first 32 sequences.

Scripts:

- `v2_c256_rope_gate_diagnostics.py` / `MaxText/bam_gate_diagnostics.py`
- `v2_c256_rope_structure_diagnostics.py`
- `v2_c256_block_rope_loss_ablation.py`
- `v2_c256_compressed_local_qk_diagnostics.py`

Artifacts are under `/data0/xd/bam_diagnostics/v2_c256_rope_20260821/`.

## Corrected coordinate interpretation

`bam_abs_v_compression_dim=8` compresses only the fetched-M cache view. The
cross-layer M remains 32x32; LocalQK injects the column answer into head dims
`0:32` and the row answer into `32:64`. Thus the prior PartialRoPE24 run
(`0:40` NoPE, `40:64` RoPE, then unrotated LocalQK) did **not** rotate an
unused/uninjected tail and is not footprint-aligned. Its matched-MHA result is
still a valid arbitrary coordinate-split control.

## LocalQK gates versus realized signal

`Qg/Kg-r/c` are mean effective gate strengths (gate times absolute normalized
head mix). `Q/Knorm-r/c` are realized BAM-read norm divided by standard Q/K
norm. Layer 0 has nonzero gates but zero signal because incoming M is zero.

|L|Qg-r|Qg-c|Kg-r|Kg-c|Qnorm-c|Qnorm-r|Knorm-c|Knorm-r|
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
|0|0.004078|0.004067|0.004061|0.004069|0.000|0.000|0.000|0.000|
|1|0.002570|0.008598|0.001905|0.007003|0.118|0.034|0.054|0.012|
|2|0.001863|0.003150|0.001494|0.002205|0.064|0.061|0.031|0.028|
|3|0.002502|0.003339|0.001912|0.002564|0.311|0.090|0.174|0.059|
|4|0.001018|0.003244|0.001087|0.002031|0.199|0.045|0.113|0.025|
|5|0.002111|0.001411|0.001909|0.001238|0.109|0.089|0.081|0.064|
|6|0.005635|0.009364|0.003512|0.009939|0.936|0.222|0.760|0.112|
|7|0.001268|0.008767|0.001029|0.006279|0.427|0.062|0.210|0.035|
|8|0.001382|0.007406|0.001220|0.004919|0.538|0.086|0.366|0.060|
|9|0.003053|0.007066|0.002985|0.005871|0.619|0.139|0.634|0.109|
|10|0.001356|0.006876|0.001552|0.005985|0.443|0.083|0.329|0.068|
|11|0.000973|0.003813|0.001163|0.002362|0.189|0.070|0.135|0.052|
|12|0.000768|0.002926|0.001008|0.003863|0.303|0.050|0.232|0.050|
|13|0.000754|0.005390|0.000702|0.003833|0.498|0.065|0.482|0.044|
|14|0.000763|0.001374|0.000677|0.001372|0.146|0.068|0.137|0.046|
|15|0.000615|0.004744|0.000605|0.003097|0.701|0.068|0.781|0.051|
|16|0.000570|0.003742|0.000508|0.002117|0.370|0.075|0.269|0.048|
|17|0.000780|0.005047|0.000933|0.003377|0.487|0.105|0.438|0.108|
|18|0.000689|0.004964|0.000934|0.003004|0.726|0.100|0.631|0.108|
|19|0.000829|0.003545|0.000953|0.002482|0.428|0.129|0.323|0.113|
|20|0.000813|0.003033|0.000918|0.002248|0.469|0.139|0.372|0.127|
|21|0.000810|0.002684|0.000623|0.001981|0.469|0.128|0.387|0.088|
|22|0.000739|0.002330|0.000608|0.001864|0.383|0.154|0.318|0.096|
|23|0.000874|0.001115|0.001016|0.001056|0.253|0.153|0.150|0.139|

Early/mid/late realized Q-column ratios are `.271/.430/.448`, while the
effective gates decline `.00524/.00495/.00331`; Q-row similarly changes
`.075/.079/.123` while its gate declines `.00263/.00121/.00076`. K behaves the
same way. Gate trends therefore do not determine actual injected magnitude;
M and standard-Q/K scale dynamics dominate later layers. Column reads are
usually much stronger than row reads, but the gap narrows late. Q exceeds K.

The BAM signal in dims `40:64` is not small: mean tail/front norm ratios are
about `.231` for Q and `.209` for K, and at layer 23 they reach `.58/.84`.

## Attention-logit attribution

Each entry is the mean per-head RMS of that term divided by the mean per-head
RMS of `std·std`, using all sampled distances.

|L|std×bam|bam×std|bam×bam|
|--:|--:|--:|--:|
|0|0.0000|0.0000|0.0000|
|1|0.0296|0.0428|0.0127|
|2|0.0174|0.0305|0.0034|
|3|0.0581|0.0892|0.0474|
|4|0.0244|0.0475|0.0231|
|5|0.0121|0.0198|0.0112|
|6|0.0759|0.1478|0.2072|
|7|0.0502|0.1022|0.0630|
|8|0.0678|0.1124|0.0933|
|9|0.1062|0.1397|0.1569|
|10|0.0690|0.1041|0.1020|
|11|0.0368|0.0535|0.0254|
|12|0.0288|0.0418|0.0365|
|13|0.0769|0.0941|0.1099|
|14|0.0262|0.0342|0.0228|
|15|0.1272|0.1536|0.1984|
|16|0.0426|0.0696|0.0475|
|17|0.1318|0.1602|0.1410|
|18|0.1449|0.1801|0.1854|
|19|0.1009|0.1289|0.0694|
|20|0.1102|0.1366|0.1132|
|21|0.1057|0.1275|0.1078|
|22|0.0731|0.0962|0.0740|
|23|0.0878|0.1078|0.0408|

|distance|std×bam|bam×std|bam×bam|signed bam×bam|
|---:|---:|---:|---:|---:|
|0|.0473|.0661|.1896|+.1449|
|1–4|.0517|.0718|.1061|+.0636|
|5–16|.0613|.0855|.0932|+.0390|
|17–64|.0713|.0995|.0989|+.0297|
|65–256|.0775|.1089|.0982|+.0211|
|257–1024|.0630|.0891|.0754|+.0110|
|1025+|.0545|.0755|.0644|+.0077|

The cross terms have near-zero signed means. `bam·bam` instead adds a strong
positive diagonal/local bias that decays with distance.

## Block-RoPE counterfactual

Block-RoPE separately rotates dims `0:40` and `40:64`. The intervention is
split into standard-Q/K-only, BAM-signal-only, and both.

|changed component|KL λ=.1|output rel. norm λ=.1|KL λ=1|output rel. norm λ=1|
|---|---:|---:|---:|---:|
|standard Q/K|.06384|.1662|1.9590|.6469|
|BAM LocalQK|.00587|.0265|.2401|.1358|
|both|.07064|.1757|2.2481|.6792|

The corresponding mean single-layer loss deltas are `+.000934`, `+.000189`,
`+.001148`, and `+.06098` for standard λ=.1, BAM λ=.1, both λ=.1, and both
λ=1. The hard full switch is dominated by generic MHA disruption (worst:
layer 1 `+.2708`, layer 9 `+.1846`, layer 15 `+.1808`). BAM-only λ=.1 is much
smaller but peaks at layer 15 (`+.00164`) and layer 6 (`+.00060`). As posed,
hard block-RoPE is not a credible architecture; a retraining experiment would
need a matched MHA control and a smoother/learned transition.

## LocalQK from the V-compressed M view

The counterfactual uses the learned per-layer encoder `E_v:32→8`, maps each
current column key to least-squares coordinates `pinv(E_v) r`, and reads
`M E_v`. The Direct variant pads the 8-D row answer into dims `32:40`. Cosine
and relative error compare the resulting LocalQK contribution with the current
full-M read. KL uses 128 sequences; exact loss uses the first 32.

|L|Q col cos/err|K col cos/err|Q row cos/err|K row cos/err|KL|dloss|
|--:|---:|---:|---:|---:|---:|---:|
|0|0 / 0|0 / 0|0 / 0|0 / 0|0|+0.00000|
|1|.702 / .795|.694 / .803|.032 / 1.085|.028 / 1.081|.008|-0.00011|
|2|.437 / .903|.513 / .880|-.053 / 1.054|-.070 / 1.070|.005|-0.00022|
|3|.953 / .799|.937 / .809|-.028 / 1.077|-.026 / 1.075|.112|+0.00195|
|4|.818 / .809|.758 / .867|.061 / 1.039|.055 / 1.044|.092|+0.00222|
|5|.911 / .770|.923 / .782|-.045 / 1.101|-.057 / 1.127|.096|+0.00071|
|6|.924 / .765|.875 / .784|-.061 / 1.101|-.056 / 1.096|.974|+0.07635|
|7|.684 / .800|.604 / .836|-.017 / 1.096|-.012 / 1.094|.097|+0.00181|
|8|.735 / .793|.891 / .746|.021 / 1.125|.012 / 1.123|.199|+0.00804|
|9|.778 / .801|.886 / .804|-.011 / 1.099|-.019 / 1.097|.718|+0.08686|
|10|.417 / .914|.382 / .936|-.035 / 1.085|-.032 / 1.063|.226|+0.00555|
|11|.407 / .907|.629 / .853|.001 / 1.118|.007 / 1.080|.058|+0.00064|
|12|.519 / .893|.507 / .881|.017 / 1.094|.031 / 1.075|.193|+0.00295|
|13|.372 / .940|.300 / .965|-.056 / 1.099|-.060 / 1.095|.332|+0.01032|
|14|.391 / .926|.639 / .858|.027 / 1.086|.028 / 1.065|.077|+0.00038|
|15|-.068 / 1.030|.024 / 1.004|-.012 / 1.066|-.020 / 1.059|.554|+0.09990|
|16|.091 / 1.051|.676 / .850|.029 / 1.055|.038 / 1.058|.157|+0.02305|
|17|.112 / 1.014|.311 / .957|-.013 / 1.067|.024 / 1.048|.248|+0.00365|
|18|.259 / .967|.155 / .989|-.005 / 1.089|-.004 / 1.076|.414|+0.00431|
|19|.430 / .908|.630 / .818|.036 / 1.080|.052 / 1.056|.112|+0.00715|
|20|.355 / .937|.316 / .953|.037 / 1.088|.044 / 1.087|.204|+0.00153|
|21|.375 / .931|.275 / .960|.010 / 1.115|-.007 / 1.131|.205|+0.00075|
|22|.442 / .903|.224 / .981|-.004 / 1.127|-.016 / 1.151|.170|+0.00066|
|23|.439 / .901|.451 / .892|-.094 / 1.164|-.085 / 1.147|.082|+0.00028|

Direct all-layer loss is `+0.63399`. The largest one-layer harms are layers 15
(`+.09990`), 9 (`+.08686`), and 6 (`+.07635`). The compressed column read is
partly retained but degrades with depth; the direct row interface is nearly
orthogonal to the current 32-D row contribution. This result therefore mixes
information loss with a coordinate-interface change. A second control decodes
the 8-D row result through `pinv(E_v)` before comparison.

### Pseudoinverse-decoded control

This isolates the best linear subspace reconstruction available from the
already learned encoder.

|variant/depth|Q row cos/err|K row cos/err|Q col cos|K col cos|KL|
|---|---:|---:|---:|---:|---:|
|Direct early|-.016 / 1.079|-.020 / 1.084|.776|.758|.198|
|Decoded early|.479 / .876|.485 / .874|.776|.758|.191|
|Direct middle|-.006 / 1.097|-.007 / 1.082|.444|.532|.295|
|Decoded middle|.430 / .912|.434 / .900|.447|.533|.291|
|Direct late|-.001 / 1.098|.006 / 1.094|.313|.380|.199|
|Decoded late|.397 / .927|.410 / .918|.314|.380|.196|

Decoding fixes the gross row-coordinate mismatch, but only recovers cosine
`.40-.48` and barely changes the all-layer loss delta (`+.63399` to `+.63140`).
The same layers remain dominant: 15 `+.09984`, 9 `+.08527`, and 6 `+.07562`.
Thus the failure is not primarily the Direct padding interface. The learned
8-D fetched-read subspace discards information heavily used by LocalQK,
especially in the column path and in layers 6/9/15. Reusing this compressed M
view for LocalQK is not a healthy checkpoint-local replacement.

Both variants are fixed-checkpoint counterfactuals: they measure current
circuit reliance, not the result after retraining a compressed LocalQK
architecture. Native 8-D reads would make each M contraction about 4x smaller;
a small learned row decoder would add ordinary projection work, but the loss
result says that speed hypothesis is not worth formal training without a less
destructive subspace.
