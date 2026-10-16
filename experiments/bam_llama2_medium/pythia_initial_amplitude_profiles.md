# Serial Pythia initialization amplitude profiles

This is a random-initialization forward probe, not restored checkpoint statistics or a replay of the actual first training batch. Full D1200/H16/head75/L18, T2048, four fixed saved Pile sequences, parameter seed19. Each sequence's moments are retained. Formal scan is preserved (BAM three-layer block scan, MHA flat scan; Mudd unscanned with full history and original depth-dependent widths). Float32 CPU dense attention replaces the training BF16/Splash/chunk backend; no numerical equivalence to its first bit pattern is claimed. Bypass only the final vocabulary head, which does not feed the layers. Parameter initialization is traced at T16; parameter shapes/RNG draws are sequence-length independent. Dropout0 and noQKNorm.

Source family: /data0/xd/pythia-mediumprop-t2048, parent commit5fcef1f30bbd1534b3ddf75a558f30fc60784688. Executed runner SHA256:7b58fd8e77643895b91bfcfd8cb3c6788fbc1161edc51883657d2964df05dfcf. Fixed cohort /data0/xd/bam_diagnostics/cohorts/pile-eval-t2048-seed9876-n128-v1/pile_eval_cohort.npz; token ids SHA256:2a181ff8907f5152b014e664dd142d1fe90097d36c49d122666650aacb9ba56c.8192 valid tokens/2136 distinct ids. Each local CPU forward, including initialization and compilation, took33–58s; no checkpoint transfer or diagnostic TPU required. No training health flags enabled.

Artifacts: [four-panel figure](/data0/xd/bam_diagnostics/pythia-init/full-depth-amplitudes/initial-amplitude-ratios.png), [PDF](/data0/xd/bam_diagnostics/pythia-init/full-depth-amplitudes/initial-amplitude-ratios.pdf), [all18-layer CSV](/data0/xd/bam_diagnostics/pythia-init/full-depth-amplitudes/layers.csv), [summary/theory/error checks](/data0/xd/bam_diagnostics/pythia-init/full-depth-amplitudes/summary.json). Per-sequence raw reduced moments: ARM.json in the same directory.

## Definitions and analytic checks

Let E be token embedding, x_l the incoming residual, a_l the post-W_O attention update, and f_l the post-W_2 MLP update. Report RMS(a_l)/RMS(E), RMS(f_l)/RMS(E), and also the more locally meaningful RMS(a_l)/RMS(x_l), RMS(f_l)/RMS(x_l+a_l). The latter does not manufacture an apparent deep-layer difference merely by choosing a different embedding denominator. All pooled RMS values are sqrt(mean of per-sequence squared RMS), not averages of RMS.

For Pythia s=sqrt(2/(5D))=.01825742 and output std t=2/(L sqrt(D))=.00320750. Thus W_O's expected RMS gain is sqrt(D)t=2/L=.111111. Original initialization gives s=t=.006 and gain .207846. E std is .006 on repaired/full-original, s on other Pythia arms. Semantic zeros, gate biases and norm scales remain their actual initialized values.

Condition on a fixed normalized input x. A fresh independent Gaussian W_1 makes each preactivation Gaussian with variance q=s²*||x||²=D*s²*N²; the residual itself need not be Gaussian. The formula below gives the conditional expected second moment over W_1 and independent W_2. The table uses pooled N², a small further approximation when per-token normalized norms vary. For exact GELU g(z)=z Phi(z), Z~N(0,q):

    F(q)=E[g(Z)^2]
        =q*(1/4+asin(q/(1+q))/(2*pi))
         +q^2/(pi*(1+q)*sqrt(1+2*q))
    RMS(MLP output) ≈ sqrt(m*t^2*F(q)).

This uses the uncentered second moment, not the centered GELU variance: GELU has a nonzero mean. For deep normalized inputs N≈1, all-original/fix MLP amplitude ratio is .5288, while attention W_O gain ratio is1.8706 at fixed head amplitude. Therefore the all-original attention/MLP ratio is expected to increase about3.54x even before accounting for different M/attention correlations. The measured layer18 ratio increase is4.80x. The theoretical direction is clear; the remaining difference is not assigned to a specific cause by this probe.

For first-layer attention, a uniform causal approximation can be calculated from token identity counts, without measuring attention tensors. If c_t(v) counts token v in a prefix of length t, S=mean_t sum_v(c_t(v)/t)^2=.01464696 on this cohort. For MHA, RMS(a_1)≈(2/L)*sqrt(D)*s*N_E*sqrt(S), where N_E=std(E)/sqrt(std(E)^2+epsilon_LN).

For BAM, first calculate seed factors from their actual initialization: q_u=D*s_BAM²*(std(E)*input_scale)²; q_address=256*s_BAM²*F(q_u). Then seed RMS≈sqrt(16)*.1*sqrt(q_u/(q_u+eps_write))*sqrt(q_address/(q_address+eps_write)). Static V key has std1/sqrt(32), dynamic V replaces standard W_V, and static O is zero. Dynamic gates start.05, read scale.1 and read-key epsilon1e-4.

M-shift is correlated across neighboring values: it reduces individual-token RMS but does not reduce a long prefix mean by the same1/sqrt(2) factor. At gate.5, sum of shifted values = sum raw values + .5*first − .5*last. Recompute token-count concentration with these boundary weights, then add the small dynamic local-O variance approximation. Uniform attention, independent random kernel directions and scalar RMS-normalization approximations are stated assumptions, not exact model identities.

Unconditional first-layer predictions (known initializer/configuration and cohort token counts only; no measured activation fitted):

| Arm | Attention theory | Attention measured | MLP theory | MLP measured |
|---|---:|---:|---:|---:|
| MHA | 0.008380 | 0.008627 | 0.083563 | 0.083001 |
| Mudd | 0.008380 | 0.008663 | 0.059480 | 0.060589 |
| BAM 未修复 | 0.002488 | 0.002487 | 0.091861 | 0.092264 |
| BAM 初始化修复 | 0.000254 | 0.000254 | 0.080846 | 0.081048 |
| BAM 全原初始化 | 0.000474 | 0.000474 | 0.043425 | 0.043378 |
| BAM 大 embedding＋补偿 | 0.000254 | 0.000254 | 0.091832 | 0.092169 |
| BAM 大 embedding、不补偿（仅诊断） | 0.000944 | 0.000950 | 0.091836 | 0.092197 |

Separately, condition on each layer's measured normalized-input RMS and use only initializer std/width/GELU formula to predict MLP output; condition on measured pre-W_O head RMS and use sqrt(D)t for attention output. These are conditional checks, not unconditional forecasts. Across all126 layer/arm combinations, per-arm median absolute MLP prediction errors are0.57–1.85%; worst layer3.06%. The worst output-projection error is3.58%, consistent with finite random projections. Exact measured residual-energy accounting, retaining signed cross terms, closes within3.92e-7 relative error. Independence is not assumed in this identity.

## Layer profiles

Each cell is layer1 → layer18; all layers are in the CSV and figure.

| Arm | Attention / embedding | MLP / embedding | Attention / incoming residual | MLP / post-attention residual |
|---|---:|---:|---:|---:|
| MHA | 0.4737 → 3.1773 | 4.5570 → 4.8012 | 0.4737 → 0.1426 | 4.1180 → 0.2134 |
| Mudd | 0.4756 → 3.3006 | 3.3265 → 5.7558 | 0.4756 → 0.1435 | 3.0016 → 0.2479 |
| BAM 未修复 | 0.1365 → 8.9146 | 5.0656 → 5.0245 | 0.1365 → 0.2977 | 5.0185 → 0.1617 |
| BAM 初始化修复 | 0.0424 → 17.9240 | 13.5402 → 15.2492 | 0.0424 → 0.2251 | 13.5276 → 0.1877 |
| BAM 全原初始化 | 0.0792 → 45.8356 | 7.2469 → 8.1307 | 0.0792 → 0.4023 | 7.2239 → 0.0667 |
| BAM 大 embedding＋补偿 | 0.0140 → 5.8341 | 5.0604 → 4.9988 | 0.0140 → 0.2221 | 5.0598 → 0.1865 |
| BAM 大 embedding、不补偿（仅诊断） | 0.0522 → 5.8982 | 5.0619 → 5.0181 | 0.0522 → 0.2247 | 5.0548 → 0.1875 |

The large-uncompensated arm is a forward-only intervention, not a launched training run. Mudd retains the original extra router budget and growing MLP widths; its first-layer width2432 grows to7168. This explains its increasing MLP amplitude rather than treating it as unexplained noise. Its zero second router kernel and last-history bias select the latest history at initialization.

## Interpretation and limits

1. The first BAM attention is weak only at the beginning of depth: repaired attention/embedding rises.0424→1.1944→3.1361 in layers1/2/3, and reaches17.924 by layer18. Attention/incoming-residual at layer18 is.225, so do not extrapolate first-layer weakness to all layers.
2. Small-embedding repaired and large-embedding compensated arms have almost the same deeper absolute branch amplitudes and local ratios. At layer18 attention/residual is.2251 vs.2221 and MLP/post-attention-residual .1877 vs.1865. The roughly3x difference in output/embedding ratios mostly reflects their denominators. First-forward amplitude alone therefore does NOT explain the observed terminal loss gap+.01136. Relative embedding Adam updates and seed-input learning remain candidates, not demonstrated causes.
3. All-original initialization produces a much more attention-heavy deep stack. At layer18 attention/MLP is5.637, versus repaired1.175, unfixed1.774, large-compensated1.167, MHA.662 and Mudd.573. This is a concrete initialization-balance difference worth considering, not a new loss-ranking score. All-original still trains better than unfixed; no single amplitude ratio orders all terminal losses.
4. Deep attention outputs have strong token-common components: per-sequence token-mean attention RMS/total RMS is about.90–.98 at layer18. Gaussian/GELU shared-feature correlations accumulate through residual depth. Thus an independent-token1/sqrt(T) attention approximation is unsuitable for deeper layers. The exact identity E||sum alpha_i V_i||² includes all i,j cross-covariances. RMS is not a measurement of useful information or trained expressivity.

## First MLP / embedding as a candidate mechanism

The three arms with repaired BAM-specific initialization rank consistently: first-layer MLP/embedding13.540 (repaired),7.247 (all-original),5.060 (large+compensated), versus terminal losses2.361978,2.367093,2.373340. This is a meaningful association supported by two different contrasts, not an established isolated first-layer effect. All-original also changes standard kernels throughout depth; large+compensated changes LayerNorm response and relative embedding/seed learning.

At fixed initialization widths/stds, ignoring tiny first attention,

    ratio(s_E) = sqrt(m)*sigma_W2*sqrt(F(q))/s_E,
    q = D*sigma_W1^2*s_E^2/(s_E^2+epsilon_LN).

Predicted repaired13.47 vs large+compensated5.03, measured13.54/5.06. LayerNorm makes MLP amplitude less sensitive to embedding std than the linear identity skip. Both already have MLP-dominated first outputs: the direct embedding squared-norm term is about.54%,1.87%,3.76% of the respective first output squared norm. Measured first-layer post-attention/MLP cosine is only.0010–.0023, so the independent-energy approximation is numerically adequate here. These are norm terms, not fractions of retained token information: the MLP and seed M both remain functions of embedding.

There is a backward counterpart that can also be derived without checkpoint measurement. For an embedding perturbation orthogonal to its centered feature direction and the all-ones direction, the LayerNorm Jacobian gain is1/sqrt(s_E^2+epsilon). Fresh Gaussian W1 makes the perturbation's preactivation independent of the original preactivation. Hence expected RMS tangent gain through the first MLP branch is

    J_gain = sqrt(D*m)*sigma_W1*sigma_W2*sqrt(G(q))/sqrt(s_E^2+epsilon),
    G(q) = E[(GELU'(Z))^2]
         = 1/4 + asin(q/(1+q))/(2*pi)
           + q/(pi*(1+q)*sqrt(1+2*q))
           + q/(2*pi*(1+2*q)^(3/2)).

Predicted gains: repaired13.93, all-original7.30, large+compensated5.21. Scalar numerical Gaussian quadrature verified both F and G formulas to numerical precision. This shows the forward ratio is related to embedding-Jacobian conditioning, not only direct residual composition. It does NOT imply Adam gives proportionally larger absolute updates: normalization largely removes gradient magnitude, while relative update/embedding scale remains different.

Most targeted improvement candidate, not launched: keep large embedding+compensated seed, and form first-block output c*E + attention + MLP(LN(E+attention)), c≈.37373. All attention/MLP computation inputs, initialized parameters, initial M and embedding parameter scale remain unchanged; only the first embedding identity skip is attenuated. It matches the repaired first MLP/direct-skip ratio without globally amplifying later residuals. This specifically tests the identity-skip mechanism; lack of recovery would not rule out LayerNorm/MLP conditioning or relative embedding learning as contributors. The default first-block MLP has no M write, so no first-layer MLP-write side effect needs compensating. No new training was started from this analysis.

## Reproduction

From the family worktree, for each ARM in mha,mudd,unfixed,bam_fix,all_original,large_comp,large_uncomp:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=MaxText:MaxText/tests  taskset -c 0-15 /data0/xd/conda/envs/maxtext-cpu/bin/python  experiments/bam_llama2_medium/probe_initial_residual_amplitudes.py  --arm ARM --cohort /data0/xd/bam_diagnostics/cohorts/pile-eval-t2048-seed9876-n128-v1/pile_eval_cohort.npz  --output /data0/xd/bam_diagnostics/pythia-init/full-depth-amplitudes/ARM.json
python3 experiments/bam_llama2_medium/report_initial_residual_amplitudes.py  /data0/xd/bam_diagnostics/pythia-init/full-depth-amplitudes
```

The report needs NumPy/Matplotlib only; the runner uses the pinned MaxText CPU environment. The initial tiny smoke used the same real dimensions/depth at T128 to verify scan collection layout; formal profile results above use T2048. No trained parameters were loaded or modified.
