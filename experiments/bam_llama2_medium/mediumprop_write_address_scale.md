# MediumProp write-address norm scales

Runtime worktree `/data0/xd/mediumprop-write-address-scale`, branch `codex/mediumprop-write-address-scale`.
Parent: `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75TruePile`.

Two runs append `WriteAddressNormScaleTruePile` / `WriteAddressNormScalePerHeadTruePile` to the parent prefix before `TruePile`.
Both use the bare strongest 2:1/independent-VO/strict-negative-diagonal/full-NoPE parent, without read-side gamma.
The already-trained write-content-scale experiment instead included read gamma=.5; it is a related result, not an isolated content/address comparison.

Each of the embedding, 17 attention and 17 private-MLP live writers independently learns an address multiplier AFTER RMS normalization:
`deltaM = sum_n sigmoid(z_n) norm(c_n) [gamma_n * norm(a_n)]^T`.
Coordinate gamma is `[16,32]`, head gamma is `[16]` broadcast over address coordinates; head scaling is algebraically equivalent on either outer-product factor.
Both initializers are 1, preserving all old initial parameters and the parent's initial function. Initializing writes to .5 would also perturb all later M consumers, unlike dynamic-localO-only read gamma=.5.
Last-layer attention and private-MLP writes remain absent. MLP widths stay 2510/1128/tail3904.

Parameters: parent432112192; coordinate432130112 (+17920=.012444 W_Q); head432112752 (+560=.000389 W_Q). Negligible extras are not deducted from MLP.
Scale parameters inherit existing weight-decay exemption and generic training-health weight/raw-gradient norms. No internal activation or new BAM-specific statistics.
Tuned Splash, v8t fused read/write, VMEM48MiB, T4096, batch16/chip, scan17 single layers plus tail, full remat, CP200/latest2, TB rotation500, plan13500.

Native reference and Pallas four-factor adapter use an explicit `AddressScaledWrite` type; coordinate sizes never determine the scaled side. Each backward tile emits a FP32 scale-gradient partial, summed over tokens/batch/tiles; replicated scale gradients must agree under batch sharding. The existing three-factor path is preserved. The reusable four-factor content implementation is taken from verified runtime cf663b3; its unrelated attention/read/config changes are not imported.

Infrastructure audit: fresh main c113bb8f, runtime files and rotating-writer flow inherited; inspect main again before sealing. Historical parent model options differ only by the intended scale flag/granularity. CPU launch gate includes exact budgets, parent initialization identity, all scale gradients live, native/Pallas complete-model gradients, tied-coordinate/head gradient sums, batch-sharding sum, shared BAM regression and summary-writer tests.

Before launch bet (R uses matched MHA reference): coordinate terminal1.005, range.995–1.02; head1.000, range.995–1.01. Coordinate scales can learn column-relative write amplitudes; head scales overlap more with write gates. Expected speed loss below1%, to verify at startup against bare parent. Reports every2000; review5000; total13500.
