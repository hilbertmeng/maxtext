# MediumProp27 layers at the original parameter budget

Runtime worktree `/data0/xd/mediumprop-qk75-sparse`, branch `codex/mediumprop-qk75-sparse`.
Architectural base `Llama2MediumPropL27`: D1200,H16,head75,L27,T4096,MLP1600.
MHA RUN `BamMHAMediumPropL27C256TruePile`; BAM RUN
`BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile`.
Both repaired TruePile4096,13500 steps, inherited LR3e-4 and optimizer.
MHA uses the standard full QKV/WO C256 control, RoPE74/NoPE1, ordinary layer scan.
BAM retains M75x32/C8,QK57+RoPE18,full-M sharedrank4 dynamicQK plus static reads,
shared dynamicVO keys with independent gates,noWV,WO kept,AllLocal.
Private GELU address LoRA R256 MLP writes at zero-based1/4/7/10/13/16/19/22/25,
nine3-layer block scans. MLP[2311,2184,2311] exactly repays BAM parameters.

Actual full parameter trees:18-layer MHA432121200;27-layer MHA432142800
(+21600=.015W_Q,normalization gains);27-layer BAM432128192
(-14608=-.0101444W_Q vs27-layer MHA). Uniform MHA1599 is further from the old budget
than1600:432045600 vs432142800. No hardware rounding. W_Q=1200^2=1440000.

Targets: MHA27 versusMHA18; BAM27 versusMHA27 and currentBAM21.
All are equal total-parameter comparisons to nearest integer channels.
Existing18-layer Prop classes remain unchanged.

Pre-run bets at13500: MHA27-MHA18 -.005; BAM27-MHA27 -.130; BAM27-BAM21 -.004.
Speed bets UE5a: MHA27 .640 vsMHA18 .714 (-10.4%); BAM27 .425 vsBAM21 .466 (-8.8%).
CPU: exact parameter trees/full train-step shapes plus 27-layer finite forward/gradients passed;
47-test BAM regression passed once after fixing the MHA-control fetched-M flag.
