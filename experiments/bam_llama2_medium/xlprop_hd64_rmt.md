# XLProp HD64 RMT transfer

- RUN: `RMTXLPropHD64T4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero`.
- Worktree `/data0/xd/rmt-xlprop-noo`, branch `codex/rmt-xlprop-noo`; TPU `xd-v5p-32-2910072-maxtext`, primary UC1a. Main `MaxText/exp.py` is ledger only.
- Parent: XLProp QKVZeroInit+SharedWriteNorm+EmbedSeedZero, runtime `671a0f2`, stopped12293 after12000 review. Current parent effective attributes match that runtime except comparison metadata.
- Pure JAX, direct layer scan, no fetchedO or O read. Keep28 layers,20 heads,T4096 TruePile,global batch128 and WD exclusions. D1920->1280,head96->64,RoPE24->16. RMT matrix60x96->60x64; proxy first20 rows, dynamic-read tail40 rows,C10. Address LoRA output20x60 unchanged, so R384 retained.
- MLP4109: total679,685,720 versus HD64 MHA679,645,440, delta+40,280 (+.00593%, .02458 W_Q total). Adjacent width4108 has a larger absolute mismatch. Uniform layers, no hardware rounding.
- RMTHealthDefaults captures all-layer carry/raw RMS/shared-energy, dynamic/static writes, contents and gradients. Checkpoint every250, permanent2000, latest2.
- LR2.5e-4,warmup240,total24000. Baselines: Llama2XLPropHD64, MuddLlama2XLPropHD64, HD64 BAM independent every-third MLP write.
- Bet: RMT-BAM -.010@12000, -.005@24000; .42step/s vs BAM .446 (~-6%; health differs).
- Focused CPU checks: full parameter tree/nearest budget/init/norm scope, consumed2-layer scan forward+gradient and health export. CPU/AOT/queue preparation in parallel; retained FLEX_START compiler owns no lifecycle.

Artifacts: `/data0/xd/bam_diagnostics/rmt-readnorm-launch/hd64-rmt-*`, `test_hd64_rmt.sh`.

Runtime `93a139a0122fba18c3846b8806edd2e133a74aa6`; CPU2 checks22.9s, AOT ready and FIRST_STEP4 verified. Worker45 reached; actual UC data/global batch128,24000 schedule,R384,all-health,pure-JAX and retention verified. Steps10-14 mean .4078 step/s: MHA .711 -42.6%, Mudd .620 -34.2%, BAM .446 -8.6%; RMT health broader than baseline health.
