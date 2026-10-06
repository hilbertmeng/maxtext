# HD64 RMT transfer to legacy XL T2048

RUN `RMTXLT2048AllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero`.
Worktree `/data0/xd/rmt-xlprop-noo`, branch `codex/rmt-xlprop-noo`; main `exp.py` remains the ledger.
Owned trainer `xd-v5p-32-2910075-maxtext`; primary UE5a, passive EW4b/UC1a after five minutes without capacity. All training capacity is spot. Recent UC/EW leases are repeatedly short; keep accepted queues, do not cycle provisioning queues. Borrow only the verified idle FLEX_START compiler `llm-jax-v6e-1-0` in EW4a for AOT; never enroll it in cleanup.

Source is the actual HD64 RMT runtime `93a139a0122fba18c3846b8806edd2e133a74aa6`, not later MLP-input-norm/attention-bias experiments. Preserve pure JAX layer scan, QKV static zero initialization, VectorNorm, no layer matrix pre-norm, shared normalized attention/MLP/embedding writes, learnable zero embedding seed address, dynamic direct unembedding, learned final matrix norm, and NoO/no fetch.

Destination: D2048, 24 layers, 16x128 heads, T2048, batch256 on v5p-32. RMT M48x128 has first16-row control proxy and tail32 dynamic reads, compression C8. RoPE32 is independently projected from normalized proxy; matrix QK contributes unrotated96 coordinates. Full48-row attention/MLP/embedding writes. User specifies address LoRA R256 for all three. No permanent vector residual stream.

MLP7442 is the nearest uniform integer width: 1,420,865,056 parameters versus historical MHA1,420,920,832, difference -55,776 (-.00393%, -.013298 W_Q total). One unit of uniform width changes147,456 params; width7443 overshoots91,680. No hardware rounding. Match old XL legacy2048 regional data, LR2e-4, warmup500, total50000, actual historical AOT WD `wd_mults=[]`.

Inherit RMTHealthDefaults: all-layer raw M/carry/shared-energy, dynamic/static read/write and contents, and gradient health. Save every250, permanent2000, latest2 for diagnosing late erosion. Direct baselines: current old XL BAM independent-every-third, old XL Mudd, historical XL16x128 MHA. Report XL500-step loss windows, with gain/Mudd and gain/BAM against the same MHA and synchronized common steps.

Focused CPU gates: exact full-budget/nearest width, effective config/shape/init/norm scope, consumed scan forward+gradient and health export. CPU/AOT/training prequeue run in parallel; sealed runtime attributes are separately checked.

Before launch bet: at20000, RMT-BAM +.005; at50000 +.010. Terminal gain/Mudd ~1.1x. Speed ~.47step/s, ~-5% versus BAM .494 (extra RMT health; not matched). HD64 already lost to BAM by13k and showed late MLP amplitude/shared-energy growth; shorter T2048 and wider heads offer a direct test, not a presumed cure.

Artifacts `/data0/xd/bam_diagnostics/rmt-readnorm-launch/old-xl-rmt-*`, focused test `test_old_xl_rmt.sh`.
