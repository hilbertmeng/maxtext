# Old XL T2048 full-history Mudd control

Implementation: `/data0/xd/mudd-xlprop-truepile`, branch `codex/mudd-xlprop-truepile`.
RUN: `MuddLlama2XLT2048Head16x128`; owned TPU `xd-v5p-32-2910071-maxtext`.
Runtime: `367ee52bc60135da1b65760a977fcc2adc637828`.

D2048, 24 layers, 16×128 heads, SwiGLU MLP base5504 with the original Mudd depth-dependent128-rounded widths. Sum132096 equals24×5504. Full growing layer history uses the already verified Mudd implementation; non-scan is required by that history and varying widths. T2048 legacy Pile, global batch256, LR2e-4, warmup500, plan50000; C256 attention with bf16 logits. Match historical XL MHA/BAM actual AOT optimizer via explicit `wd_mults=[]`. Generic training health ON. Checkpoint250, permanent2000, latest2.

Exact model shape audit: 1,426,013,701 parameters, MHA+5,092,869 =1.214W_Q (+.358%). CPU gate passed4 checks in36.7s.

Direct baseline: `BamMHALlama2XLHead16x128C256`. Add the new Mudd run to the current old-XL BAM comparisons after FIRST_STEP for a live BAM/Mudd MHA-relative gain ratio.

Initial zone UC1a, based on current six owned UC leases; add UE5a/EW4b passive candidates if primary capacity stalls. Borrow user-owned FLEX_START `llm-jax-v6e-1-0` in EW4a for AOT only; never reclaim it.

Targeted CPU gate: exact full-model shape/budget and complete history; tiny consumed-history forward/gradient. CPU, retained-worker AOT, and new training queue run concurrently through `launch_train_parallel.py`.

Pre-run bet: terminal Mudd−MHA −.050; .60step/s vs historical MHA .708 (−15.3%).

Started UC1a; AOT loaded/FIRST_STEP verified5. Worker configuration matches sealed runtime, zone-local legacy dataset and global batch256. Checkpoint250/permanent2000/latest2 verified. Steady .590step/s at steps28–77 (−16.7% vs oldMHA .708; +19.4% vs currentBAM .494, BAM extra writer statistics differ). No passive queue needed. Registered as an additional direct oldXL BAM baseline.
