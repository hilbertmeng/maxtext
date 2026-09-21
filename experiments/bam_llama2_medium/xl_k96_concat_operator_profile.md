# XL K96 QK-concat operator matrix

Compare four write-outer (W) × dynamic-column-read (R) implementations:
M=mul_reduce, D=dot. Full24layers,D2048,H16,head128,M96x32,C8,
MLP6266,production batch/sequence, generic health ON and968BAM health scalars.
Static Q/K einsum unchanged. No factorized rank-to-head expansion remains.
Direct baseline within matrix is WMRM. Same v5p-32VM andzone;
sealed100-step AOT,50000 LR schedule; trace10–14, speed20–24.
Implementationworktree `/data0/xd/llf-parameter-matched`, branch`codex/llf-parameter-matched`.
Rawartifacts `/data0/xd/bam_diagnostics/xl-k96-concat-operators/`.

Classes:
- `BamXLK96ConcatOperatorWMRMFull`
- `BamXLK96ConcatOperatorWMRDFull`
- `BamXLK96ConcatOperatorWDRMFull`
- `BamXLK96ConcatOperatorWDRDFull`
