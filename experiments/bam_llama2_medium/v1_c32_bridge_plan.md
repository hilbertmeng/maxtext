# V1 → Current C32 Bridge

Goal: explain and close the loss gap between historical `BamLlama2MediumV1`
(`03367ac`, native C32) and the current fixed-scale native-C32 path. Find all
material main effects and interactions, then test whether the same factors
explain `C8 exact − V2`.

## Anchors

| Anchor | Code/config | Status |
|---|---|---|
| Historical V1 | `03367ac` + `BamLlama2MediumV1` | native-JIT exact through 800; cross-topology AOT +.0527 @1,800 |
| Current C32 | `31bfd45` + `BamLlama2MediumV2C256FetchAmplitudeGate005C32A113137Fixed` | stopped 3,348; +.0162 vs V1 @3,200; native-JIT − AOT −.0015 @400 |
| Current C8 | `31bfd45` + `BamLlama2MediumV2C256FetchAmplitudeGate005C8A565685Fixed` | stopped 10,285; recent-six +.00250 vs V2 |

## Factor inventory

| Factor | V1 | Current C32 | Evidence/control |
|---|---|---|---|
| compilation path | native v5 JIT | cross-topology AOT loaded on v5 | historical V1 AOT diverges while native-JIT is exact; modern native-JIT reverse controls required |
| execution graph | dense, non-scan | C256, layer-scan | paired 2×2 required |
| fetched/local read | shared combined `full+local_o` | no `local_o`, diagonal-one | forward/reverse controls required |
| write contraction | dot | multiply+reduce | expected semantic no-op; verify |
| P_loc bottleneck | single linear | D→256→nV GELU | modern linear control did not rescue |
| P_loc bias | none (`x`) | output bias (`x_bias`) | not removed by the prior LinearPLoc control |
| LocalQK projection | unpacked Q/K | packed Q/K | mapped/native initialization must be separated |
| RMS statistics | activation/bf16 | fp32 | historical controls show material loss effect |
| read epsilon | 1e-4 | 1e-4 | already matched in current C32 |
| read amplitude | implicit scale 2, gate .005 | fixed 11.3137/√32, gate .005 | forward-equivalent; verify step-0/Jacobian |
| layout/refactor | historical bnt-era code | btn production path | paired current-commit control required |

## Procedure

1. Re-run historical V1 with both native v5 JIT and cross-topology AOT; use
   native-JIT as the exact anchor and measure AOT drift separately.
2. Build a compatibility branch from the historical implementation; add one
   factor at a time forward and remove each factor from the current endpoint.
3. Screen all factors to 2,800 steps; extend material/trending factors to 6,800
   and endpoint/top-factor controls to 13,500.
4. For background-dependent effects, run targeted 2×2 controls. Rank persistent
   main effects separately from interactions.
5. Remove the complete harmful-factor set from the current C32 endpoint and
   require the residual endpoint gap to close.
6. Repeat the top-factor 2×2 controls at C8 and test whether their combined
   effect explains `C8 exact − V2`.
7. Run current C8/C32 native-v5-JIT reverse controls before attributing any
   residual endpoint gap to architecture or parameterization.
