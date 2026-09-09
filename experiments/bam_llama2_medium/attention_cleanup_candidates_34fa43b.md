# attentions.py 可清理分支审计

审计基准：`refactor-bam` / `34fa43b4910d1e62fff7c7becf544b7d8952cb8c`。
`MaxText/layers/attentions.py` 为 4,141 行，审计时该文件无工作树修改。
源码 SHA256：`c4be44805dc603d414482b9cb04d27fcc7c8fa0b9301b84b18040e3146dd9461`。
实验结论取当前工作树 `MaxText/exp.py`（含其他任务尚未提交的台账更新），不是只看 Git HEAD。
本节以下保留原始候选审计与估算；后续实施范围及验证见 [清理实施记录](attention_cleanup_34fa43b.md)。

## 统计口径

- 只计算 `attentions.py` 的**净减少物理行数**：包含被删除块内注释/docstring，不靠清空空行增加数字。
- A 组给出可核对的直接删除行数；B/C 组是按参数创建、前传、辅助函数、配置读取和约束逐块估算的范围，尚不是实现后的 diff 统计。
- 删除实验分支意味着该实验以后按登记的 runtime commit 复现；`exp.py` 的配置类、继承链、runtime hash 和实验结论应保留，并标为 ledger-only。不能删掉实现却默默让旧选项退化成另一个模型；集中做“不再支持”的校验。
- 多项共享参数/验证/辅助函数，不能机械相加。特别是 BatchedLocalQK 与 grouped norms、QK72 与 seed、旧 LocalQK tiers 与 grouped norms 有交叉。
- 以下“无调用”仅指当前仓库（含测试及脚本）内检索结果；不声称检查了所有独立 worktree 或仓库外脚本。
- 原有实现中设备计算为零、参数未参与前传，与 Python 分支真正不可达是不同概念。

## A. 可精确定位的死代码 / 无调用接口：48 行

| 项 | 当前定位 | 依据 | 净减少 |
|---|---|---|---:|
| 无调用 `_contract_bam_read` 包装函数 | 1973–1978 | tuple 重构后生产路径与测试都直接使用 `_contract_bam_read_sides`，只剩定义 | 6 |
| 无用 import / alias / vmap | 41、59、108 | `rearrange` 未使用；`PRNGKey` 只在 docstring 出现；`dynamic_vector_slice_in_dim` 无调用 | 3 |
| 模块级 `global_block_*` / `global_*_layout` 等初值 | 92–103 | `tpu_flash_attention` 的 327–338 是同名局部赋值，闭包使用局部变量；没有 `global` 声明或仓库内外部引用这些模块变量 | 12 |
| `bam_read(return_key_stages=True)` 返回诊断阶段的接口 | 2172–2194；移除 2128 同行参数 | 仓库内没有生产、测试或脚本调用该选项；分支可执行，但没有调用方。若保留研究接口，可迁到 diagnostics | 23 |
| `key_row_mode` / `key_col_mode` 独立覆盖接口 | 1901–1902、2132、2157，及同行参数/表达式改写 | 原 fetched-output-gate 退出后，无调用方传入非默认覆盖；统一使用 `key_mode` 即可。**不是删除 row/col 独立 gate 或 scale** | 4 |

前三项是静态无消费者的代码，共 21 行；后两项是当前无消费者的可选接口，共 27 行。
A 组已在内存中构造删除草稿，确认净减 48 行并通过 `ast.parse`；未写回源文件，未将语法通过等同于数值回归通过。

## B. 有明确负结果或已被测速否定的候选

“推荐归档”表示将该选项退出当前生产实现，保留历史 runtime 复现能力；不表示未经验证直接删除。

| ID | 可清理逻辑 | 当前代码定位 / 计数依据 | 实验依据（Δloss > 0 为更差） | 预计净减少 |
|---|---|---|---|---:|
| B1 | Batched Q/K LocalQK 读路径 | setup 2997–3010 共 14 行；forward 3487–3512 共 26 行；另有属性、shape 限制和特殊初始化 | `BamV2GQChunk256BatchedLocalQKReadEightLayerProfile`：650.57 vs 644.20 ms；scan 版 679.40 vs 672.17 ms，都明确 rejected。删除 batched，不删除 packed projection | **50–60** |
| B2 | `head_rank_gate` 直接 sigmoid 路由 | `factorized_head_bam_read` 2261–2264、2311–2320；packed gate-width、3548 等门路径、参数传递 | `BamLlama2MediumV2C256Paired40LocalQKRank2HeadRankGate`：5k–7k 对 SharedRankGate 稳定 +.0021～+.0031；保留 `legacy` 和 `shared_rank_gate` | **22–30** |
| B3 | LocalQK 外置 amplitude / depth-amplitude 实验族 | 2587–2594、2789–2796、3281–3286、3334–3345；3478–3481、3564/3576；2355–2359 | 修正 scan 层号后的 `...SharedGateDepthAmplitude005/050`：+.006、+.0047～+.0055；明确 harmful。没有把早期错误层号的实验当作真正 depth scaling | **45–55** |
| B4 | fetched-read 的 **depth scaling** | `_depth_scaled_bam_read_amplitude` 1872–1883；2552–2555、3259–3266；3303、3307–3316、3322–3329 | `...DepthAmplitudeGate050ScanLayerFix`：+.017；Gate005 修正版约 +.041。只退出 depth prior，保留不按深度缩放的 fixed/learnable amplitude 和插值 | **45–55** |
| B5 | LocalQK / Fetch read-key 上的 `2*SiLU` | `_activate_bam_read_key` 1859–1867；1886/2124/2198 的参数透传、2577–2578、2625–2626、2669–2680、3426 等 | `...FetchColSiluReadKey`：+.00486@1800；`...LocalQKColSiluReadKey`：+.00737@1800，分别从 800/600 起为正。结论限于已跑的 col 实验；row/both 无已建立收益 | **38–48** |
| B6 | 非零、Q/K 配对的 row-key seed | `_packed_factorized_local_qk_init` 2369–2372、2394–2400；2457–2458、2732–2736、2989 | `...SeededPaired40`：+.0057；`...SeededPaired72`：虽好于 dead-tail control，仍 +.0003 vs Paired40 / +.0015 vs V2。**保留正常 Paired40 adapter 的 paired initialization** | **18–25** |
| B7 | `per_head_static` LocalQK | setup 2944–2954；3586–3601 的无 gate kwargs 特例；静态 W_R broadcast 特例可后续收口 | `...CombinedReadPerHeadStaticLocalQK`：+.0120 vs Combined、+.0209 vs PerHead@4800。只退出静态 keys，不由此否定动态 per-head keys | **24–30** |
| B8 | QK72 / `qk_tail` 扩展与 padding | 2459–2460、2716、2728–2731；3462–3468、3635–3638 | `...FullMPostReadV8QK72PartialRoPESeparateQKPairedInit` 的 row key / adapter 梯度恒为零；seed 激活后也无独立收益。是训练中的 dead tail，不是不可达 Python 代码。保留有效的 Paired40/head_tail 与 PartialRoPE | **18–24** |
| B9 | pre-RoPE LocalQK + adjacent pairing 实验路径 | pre 分支 3994–4005；`_apply_adjacent_rope` 3643–3648；4011–4013；injection/pairing 属性与约束 | 历史 PreRope +.0101，Adjacent 进一步 +.0038；Paired40 Rank2+QKNorm 虽 -.00111@6400，但收益衰减、慢 2.2%。归档这些具体组合，**保留普通 Transformer QKNorm 与 BAM PartialRoPE** | **30–40** |
| B10 | 取消 LocalQK pre-RMS bias 的开关 | 2576、2673–2676、3494、3545 等；固定保留加 bias 的路径 | Medium `...LocalQKNoPreRMSBias` +.00267；XL `...NoPreRMSBias` 稳定 +.0024～+.0033。保留 bias 参数和加法 | **6–9** |
| B11 | native fetch diagonal 开关 | `_bam_fetch_op` 1787 的 guard；2628 的属性、3902 的调用；固定 diagonal-one 操作 | Clean NativeDiagonal +.0331；SharedReadLLFNativeDiagonal +.02358。没有可观速度/缓存收益；历史 V1/Compat 仍要回到其 runtime | **3–5** |
| B12 | 只归一化 write address、不归一化 data 的开关 | 2658 的属性；3710 直接使用 `self.write_data_norm(u1)` | `...WriteAddressRmsOnly`：+.034 plateau。分支虽可删，**物理行数收益很小**：同行三元表达式收口不算删一整行 | **1–3** |
| B13 | source compression 的 `mul_reduce` 替代实现 | 3745–3750、3760–3765；2642–2643 的属性和 2746 的约束 | `BamDirectPLocR256GeluBf16PackedSourceMulSixLayerProfile`：716.26 vs 708.95 ms，慢 1.03%；可固定 dot。该结论来自已测形状，不宣称所有未来硬件都如此 | **13–16** |
| B14 | 动态遗忘 gate | 3162–3176 参数创建；3728–3733；`_update_bam_matrix` 1820–1823；属性/约束 | `...RmsGateOnlyDynamicForget`：-.0007@2800，台账结论 null；速度 .293 vs .295，未建立收益。证据较短，优先级低于长程明确负收益项 | **28–35** |
| B15 | whole-matrix read RMSNorm 选项 | `_matrix_for_read` 3626–3631；2627、2711；按 raw M 简化调用 | `...FactorizedLocalQKNoMNorm` 完成 13500，相对有 MNorm 对照在 5600–7000 改善 -.0078，且 .325 vs .315 steps/s；V1/V2 已继承 none。历史对照较早停止，不把该 gap 写成同终点比较 | **6–10** |

这一组与 A 合并，扣除 seed/batch/amplitude 等重复条件后，规划规模约 **380–490 行（当前文件约 9%–12%）**。不是已实现或已验证的 diff，也不表示性能会相应提升。B9/B14 属于证据较弱的归档候选，可以留到下一批。

## C. 可进一步收口，但有兼容性、历史正收益或实验归因限制

| ID | 候选 | 预计净减少 | 必须说明的限制 |
|---|---|---:|---|
| C1 | fetched K/V 的 learned decoder：V8→32、V8→8、K16→32，以及共享/per-head decoder 选项 | **50–65**；保留旧参数树兼容创建时约 **25–35** | V2 四个 AbsVRowDecode 都更差（+.00262 / +.00365 / +.00576 / +.00432）；AbsK16Project 对 Direct +.0075。但早期 V1 Project 曾 -.0011@3600，不能说所有谱系都负收益。`abs_v_row_decoder` 在 Direct 路径仍通过 `self.param` 创建，代码明确标为历史未使用参数；删除会影响 checkpoint/optimizer tree。必须和普通死代码区分。保留真正用于压缩的 `abs_v_cache_projection`，也保留 AbsK Direct 的缓存选项 |
| C2 | learned/grouped read RMS 与 write gamma / post-RMS beta、dormant norm 兼容路径 | **120–170**，与 B1/B7 有交叉 | 核心入口 1692、1830、2570–2573、2835–2847、3177–3195、3444–3458，加上 grouped norm 参数创建调用。Medium 写 gamma +.00937、beta +.00680、组合 +.00463；XL 近中性/微差（+.00069～+.00097）。旧 learned read 只有 -.0015 的短程小幅结果或 +.0016；不是全数严格负收益。`GroupedRMSNorm` 当前也承载主线的**固定** write RMS，不能整类删除却漏掉归一化。可将固定 RMS 包装保留，或显式替换后验证；老参数树按 historical runtime 处理 |
| C3 | 旧 `shared + R_q/R_k rematrix`、unpacked factorized、动态 per-head LocalQK 多套实现 | **100–160**，与 C2 明显重叠 | setup 2955–3084、3586–3624、bam_read rematrix/concat 兼容。是“只维护当前 factorized packed 主线”的架构取舍，不是全部已被实验证明无效：动态 per-head 曾相对 Combined 改善 -.0087，只是较慢。当前单元测试也覆盖旧 read 语义 |
| C4 | `write_v_mode='mix'` 与 `o_tail` | mix **30–40**；o_tail **8–15**，与 C2 的 u2 norm 选项重叠 | mix：1807–1814、3147–3161、3692–3699；台账称 negligible，但实际短程 -.0037，并非严格负收益。o_tail：537 附近历史实验 +.0272；带 grouped bias 仍 +.0198 vs Direct，可归档。不要顺带删除 static write：其 loss 更差，但速度曾提高 11.3% |
| C5 | 多 fetched-read heads 打包为一个 MHA head | **20–35** | `bam_fetched_read_num_heads` / `_pack_fetched_bam_heads` / gate stats 的倍数 head 处理。唯一 XL32-readheads 试验同时把 M 改为32×64、C16、P_loc512，loss +.01555、慢3.1%，不能独立归因于 fetched head 数。未来扩展/缓存目标是否保留需要单独取舍 |
| C6 | 5-D fetch-axis、双矩阵 codebook、旧 rematrix 返回兼容 | **15–30**（若连 rematrix 一起删，计入 C3） | 1930–1931、1953–1960、2149 等。当前 BamAttention full fetch 为4-D，主线不走旧5-D；但测试有 `test_single_fetch_axis_squeeze_matches_values_and_gradients` 和历史 combined-read 比较，因此是**迁出兼容测试/接口**，不是零消费者死代码 |
| C7 | `dynamic_rms_gelu_mix` 中额外 GELU，保留 learnable mix-scale | **4–8** | CleanMixScaleOnly 比 CleanGeluAlphaMix 晚期仅 -.00019，支持“GELU 无额外持续收益”；但 GELU+scale 整体对 Clean 有 -.00276，scale-only 有 -.00295。不能把 learned scale 一并删掉，也不夸大单独 GELU 的负结果 |
| C8 | LocalQK rank expansion 的 dot 路径，固定 mul-reduce | **10–15** | 2343–2344、函数参数/约束和调用；Rank2 799.89→707.10 ms、Rank4 820.75→720.49 ms。只适用于已测 TPU 形状；dot 很适合作为非零 forward/VJP 的参考实现，可移到测试，而不是失去验证能力 |

C 组不能直接累加成“再删多少”，因为共享实现与历史兼容策略决定最终净额。

## 不应归入无收益垃圾代码

- **Rank2、SharedRankGate、Paired40/post-read V projection、PartialRoPE**：Paired40 配对 adapter 完成训练后对 V2 -.00261；Rank2 对 Paired40 -.00368；SharedRankGate 在 Medium 对 Rank2 -.00156。XL SharedRankGate 较差不推翻 Medium 结果。
- **row/both interpolation 与无 depth 的 amplitude**：Medium row-only 对 Gate050 约 -.003；XL 两个继承类台账仍标 running，不能由旧 plain-control 的失败实验删除整套实现。本次未查询远端实时状态。
- **LocalV rank2 / shared、LLF**：这些已合入主线且有正收益；LLLF 调度本身主要在 `exp.py` / fusion 中选择，归档 LLLF 配置不会从此文件删掉一个专属算法块。
- **AbsK16 Direct、CompressedVLocalQK、static write**：有 loss 代价，但同时有缓存或速度收益，不能只看 loss 宣判没有价值。
- **Rank4、PLocR512、C4/C16/C32 等数值设置**：很多只改变维度；没有独立算法分支时，归档该配置对 `attentions.py` 的净删行数接近 **0**。不要把通用 rank/维度支持按某个失败超参数整段删除。
- **普通 MHA/GQA、KV cache、Flash/CuDNN、MLA**：当前 BAM 不用不等于仓库无用。没有相应实验/消费者证据支持本次删除。
- **read/write dtype、epsilon、init 参数**：历史存在明确数值谱系差异；不能因为某次 bf16 更差就删掉复现所需支持。
- **健康指标与归因接口**：仍被训练/诊断使用的指标不能因为不能降 loss 被当作失败架构。
- **`bam_write_u_proj` / `P_agg_u`**：代码注明预训练模型适配用途；未获得该用途已废弃的证据。

## 已移除的实现：本次可减少 0 行

先前输出 gate（LoRA/GELU/SiLU/linear、factorized head-coordinate）及 fetched-key pre-RMS bias 的删除已记在 `attention_simplification_audit.md`；不能再次计入。
`bam_fetched_row_rank`、fetch-rank2、FetchReadR512Gelu / FetchColReadR128Gelu、self-read gate、query streaming-scan 等也只在当前台账或历史工作树留有配置，当前 `attentions.py` 中没有对应算法实现。本次将它们计为 **0 行**，而不是依据类名估算删除量。

## 落地顺序

1. A 组：静态清理及无调用接口退役。为需要保留的诊断接口建立单独入口，而不是静默改变返回值。
2. B1–B8：先清晰退出 batch/head-rank/depth-amplitude/SiLU/seed/static-key/QK72 族，再处理小开关和较弱证据项。保留 SharedReadLLF、CleanControl 和历史 XL Rank2 的主线语义。
3. C1/C2：确定 checkpoint 与历史实验复现边界后再清理参数创建；不要把未使用参数创建误当成无兼容影响的代码。
4. 实现后核对参数 names/shapes、init 顺序与 sharding；复用已有非零 forward/VJP 验证及 BAM/local-fetch 测试。字节数/行数下降不等于 TPU 加速；没有测量前不声称速度提升。

本报告的代码定位均以 4,141 行快照为准。若其他任务修改了源文件，应先按函数名和 selector 重新定位。
