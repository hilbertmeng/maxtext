# Attention cleanup implementation

工作树 `/home/xd/projects/maxtext`，分支 `refactor-bam`；对照 runtime
`34fa43b4910d1e62fff7c7becf544b7d8952cb8c`。本任务不创建 RUN、不占用 `xd-` TPU。

## 范围

按 [候选审计](attention_cleanup_candidates_34fa43b.md) 清理 A 全部，以及
B1/B3/B4/B5/B7/B8/B9/B13/B14。包括无消费者接口、batched LocalQK、
LocalQK 外置 amplitude、fetched depth amplitude、read-key SiLU、静态 per-head
LocalQK、QK-tail 扩维、pre-RoPE/adjacent RoPE、source mul-reduce、动态遗忘。

按用户要求保留 B2/B6/B10/B11/B12/B15：`head_rank_gate`、非零 paired row-key
seed、关闭 LocalQK pre-RMS bias、native fetch diagonal、关闭 write-data RMS、
whole-matrix read RMSNorm。随后按用户追加要求，继续移除 C3 中动态 per-head
LocalQK 和 shared-key rematrix：删除 `R_q/R_k` 参数与 `bam_read` 的 R 参数，
底层 contraction 只保留带 head/rank 轴的路径。未打包 factorized LocalQK 保留，
其余 C 组未清理。常数 matrix retention、无深度 fetched
amplitude、rank 1/2/4、partial RoPE、post-read V projection 和 SharedRead/LLF 保留。

首轮从 4,141 行降至 3,789 行；追加清理再减少 76 行，最终 **3,713 行**，
合计净减 **428 行（10.3%）**。

## 历史配置

`MaxText/bam_config.py` 在 pyconfig 校验和 BamAttention.setup 中拒绝本轮退役选项，
错误包含具体字段和值，并指向 exp.py 所登记的历史 runtime。校验最终解析值，
不会因为父类曾启用 depth scaling 而拒绝已覆盖为 False 的后代。
`shared`/`per_head` 仅在启用 LocalQK 的层中退役；不使用 LocalQK 的配置可保留
未生效的旧字段。模块级校验使用实际 `layer_mode`，避免直接构造模块绕过检查。

`MaxText/exp.py` 为 74 个受影响类增加 ledger-only 注释；本轮没有修改配置值、
类名、继承关系、runtime hash 或结论（注释前后 AST 完全相同）。此前已经退役的
其他实验族仍按原 ledger 约定处理；本轮校验只覆盖本轮删除的选项。

## 验证

使用 diagnostics skill 固定 CPU 环境 `/data0/xd/conda/envs/maxtext-cpu`。

- `MaxText/tests/bam_config_test.py`：5 项配置校验测试。
- `MaxText/tests/bam_attention_test.py`：46 项保留功能测试。
- `MaxText/tests/bam_local_fetch_test.py`：LocalFetch/LLF scan、参数与梯度测试。
- `check_read_simplification.py --reference 34fa43b`：84 组非零 read 对照，覆盖
  FP32/BF16、rank 1/2/4、legacy/shared-rank/head-rank、row/col/both、dot/mul-reduce；
  输出与 M/key/mix 梯度的最大相对 L2 差异 **0**。
- `check_attention_cleanup.py --reference 34fa43b`：整层参数初始化、非零 read
  输出/M_out、参数/x/M 的随机 cotangent VJP；比较 CleanControl、SharedRead L/F、
  SeededPaired40、HeadRankGate、XL Rank2、四个保留开关、row interpolation、固定 amplitude，以及未打包 factorized LocalQK。
  为避免零初始化掩盖错误，对所有参数施加确定性小扰动。测试保留各实验的 M/head
  维度，缩小 batch/sequence/embedding/head 数和 P_loc bottleneck。XL 初始化显式传入
  `bam_k/bam_v`。`--start-case` 支持从指定下标继续检查。

**追加清理后全部通过**：5 + 46 + 6 = 57 项测试；84 组 read 与 10 组整层对照。
整层参数初始化、输出/M_out、参数/x/M 梯度均逐元素完全相同（最大绝对差异 0）。
机器可读摘要见 [验证结果](attention_cleanup_34fa43b_results.json)。
通用 Attention/AttentionOp/MLA 与 GroupedRMSNorm 的 AST 未改变；`git diff --check` 通过。
这里不推断 TPU 速度或训练 loss 轨迹。

复现命令（从主工作树运行）：

```bash
.claude/skills/tpu-diagnostics/scripts/run_bam_unit_tests.sh /home/xd/projects/maxtext
PYTHONPATH=MaxText python MaxText/tests/bam_config_test.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=MaxText /data0/xd/conda/envs/maxtext-cpu/bin/python MaxText/tests/bam_local_fetch_test.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=MaxText /data0/xd/conda/envs/maxtext-cpu/bin/python experiments/bam_llama2_medium/check_read_simplification.py --reference 34fa43b
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=MaxText /data0/xd/conda/envs/maxtext-cpu/bin/python experiments/bam_llama2_medium/check_attention_cleanup.py --reference 34fa43b
```
