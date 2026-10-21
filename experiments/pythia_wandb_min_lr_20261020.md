# Pythia 衰减到0与衰减到0.1×峰值：W&B训练loss配对比较

Pythia v0套件中6.9B/12B的cosine衰减到 `min_lr=0`，v1统一改为0.1×峰值。
6.9B v1在公开W&B上只有0–7009步，故用12B比较；2.8B两版调度相同，作重训噪声对照。
结果：[JSON](pythia_wandb_min_lr_20261020.json)；脚本：[pythia_wandb_min_lr.py](pythia_wandb_min_lr.py)。

## 结论

- 12B终点（136001–143000）训练loss：v1 − v0 = **+0.0043**（v0更低）；末1000步 +0.0027。
- 2.8B对照同窗口：**−0.0009**；末1000步 −0.0014。全程5000步分箱在 −0.003…+0.001 之间。
- 12B差值在约115k步（训练77%）前处于对照噪声范围；之后随退火出现，125k起稳定在 +0.004–0.005。
  50k–115k间v1略低0.001–0.002，幅度与2.8B对照相当，不作解释。
- 0.0043 nats约为终点loss的0.25%，约为Pythia 6.9B→12B规模差距（~0.05）的1/10；
  对下游准确率在重训噪声（约0.5–1分）以下，与官方v0/v1 lm-eval文件的无显著差异一致。

两版数据顺序相同，同一步看到同一批数据，因此逐步差值是配对的。SEM忽略步间相关，偏小；
可信的噪声尺度用2.8B对照的分箱波动（约±0.002）。指标是训练 `train/lm_loss`，不是验证集loss。

## 学习率调度：截断cosine

GPT-NeoX（Pythia所用，[learning_rates.py#L54-L76](https://github.com/EleutherAI/gpt-neox/blob/bab229d/megatron/learning_rates.py#L54-L76)）
不是把cosine缩放到 [min_lr, peak]，而是 `lr = max(peak·(1+cos(π·n/T))/2, min_lr)`，其中 n = 步数−warmup
（上限 T−warmup），T = 143000，warmup = 1430。因此cosine朝约0衰减（终点未截断值约1e-3×峰值：
2.8B为1.58e-7，12B为1.18e-7），触及min_lr后走平。0.1×峰值的触及点为 1430 + 143000·arccos(−0.8)/π ≈ **115139步**。

W&B记录的lr与该公式吻合（四条链最大相对偏差≤3.5e-3×峰值，来自步序偏移）：v0与v1 12B在115139步前逐步相同，
之后v1在1.2e-5走平、v0继续降到1.18e-7。v1沿用同一调度器，两版只在最后约27.9k步（19.5%）不同，
这正是12B loss差值从约115k步开始出现的原因。2.8B两版min_lr相同，lr全程一致。

MaxText `create_learning_rate_schedule` 的cosine是 `lr = final + (peak−final)·(1+cos(π·p))/2`（p按T−warmup归一），
与NeoX截断cosine不同：中后期lr更高（71.5k步0.557 vs 0.516×峰值；115139步0.183 vs 0.100），终点同为0.1×；
全程lr积分高6.2%，115139步后的积分高28%。exp.py中 `BamPythia2p8B*` 配置未覆盖这一调度
（`decay_method` 继承默认cosine，`cosine_learning_rate_final_fraction=0.1`），因此与Pythia的“训练超参一致”在lr曲线形状上并不成立。

## 5000步分箱（v1 − v0，正值有利于衰减到0）

| 步数 | 12B v0 | 12B v1 | Δ12B | 2.8B Δ（对照） |
|---|---:|---:|---:|---:|
| 1–5000 | 3.2086 | 3.2132 | +0.0046 | −0.0064 |
| 5001–50000 | — | — | −0.0025…+0.0006 | −0.0022…+0.0010 |
| 50001–115000 | — | — | −0.0019…−0.0004 | −0.0032…+0.0003 |
| 115001–120000 | 1.7526 | 1.7532 | +0.0005 | 缺 |
| 120001–125000 | 1.7486 | 1.7508 | +0.0022 | 缺 |
| 125001–130000 | 1.7448 | 1.7487 | +0.0039 | 缺 |
| 130001–135000 | 1.7414 | 1.7467 | +0.0053 | 缺 |
| 135001–140000 | 1.7403 | 1.7448 | +0.0045 | −0.0009 |
| 140001–143000 | 1.7394 | 1.7437 | +0.0043 | −0.0009 |

逐箱完整数值、n和两侧均值见JSON的 `bins`。

## 数据来源与核验

- 来源：W&B公开项目 `eleutherai/pythia`，匿名GraphQL读取；每个run的 `history` 按2000步分块完整下载。
- 只取记录 `train/lm_loss` 的日志rank；链内后段覆盖重叠步（续训从最近checkpoint重启）。
- 脚本逐run断言架构与 `min_lr`，并记录 `weight_decay`/峰值lr/序列长度。全部 `weight_decay=0.1`、
  `seq_length=2048`；终点lr实测v0 12B为1.18e-7、v1 12B为1.2e-5，2.8B两版均为1.6e-5。

| 链 | run（按步顺序） | 配置 |
|---|---|---|
| v0 12B | `3n4tq9zd`(1–58999)、`22k7ocbw`(58001–143000) | 36层×5120，峰值1.2e-4，min_lr 0 |
| v1 12B | `ktjvz2on`、`13dd1daa`、`wt8xor8f`、`1pj5zple`、`2jny2vqa`、`2iqez2mg`、`2prrq2c0`(73001–125000)、`kg5ni1dl`(125001–143000) | 36层×5120，min_lr 1.2e-5 |
| v0 2.8B | `1c9jr6vp`(1–48000)、`12vuw5ef`(47001–143000) | 32层×2560，峰值1.6e-4，min_lr 1.6e-5 |
| v1 2.8B | `2l7ymzlr`(1–49693)、`4x2lfblu`(48001–99356)、`2j0vfxrj`(136001–143000) | 同上；99357–136000无公开日志 |

排除项：

- 分组名“13B Decay”的另一链（load目录 `.../pythia/13B-decay`，run `3eahol6d`/`iesl48iq`/`nlawi55i`等）
  配置为32层×4096即6.9B形状，loss比真12B高约0.2，不是v0 12B。
- v1 12B在约73k步后曾有另一分支（`2pl0f0gn`→`3hssg7up`，终点115001–143000）；与采用的最新链终点只差0.0013。
- v1 6.9B（`v2-6.9b_1ukrki3t`）公开日志仅0–7009步；v0 6.9B完整（`3e8nnx58`、`vi7laank`），但无v1可配对。

## TensorBoard导出

`/data0/xd/tensorboard_logs` 下：

| 目录 | 版本 | W&B链 | 训练步覆盖 |
|---|---|---|---|
| `Pythia_2.8B_2022` | v0（min_lr 0.1×） | `1c9jr6vp`、`12vuw5ef` | 1–143000 |
| `Pythia_6.9B_2022_Decay` | v0（min_lr 0） | `3e8nnx58`、`vi7laank` | 1–143000 |
| `Pythia_12B_2022_Decay` | v0（min_lr 0） | `3n4tq9zd`、`22k7ocbw` | 1–143000 |
| `Pythia_2.8B_2023` | v1（min_lr 0.1×） | 同上表v1 2.8B | 1–99356、136001–143000 |
| `Pythia_12B_2023` | v1（min_lr 0.1×） | 同上表v1 12B | 9–143000 |

2022三个目录为先前导出；2023两个由 [export_pythia_wandb_tb.py](export_pythia_wandb_tb.py) 生成，标签相同
（`loss`、`learning/loss`、`lr`、`learning/current_learning_rate`、`validation/loss`、`test/loss`），
`source.json` 记录每段run的配置、summary、计数与缺失区间。导出的训练loss与本页缓存逐步完全一致。

`validation/loss` 不能跨版本直接比较12B：Megatron验证迭代器在每次评测时前进10个batch，
v0 12B每1000步评测，v1 12B在18000步后只在143000评测，终点评测的验证batch不同
（终点v0 1.7262、v1 1.7412）。2.8B两版均每40000步评测，终点val差 −0.0021、test差 −0.0020，与训练loss对照一致。

## 复现

```bash
python3 -I experiments/pythia_wandb_min_lr.py \
  --cache /data0/xd/bam_diagnostics/pythia-wandb-min-lr \
  --out experiments/pythia_wandb_min_lr_20261020.json
```

原始逐步日志缓存（每run一个JSON，共28MB）：`/data0/xd/bam_diagnostics/pythia-wandb-min-lr`；
缓存存在时脚本不重新下载。官方Pythia配置及paper Table 6的WD笔误（实际0.1）见
[EleutherAI/pythia#132](https://github.com/EleutherAI/pythia/issues/132)。
