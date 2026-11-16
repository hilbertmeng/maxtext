# Total-budget Medium depth residual controls (2026-11-16)

Owner authorized six new RUNs: DHC x4 tanh/sum, mHC x4 SK20 A.6 learned-head, and Full AttnRes on each L12/L24 total-budget MHA parent. Start from scratch; no existing experiment is restarted.

Runtime worktree: `.auto-agent/worktrees/medium-depth-residual`, branch `mqy/medium-depth-residual-20261116`. Parent merge e31417a5 includes upstream refactor-bam75099911 and the existing residual implementation. Training infrastructure matches reviewed main e31417a5: no historical infrastructure is brought back. Only experiment configurations and launch tests are new. TensorBoard rotation500 plus rotating writer construction/begin_step are inherited; five summary regression tests pass.

Use the matched parent's TruePile4096 replicas, T4096, batch16/device on v5p-16 (128 sequences/update;524288 tokens/update), original13,500 steps/LR endpoint, optimizer and seed. All six are unrolled because residual connections require it, using minimal remat, tuned Splash1024 and existing math-preserving Pallas HC kernels. DHC/mHC controller overhead is repaid through uniform FFN width; AttnRes repays the nearest one hidden unit. Integer rounding leaves <0.007% total deviation.

| Configuration | Total parameters | FFN | Delta vs matched parent |
|---|---:|---:|---:|
| Llama2MediumPropL12TotalBudgetTruePile | 432111376 | 3728 | +0 |
| Llama2MediumPropL12TotalBudgetTruePileDHC4Tanh | 432129088 | 3723 | +17712 |
| Llama2MediumPropL12TotalBudgetTruePilemHC | 432123421 | 3661 | +12045 |
| Llama2MediumPropL12TotalBudgetTruePileAttnRes | 432128464 | 3727 | +17088 |
| Llama2MediumPropL24TotalBudgetTruePile | 432133152 | 2874 | +0 |
| Llama2MediumPropL24TotalBudgetTruePileDHC4Tanh | 432159744 | 2869 | +26592 |
| Llama2MediumPropL24TotalBudgetTruePilemHC | 432130229 | 2807 | -2923 |
| Llama2MediumPropL24TotalBudgetTruePileAttnRes | 432158496 | 2873 | +25344 |

CPU gate verifies full parameter trees, resolved schedules, finite tiny forward/gradients, remat consistency and BF16 outputs for all six. Exact sealed-config and full-topology AOT gates precede training. Require actual Loaded compiled function and FIRST_STEP; queue submission alone is not a successful launch.

Scientific expectation: small same-step improvements over the matched MHA are plausible, with larger relative benefit at L24; ordering among DHC/mHC/AttnRes is uncertain. Historical DHC has fatter gradient tails, so inspect raw norm/loss/logits jointly. Compare each method directly to its own parent, not across different D widths. Conservative speed expectation is a modest slowdown relative to ~.999 L12/~.671 L24 MHA; investigate >25% sustained slowdown before attributing it to architecture.

Scope: six new Spotv5p-16 trainers, independent exact IDs6111601..6111606; primaryUE5a/backupUC1a per current regional lease policy; no EW4b default. Use exact per-config AOT, independently release compiler-owned resources. Protected compiler0 and existing Pythia a33933a6 training remain untouched. No FLEX_START requests. Stop on failed CPU/config/AOT or nonfinite health; never change hyperparameters automatically. Train to inherited13500; native200-step windows, public reports each1k plus terminal; shared ten-minute health wakeup.
