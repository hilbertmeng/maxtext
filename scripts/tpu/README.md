# TPU 训练控制脚本

本目录保存从 `tpu-ag` 控制 GCP TPU VM 和 MaxText 训练的版本化脚本。它替代过去散落在
`/home/lishengping/mengqy/projects/` 和 `/home/lishengping/tpu/` 下的副本。

## 文件

- `run_exp.sh`：安全的实验入口。读取一个显式 profile；默认 `plan`，不会创建 TPU。
- `auto_train_arc_maxtext.sh`：TPU 排队、安装、代码同步、训练启动和重建循环。
- `install_0812_v5p_mqy_maxtext_jax081.sh`：复制到 TPU worker 执行的环境安装脚本。
- `auto_train_arc_maxtext_abbc_d30f81a.sh`：旧 ABBC 文件名的兼容 wrapper。
- `profiles/template.env`：普通 ARC 训练配置模板，默认跟踪 `refactor-arc`。
- `profiles/abbc_validation.env`：历史 ABBC validation 的 pinned-commit 配置。

运行日志和 PID 默认仍写入 controller 上的
`/home/lishengping/mengqy/projects/logs/`，不写入 Git 仓库。

## 两份历史 auto-train 脚本的差异

历史文件：

```text
/home/lishengping/mengqy/projects/auto_train_arc_maxtext.sh
/home/lishengping/mengqy/projects/auto_train_arc_maxtext_abbc_d30f81a.sh
```

两者只有代码同步策略不同：

- 原版：TPU worker checkout `refactor-arc` 后 `git pull`，每次可能得到更新的 commit。
- `_abbc_d30f81a`：fetch `recurrent-mudd-abbc-validation-d30f81a`，然后 detach 到固定 commit
  `4f0bd7d891dae790b0cb6947a2632e4691f7547c` 并验证 HEAD。

其余 TPU 创建、环境安装、设备清理、编译缓存和训练启动逻辑相同。现在两种模式合并到
`auto_train_arc_maxtext.sh`：

- `MAXTEXT_SYNC_REF=refactor-arc` 且不设置 `MAXTEXT_EXPECTED_COMMIT`：原版浮动分支语义。
- 同时设置 `MAXTEXT_SYNC_REF` 和 `MAXTEXT_EXPECTED_COMMIT`：固定 commit 语义。

默认是原版语义。普通训练不应继承 ABBC 的历史 pin。

## 使用

复制模板并填写实验身份：

```bash
cp scripts/tpu/profiles/template.env scripts/tpu/profiles/my_exp.env
$EDITOR scripts/tpu/profiles/my_exp.env
```

只显示解析后的配置，不产生云端副作用：

```bash
bash scripts/tpu/run_exp.sh plan --config scripts/tpu/profiles/my_exp.env
```

启动 controller：

```bash
bash scripts/tpu/run_exp.sh install+train --config scripts/tpu/profiles/my_exp.env
```

查看 controller 和 TPU 状态：

```bash
bash scripts/tpu/run_exp.sh status --config scripts/tpu/profiles/my_exp.env
```

停止 controller 只会停止排队/监控脚本，不会删除 TPU，也不会自动杀死已经在 TPU worker 上
运行的训练：

```bash
bash scripts/tpu/run_exp.sh stop-controller --config scripts/tpu/profiles/my_exp.env
```

删除 TPU 是独立高风险动作，必须额外显式确认：

```bash
CONFIRM_DELETE_TPU=yes-really-delete \
  bash scripts/tpu/run_exp.sh delete-tpu --config scripts/tpu/profiles/my_exp.env
```

## 配置原则

- 每次实验必须显式填写 `EXP`、`ID`、`TPU_TYPE`、`ZONE` 和 `PROJECT_ID`。
- `RUN_NAME` 默认等于 `EXP`，但可以单独覆盖。
- 普通训练使用 `MAXTEXT_SYNC_REF=refactor-arc`，不要填写 `MAXTEXT_EXPECTED_COMMIT`。
- 需要复现实验时同时记录 branch/ref 和完整 40 位 commit。
- controller repo 与 TPU worker repo 是两个路径：
  - controller：`/home/lishengping/mengqy/projects/maxtext`
  - TPU worker：`/home/lishengping/projects/maxtext`
- 脚本只负责训练控制，不替代实验类和 checkpoint/data provenance 记录。

## 迁移兼容

仓库外旧脚本在迁移前会保存到带 UTC 时间戳的备份目录。旧路径随后只作为兼容入口指向本目录，
避免出现两份可独立漂移的实现。
