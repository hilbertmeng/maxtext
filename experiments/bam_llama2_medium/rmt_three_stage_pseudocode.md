# NoO RMT：三个融合 kernel 的前传与解析反传

2026-09-27。对应当前实测选中版本，而非尚未实现的设计：

- v5p：`4dd7038fd2fd0a2c68c13c99a2da6fa43ba92f4f`，`RMTThreeStageMiddleChunk6458Profile`。
- v6e：`e30ec00197b2730764acea71f3bc56f642cf9274`，`RMTThreeStageMiddleRecomputeSavedV6eB4Profile`。
- 实现 worktree：`/data0/xd/rmt-pallas`；分支：`codex/rmt-pallas`。实现已合入主分支 `refactor-bam`；以上 hash 保留为原始实测版本。

以下省略 BF16 的逐条 cast、物理 padding 和分布式分片，只保留数学运算、融合边界、分块和保存／重算策略。数学等价不表示逐位 BF16 等价。所有反传均为显式解析式；没有用 `jax.vjp` 代替热路径。共用子程序均在所在 Pallas kernel 内展开，不是额外 kernel launch。

## 整层连接关系

```text
Forward:
  (Q, K, V, xA) = Kernel1_AttentionRead(M)
  DA            = Attention(Q, K, V)          # 独立 attention 实现
  (MA, Y, xF)   = Kernel2_AttentionWriteMLPRead(M, xA, DA)
  DF            = reshape(MLP(flatten(Y)))    # 独立 MLP 实现
  Mnext         = Kernel3_MLPWrite(MA, xF, DF)

Backward:
  (δMA, δxF, δDF) = Kernel3_Backward(δMnext)
  δY              = MLP_Backward(δDF)
  (δM_write, δxA, δDA) = Kernel2_Backward(δMA, δY, δxF)
  (δQ, δK, δV)    = Attention_Backward(δDA)
  δM_read         = Kernel1_Backward(δQ, δK, δV, δxA)
  δM              = δM_write + δM_read        # M 有两条消费者路径
```

三段是 memory 读写主干；不包括 attention softmax、MLP 大 GEMM、优化器和共享参数梯度的最终归约。NoO 表示 DA 中没有额外 LocalO 读取项。

## 记号与执行约定

下面公式写单个 token；实际对一个 token tile 向量化执行。不同 token 的这些 memory 读写相互独立，跨 token 混合发生在中间的 Attention。

|记号|含义／当前形状|
|---|---|
|`M`|每 token 的矩阵状态，`[k,v] = [48,75]`|
|`h, c, r, ρ, d`|`16, 32, 8, 4, 1200`；`c=k-h`，`d=h*v`|
|`M0, Mt`|`M[:h,:]` 与 `M[h:,:]`|
|`C`|压缩参数，`[c,r]`；`E = concat_rows(0[h,r], C)`|
|`S_A`|attention 静态读参数，`[3h,k]`|
|`R_F`|MLP 静态读参数，`[h,k]`；为源码参数的转置|
|`S`|静态写参数，`[h,k]`；attention 和 MLP 各有一套|
|`A, D, g`|动态写地址 `[h,k]`、原始 head 输出 `[h,v]`、动态写门 `[h]`|
|`δz`|loss 对 z 的梯度；`⊙` 为逐元素乘法|
|`ACC(θ, contribution)`|先沿 tile 的 token 求和，再累加到该 batch 的 FP32 参数梯度 partial|

为统一书写，投影矩阵采用 `输出维 × 输入维`，即 `p=W x`；可能与源码存储方向相反。`flatten/reshape` 不改变 head/value 的逻辑次序。

前传按 `(batch, token_tile)` 并行。反传对每个 batch 顺序遍历 token tiles，使共享参数梯度 partial 留在 VMEM 中累加；最后写回该 batch 的 partial，再在 kernel 外沿 batch／分片做所需归约。各算法中的 `ACC` 均遵循此约定，不是逐 token 写 HBM 或做全局原子加。

scan carry 的逻辑形状是 `[batch,token,k,v]`；wrapper 转置后，Pallas memory 主干内部主要采用 token-minor 布局 `[batch,k,v,token]`。逻辑转置不代表必然发生 HBM copy，需检查编译结果。写反传的联合收缩暂用 token-major 布局；Kernel2 内的转换只发生在片上临时量。

## 片上共用解析子程序

### RMS 归一化及 proxy

对长度 n 的向量 z：

```text
Nε(z):
  a = rsqrt(mean(z²) + ε)
  return a*z

NBε(z, u):                              # Nε(z) 的反传
  a = rsqrt(mean(z²) + ε)
  return a * (u - z * mean(u*z) * a²)

Proxy(M, s):
  z = flatten(M[:h,:])
  n = Nε(z)                              # 在整个 d=1200 上归一化
  return x=s⊙n, z, n

ProxyBackward(z, n, s, δx):
  ACC(s, δx⊙n)
  return reshape(NBε(z, s⊙δx), [h,v])
```

动态读 key 使用 `ε_read`；proxy 和动态写使用 `ε`。对矩阵 A、D 或 key 使用 N/NB 时，分别对每一行归一化。

### 压缩动态读

```text
Read8(CM, Z, g):                         # CM:[r,v], Z:[h,r]
  Zn = Nε_read(Z)
  U  = Zn @ CM
  return 0.2*g[:,None]⊙U

Read8Backward(CM, Z, g, δY):
  Zn  = Nε_read(Z)
  U   = Zn @ CM
  δU  = 0.2*g[:,None]⊙δY
  δg  = 0.2*sum_value(δY⊙U)
  δCM = Znᵀ @ δU
  δZ  = NBε_read(Z, δU @ CMᵀ)
  return δCM, δZ, δg
```

这是读运算的数学表达。实际小 rank 读采用逐 rank 累加，避免展开 `[h,r,v,tile]` 大中间量；V 读把 `0.2*g` 提前乘到归一化 key 上。

### 写地址／门网络

attention 和 MLP 分别有独立的 `Wd, Wu, Wg, bA, bg`。隐层宽度 256。下投影没有 bias。

```text
WriteProject(x):
  [z; l] = [Wd; Wg] @ x                  # 合并下投影与 gate 投影
  u = GELU(z)
  A = reshape(Wu @ u, [h,k]) + bA
  g = sigmoid(l + bg)
  return A, g, (z,u)

WriteProjectBackward(x, z, u, g, δA, δg):
  a = flatten(δA)
  ACC(Wu, a @ uᵀ);  ACC(bA, δA)
  δz = (Wuᵀ @ a) ⊙ GELU′(z)
  δl = δg ⊙ g ⊙ (1-g)
  δp = concat(δz, δl)
  ACC([Wd;Wg], δp @ xᵀ);  ACC(bg, δl)
  return [Wd;Wg]ᵀ @ δp
```

GELU 为实际使用的 tanh 近似：令 `t=tanh(√(2/π)*(z+0.044715*z³))`，则
`GELU′(z)=0.5*(1+t)+0.5*z*(1-t²)*√(2/π)*(1+3*0.044715*z²)`。

### 静态＋动态写及联合反传

```text
Write(M, A, D, g, S):
  An = Nε(A); Dn = Nε(D)
  Mout = M + Sᵀ @ D                      # 原始 D，静态写不乘动态门
  for head j = 0 ... h-1:
    Mout += g[j] * outer(An[j], Dn[j])
  return Mout

WriteBackward(A, D, g, S, G):            # G = δMout
  An = Nε(A); Dn = Nε(D)
  UA = Dn @ Gᵀ                           # [h,k]
  UD = (g[:,None]⊙An) @ G                 # [h,v]
  SD = S @ G                              # 静态写给 D 的梯度
  δS = D @ Gᵀ
  δg = sum_key(UA⊙An)
  δA = NBε(A, g[:,None]⊙UA)
  δD = NBε(D, UD) + SD
  ACC(S, δS)
  return δA, δD, δg                       # δMin=G，另走恒等路径
```

实际 `WriteBackward` 将上面四组收缩合到一次 batched MXU dot，而不是四次独立 kernel：

```text
Z = [[0[k,k], G],                        # [k+v,k+v]
     [Gᵀ,     0[v,v]]]
L = [[g⊙An, Dn],                         # h 行
     [S,     0 ],                        # h 行
     [0,     D ]]                        # h 行
P = L @ Z                                # 对 token 子块做 batched dot
UA = P[:h,:k];   UD = P[:h,k:]
SD = P[h:2h,k:]; δS = P[2h:,:k]
```

这里的 `k+v=123` 由 MXU 按物理布局处理 padding；大零块、P 都是片上临时量，不落 HBM。v5p Kernel2 将这一步按 64 token 子块执行。

## Algorithm 1 — Attention read 前传

**输入：** M、position、`S_A,C_A,s_A,W_A,bB,bg`。**输出：** Q、K、V、xA。

```text
for each (batch, tile of B1f tokens) in parallel:
  LOAD M, positions and shared parameters into on-chip memory
  xA, z, n = Proxy(M, s_A)

  p = W_A @ xA
  unpack p into B[ρ,c], T[2h,ρ], lQK[2h], ZV[h,r], lV[h], P[2h,18]
  B  += bB
  gQK = sigmoid(lQK + bg[:2h])
  gV  = sigmoid(lV  + bg[2h:])

  E = concat_rows(0[h,r], C_A)
  [Static; CM] = [S_A; Eᵀ] @ M            # 合并静态读与压缩读的 MXU dot

  U = B @ M[h:,:]                         # [ρ,v]；逐 rank 归约，不展开外积
  J = T @ B                               # [2h,c]，合成 QK 读 key
  F = T @ U                               # [2h,v]
  a = rsqrt(mean_key(J²) + ε_read)
  QK = Static[:2h,:] + (0.2*gQK*a)[:,None]⊙F

  QK[:,v-18:] = RoPE(P, position)          # 替换；不是相加
  Q = QK[:h,:] / sqrt(v)
  K = QK[h:,:]
  V = Static[2h:,:] + Read8(CM, ZV, gV)
  STORE Q, K, V, xA to HBM
```

反传 residual 保留输入 M、position 和参数引用；上述内部动态量在反传中片上重算。

## Algorithm 2 — Attention read 反传

**输入：** 上述 residual，`δQ,δK,δV,δxA_external`。**输出：** `δM_read` 和参数梯度 partial。

```text
for each batch in parallel:
  INIT on-chip FP32 shared-parameter gradient accumulators
  for each tile of B1b tokens:
    LOAD M, position, δQ, δK, δV, δxA_external and parameters
    RECOMPUTE Algorithm 1 intermediate state on chip

    H = concat_rows(δQ/sqrt(v), δK)
    δP = RoPEᵀ(H[:,v-18:], position)
    H[:,v-18:] = 0                        # 替换维度不向原 QK 读回传
    δStatic = concat_rows(H, δV)
    ACC(S_A, δStatic @ Mᵀ)
    δM = S_Aᵀ @ δStatic

    δCM, δZV, δgV = Read8Backward(CM, ZV, gV, δV)
    ACC(C_A, M[h:,:] @ δCMᵀ)
    δM[h:,:] += C_A @ δCM

    δF = (0.2*gQK*a)[:,None]⊙H
    e = sum_value(H⊙F)
    δgQK = 0.2*a⊙e
    δJ = -(0.2*gQK*e*a³/c)[:,None]⊙J     # 读 key 归一化分母的路径
    δU = Tᵀ @ δF
    δT = δF @ Uᵀ + δJ @ Bᵀ
    δB = δU @ M[h:,:]ᵀ + Tᵀ @ δJ
    δM[h:,:] += Bᵀ @ δU

    δlQK = δgQK⊙gQK⊙(1-gQK)
    δlV  = δgV ⊙gV ⊙(1-gV)
    δp = pack(δB, δT, δlQK, δZV, δlV, δP)
    ACC(bB, δB);  ACC(bg, concat(δlQK, δlV))
    ACC(W_A, δp @ xAᵀ)
    δxA = δxA_external + W_Aᵀ @ δp
    δM[:h,:] += ProxyBackward(z, n, s_A, δxA)
    STORE δM_read = δM
  FLUSH shared-parameter gradient partials for this batch
```

`δU,δT,δB,δM` 的 rank 路径在实现中逐 rank 求和，避免构造高阶广播张量。K1 的静态／压缩读前传合并，反传仍按上述各路计算；不要将 K2 的合并线性反传误写到 K1。

## Algorithm 3 — Attention write + MLP read 前传

**输入：** `M,xA,DA`、attention 写参数、MLP 读参数。**输出：** `MA,Y,xF`。

```text
for each (batch, tile of B2f tokens) in parallel:
  LOAD M, xA, DA and parameters
  A, g, _ = WriteProject_A(xA)
  MA = Write(M, A, DA, g, S_writeA)
                                           # MA 留片上，立刻接 MLP 读
  xF, zF, nF = Proxy(MA, s_F)
  [flatten(ZF); lF] = W_F @ xF             # key 与读门投影合并
  gF = sigmoid(lF + bF)
  CM = C_Fᵀ @ MA[h:,:]
  StaticF = R_F @ MA                      # 当前实现为两次片上 dot
  Y = StaticF + Read8(CM, ZF, gF)
  STORE MA, Y, xF to HBM                  # 没有写、读之间的 HBM 往返

  v5p backward residual: retain MA, xA, DA and parameter references
  v6e backward residual: retain M,  xA, DA and parameter references
```

两种配置均保留 Y、xF 的 checkpoint，避免外层为获得这两个输出重跑整个中段。v6e 仍需输出 MA 给 K3；“不保留 MA 作为中段 residual”不表示前传不输出 MA。

## Algorithm 4 — Attention write + MLP read 反传

**输入：** 中段 residual、`G_external=δMA`、`δY`、`δxF_external`。**输出：** `δM_write,δxA,δDA` 和参数梯度 partial。

```text
for each batch in parallel:
  INIT on-chip FP32 shared-parameter gradient accumulators
  for each tile of B2b tokens:
    LOAD saved matrix, xA, DA, G_external, δY, δxF_external and parameters

    if v6e:
      A, g, (zw,uw) = WriteProject_A(xA)
      MA = Write(saved_M, A, DA, g, S_writeA)
                                           # 只在片上重做写；网络状态供后面复用
    else:                                  # v5p
      MA = saved_MA

    RECOMPUTE MLP read state xF,zF,nF,ZF,gF,CM from MA
    δCM, δZF, δgF = Read8Backward(CM, ZF, gF, δY)

    E = concat_rows(0[h,r], C_F)
    L = concat_rows(R_F, Eᵀ)
    H = concat_rows(δY, δCM)
    δL = H @ MAᵀ                          # 合并静态读、压缩读的参数梯度 dot
    ACC(R_F, δL[:h,:])
    ACC(C_F, δL[h:,h:]ᵀ)                  # E 顶部的零行不是参数
    G = G_external + Lᵀ @ H               # 合并两路对 MA 的线性反传

    δlF = δgF⊙gF⊙(1-gF)
    δpF = concat(flatten(δZF), δlF)
    ACC(W_F, δpF @ xFᵀ);  ACC(bF, δlF)
    δxF = δxF_external + W_Fᵀ @ δpF
    G[:h,:] += ProxyBackward(zF, nF, s_F, δxF)
                                           # 三个输出的梯度到此汇齐
    RELEASE read temporaries after their last use

    if v5p:
      A, g, (zw,uw) = WriteProject_A(xA)    # 推迟网络计算，缩短 live range
      for token subchunk of 64 within this 128-token tile:
        δA[sub], δDA[sub], δg[sub] =
          WriteBackward(A[sub], DA[sub], g[sub], S_writeA, G[sub])
    else:
      δA, δDA, δg = WriteBackward(A, DA, g, S_writeA, G)

    δxA = WriteProjectBackward_A(xA, zw, uw, g, δA, δg)
    STORE δM_write=G, δxA, δDA
  FLUSH shared-parameter gradient partials for this batch
```

`RELEASE` 表示应在最后使用后结束中间量生命周期，供编译器复用 VMEM；不是声称源码有显式内存 free。中间的 G 直接供写反传使用，不先存 HBM 再由另一 kernel 读回。输出 δM 可以与输入 G_external 合法 alias，但不能覆盖仍存活、仍需读取的输入。

## Algorithm 5 — MLP write 前传

**输入：** `MA,xF,DF`、MLP 写参数。**输出：** `Mnext`。

```text
for each (batch, tile of B3f tokens) in parallel:
  LOAD MA, xF, DF and parameters
  A, g, _ = WriteProject_F(xF)
  Mnext = Write(MA, A, DF, g, S_writeF)
  STORE Mnext to HBM

Backward residual: xF, DF and parameter references
                    # 不需要 MA：本段对 MA 的局部导数为恒等映射
```

## Algorithm 6 — MLP write 反传

**输入：** `xF,DF`、参数、`G=δMnext`。**输出：** `δMA,δxF,δDF` 和参数梯度 partial。

```text
δMA = G                                   # 恒等直通，不必为此单独读写一份 M
for each batch in parallel:
  INIT on-chip FP32 shared-parameter gradient accumulators
  for each tile of B3b tokens:
    LOAD xF, DF, G and parameters           # 不读前传的 MA
    A, g, (z,u) = WriteProject_F(xF)
    δA, δDF, δg = WriteBackward(A, DF, g, S_writeF, G)
    δxF = WriteProjectBackward_F(xF, z, u, g, δA, δg)
    STORE δxF, δDF
  FLUSH shared-parameter gradient partials for this batch
```

## 当前实测选中的分块与保存策略

|项目|v5p|v6e|
|---|---:|---:|
|K1 前传／反传 token tile|128 / 128|128 / 256|
|K2 前传／反传外层 token tile|128 / 128|128 / 128|
|K2 写反传内部 token subchunk|64|128（整个外层 tile）|
|K3 前传／反传 token tile|128 / 128|256 / 256|
|K2 backward matrix residual|保存 MA|保留 M，片上重建 MA|
|K2 输出 Y、xF checkpoint|保存|保存|
|scoped VMEM 预算|58 MiB|96 MiB|
|物理 VMEM／TensorCore|64 MiB|128 MiB|

三个 kernel 的前反传 tile 可独立选择。这里的 token subchunk 是 token-local memory 写反传的分块，不是 attention 的 Q/K/V 序列分块。

三段前传只在最终输出处写 HBM；内部临时量留片上。反传保存／重算如上所示，但外围 layer scan 的 checkpoint 策略仍可能重跑其他前传段；这里没有宣称整层完全取消 remat。静态权重、DMA 缓冲、FP32 参数梯度 partial、MXU padding 与临时结果均占 VMEM，不能只按一个 M tile 的字节数估算峰值。

## 实现对应

以下路径相对 `/data0/xd/rmt-pallas`；精确版本见本文开头的 commit。

- K1：`MaxText/layers/rmt_pallas_attention_read.py`，V 分支：`rmt_pallas_v_read.py`。
- K2 前传／residual：`MaxText/layers/rmt_pallas_full_write_read.py`。
- K2 反传调度：`MaxText/layers/rmt_pallas_full_write_read_minor.py`；读解析反传：`rmt_pallas_write_read.py`。
- K3、写投影解析梯度：`MaxText/layers/rmt_pallas_projected_write.py`。
- 写前传：`MaxText/layers/rmt_pallas_minor.py`；联合写反传及 qchunk：`rmt_pallas_write_reverse.py`。
- 整层接线：`MaxText/layers/rmt.py`。

原始实测、数值检查和未选中变体见 [rmt_pallas.md](rmt_pallas.md)。本文是数学与调度伪代码，不替代 BF16 实现中的 cast／归约顺序，也不表示已经完成长期训练收敛验证。


## Rank-H 写更新与寄存器分块后续（独立 worktree）

2026-09-28 后续实现见 `/data0/xd/rmt-pallas-rankh`，分支
`codex/rmt-pallas-rankh`；完整测速与选型见 [rmt_rankh_write.md](rmt_rankh_write.md)。
上面的三段边界不变，K2/K3 中的 `WriteForward/Backward` 可替换为以下形式。
仅适用于静态、动态项共用 D 的层写入，不套用到 embedding seed write。

```text
Algorithm: RankHWriteForward(M, A, D, g, S)
  An = RMSNorm(A); rD = rsqrt(mean(D²) + ε)
  C = S + (g · rD) · An
  for key row k:
    acc[v, token] = 0
    for head j:
      acc += C[j,k,token] · D[j,v,token]
    Mout[k,v,token] = M[k,v,token] + acc
  return Mout

Algorithm: RankHWriteBackward(A, D, g, S, G)
  recompute An, rA, rD, C
  δC = G D                       # sum over v
  δDdir = Gᵀ C                   # sum over k
  b = sum_k(δC · An)
  δA = RMSBackward(A, (g · rD) · δC)
  δD = δDdir − (g · rD³ / V) · b · D
  δg = rD · b
  δS = sum_tokens(δC)
  return δA, δD, δg, δS

Algorithm: RegisterBlockedContractions(C, D, G)
  # token occupies128 SIMD lanes; k/v outputs occupy sublanes.
  # V=75 padded to80; two heads share each loaded G row.
  stage G[k,v,token] and Gt[v,k,token] in VMEM
  for two-head group J:
    δC[J,k,token] = 0
    for v:                       # selected candidate fully unrolls75 terms
      row = Gt[v,k,token]
      for j in J: δC[j] += D[j,v,token] · row
    store δC[J]
  for two-head group J:
    δDdir[J,v,token] = 0
    for k:                       # fully unroll48 terms
      row = G[k,v,token]
      for j in J: δDdir[j] += C[j,k,token] · row
    store δDdir[J]
```

实际 v6e 指令是独立 FP32 乘、加，不假设 fused FMA。MXU 备选用 NN+NT
两次收缩，不构造包含零象限的对称矩阵。VPU 版保持写投影、归一化、门控、
两组收缩及其 epilogue 的 token-minor 布局；以上子程序都内联在完整 K2/K3
kernel 内，不各自发起 kernel。BF16 重排并非逐位等价，数值检查见测速报告。
