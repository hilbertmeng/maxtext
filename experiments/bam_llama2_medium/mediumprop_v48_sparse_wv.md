# MediumProp V48：W_V 放在独立 MLP→M 写入之前

父版：BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile，runtime c1a68fe，完成13500步。
工作树 / 分支：/data0/xd/mediumprop-k75-embed / codex/mediumprop-k75-embed。

新两路：
- BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstKeepVOTruePile
- BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstNoVOTruePile

W_V恢复到零起始0/3/6/9/12/15；独立MLP写入仍在1/4/7/10/13/16。KeepVO在恢复W_V层仍将LocalV加到标准V，LocalO正常读写。NoVO仅这些层去掉LocalV/O和对应C12压缩读，LocalQK、W_O、attention写M全部保留。其他层不变。D1200、18层、16头、M75×48/C12、地址R384、QK57+RoPE18、TruePile T4096不变。

W_Q=1,440,000。KeepVO新增6W_Q，按层扣MLP：[3365,3550,3765]，总432115072，与父版相同，MHA-6128。NoVO每个W_V层移除270944参数=.188156W_Q；净新增每层1169056=.811844W_Q；宽度[3440,3550,3765]，总432109408，MHA-11792。按整数宽度精确最近抵扣，无硬件取整。

下注：终局相对父版KeepVO -.006、NoVO -.012，NoVO-KeepVO -.006。速度.508/.520 step/s；父版.503，同基础/concat/write健康设置，NoVO删除VO的对应统计随结构消失。依据：标准V负责新内容生成，下一层独立MLP写入负责更新M；删重复VO可补回较多MLP。若两路都劣于父版，说明此处恢复W_V的内容通路不足以抵消MLP损失。

TPU拟定xd-v5p-16-2910075-maxtext、xd-v5p-16-2910076-maxtext；主区UE5a，持续无资源时可加排EW4b/UC1a。13500步、200步窗口、1000步批量报告，2800/5000判定。编译借用EW4a FLEX_START llm-jax-v6e-1-0，禁止自动回收。

验证：参数树和各层开关、六个MLP写入位置、bool旧父版等价、拒绝非周期scan开关、两block有限前向/梯度并确认标准V和独立写地址获梯度。sealed runtime及实际AOT加载/FIRST_STEP为上线关口。
产物：/data0/xd/bam_diagnostics/rmt-readnorm-launch/wv-*。

运行源码09da7cd491633d2ceba241509a8284f318d149e7。CPU针对性3项均过（约51s），共享路径通用47项均过（4组并行136.7s）；KeepVO v5p16 AOT已通过，NoVO随后在同一保护编译机串行编译。CPU与正式训练排队仍并行。

KeepVO确认FIRST_STEP7、AOT loaded、worker commit09da7cd、W_V层开关/MLP宽度、区内TruePile路径与父版WD/健康设置。步骤60–71速度中位.501step/s（父版.503，−.4%），近持平。NoVO AOT已通过，UE5a节点尚在创建。

UE5a首租：KeepVO 14:59:18–15:11:34 UTC（736s），曾正常训练；NoVO 15:10:46–15:13:26 UTC（160s），安装时抢占，尚无FIRST_STEP。两路原区恢复队列保留；NoVO另加EW4b被动候选xd-v5p-16-2910076-maxtext（独立creator PID），待实际READY再选择。抢占信息不进入台账结论。
