gs://llm_base_models_europe-west4/v5p_256/7B/xm_45x7B_moe_0922  dense模型：488000， base: 10000
基于自设计初始化方案进行moe训练：去掉mgate。将dense的mlp分为44份（非共享专家），每份128，将每份的128维度拓展为5632（44个非共享专家），然后加上dense mlp。
共享专家直接用dense mlp初始化，也是去掉了mgate


gs://llm_base_models_europe-west4/v5p_256/7B/xm_45x7B_moe_base500k_1022/  dense模型：488000， base: 500000
基于自设计初始化方案进行moe训练：去掉mgate。将dense的mlp分为44份（非共享专家），每份128，将每份的128维度拓展为5632（44个非共享专家），然后加上dense mlp。
共享专家直接用dense mlp初始化，也是去掉了mgate

gs://llm_base_models_europe-west4/v5p_256/7B/xm_45x7B_moe_1114/  dense模型：488000， base: 500000
基于常用的moe初始化方案进行moe初始化：将dense mlp复制45份，分配给44个非共享专家，1个共享专家，注意：这里的mgate保留了

gs://llm_base_models_europe-west4/v5p_256/7B/xm_45x7B_moe_base500k_1022/  440000 base: 10000
基于自设计初始化方案进行moe训练：保留mgate。将dense的mlp分为44份（非共享专家），每份128，将每份的128维度拓展为5632（44个非共享专家），然后加上dense mlp。
共享专家直接用dense mlp初始化，保留了mgate。这个和0922的方案相比区别在于保留了mgate，基座模型为440000



