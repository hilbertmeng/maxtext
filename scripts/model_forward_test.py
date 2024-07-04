import os
import sys
import yaml

sys.path.append('/home/lishengping/projects/maxtext/MaxText')
os.environ['HARDWARE'] = 'tpu'

import pyconfig

config_name = '/home/lishengping/projects/maxtext/MaxText/configs/dcformer_pp_405m.yml'
argv = [None, config_name]
pyconfig.initialize(argv)
config = pyconfig.config
# validate_train_config(config)

from layers import models
import max_utils
import jax
import orbax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax.traverse_util import flatten_dict, unflatten_dict
from flax import linen as nn
from transformers import AutoTokenizer

TOKENIZER_PATH = '/home/lishengping/tokenizer'

tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_PATH, use_fast=True, trust_remote_code=True
        )
read_dir = "gs://llm_base_models/maxtext_align_pax_dc/maxtext_align2/checkpoints"
step_prefix = "checkpoint"
step_format_fixed_length = None
# load_step = 6600

options = orbax.checkpoint.CheckpointManagerOptions()
item = {
    "state": orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
}
max_mngr = orbax.checkpoint.CheckpointManager(read_dir, item, options)
state = max_mngr.restore(2200)
params = state['state']['params']

flat_params = flatten_dict(params)
for k, v in flat_params.items():
    print(k, v.shape)


inp = '''<|extra_0|>一个则非常实用，而且轻便，扛在肩上没有负担。\n片断五\n诵读：阿宽 如喜 朗月 张凤霞\n薛如刚 云在飞 牧歌\n彼得大叔来了\n和平到来后的那个温暖的夏天,塔拉突然间失去了往昔的宁静。接下来的几个月里,一队队士兵拖着艰难的脚步,吃力地翻过那座红色的山丘来到塔拉,在门前台阶的阴凉处歇息,衣衫槛楼、胡子拉碴、步履蹒跚、饥肠辘辘,盼望得到食物,想要投宿一夜。他们是正在返家的南军士兵。火车将约翰斯顿残部的士兵从北卡罗来纳州运送到亚特兰大,将他们扔在那儿,从此,这些士兵们就开始了徒步跋涉,走上回家的路。可是,他们的家或许已经不复存在,家里的人也许是死的死,散的散了。\n他们对眼前的困难不屑一顾,一切都结束了。回家!回家!这是士兵们脑子里的惟一念头。他们正在往家里赶,这是他们惟一的支撑。他们打仗已经尽了全力,结果被打败了,如今他们很愿意安定下来,在他们反对过的旗帜下安居乐业。\n斯佳丽和玫兰尼向每一个士兵急切地打听阿希礼的消息。但是谁也没听说过他的消息,而且他们也不愿谈起失踪人员的事情。他们自己活着就足够了,他们不关心也不愿去想那数以千计躺在无名坟慕里永远也回不了家的人。\n每次失望后,家里人都努力安慰玫兰妮.让她保持信心。阿希礼肯定没有死在俘虏营中。否则北佬的牧师会写信通知他们的。他准是在回家的路上,不过他的俘虏营离得那么远。天啊,这么远的路火车都得走好几天,要是阿希礼和这些人一样步行的话......\n六月的一个下午,塔拉的人都聚集在后门廊,热切地看着波克切开这年第一个半生不熟的西瓜,他们听到屋前的碎石路上传来了马蹄声。普莉西不情愿地朝前门走去,其他人则在她身后激烈地讨论,要是来者是个当兵的,他们是该把西瓜藏起来呢还是留下来晚饭时招待客人。\n波克抱着小西瓜站在那里,不知所措,这时他们听到普莉西的喊声。\n“老天爷啊!斯佳丽小姐!玫荔小姐!快来!”\n“是谁啊?”斯佳丽一边喊,一边从台阶上跳了起来,穿过厅堂朝外冲去,玫荔紧跟在她的身后,其他人也都跟着往外跑。\n“是阿希礼!”斯佳丽心想。“哦,可能.....\n“是彼得大叔!佩蒂小姐家的彼得大叔!”大家都跑到了前门廊,看见那个高个子、灰白头发的老管家正从一匹绑着被子当马鞍、长着一条老鼠尾巴的老马背上往下爬。他那张宽宽的黑脸上总是摆出一副很有尊严的表情,现在看见了老朋友虽然非常高兴却又不想放弃尊严,结果是他的眉头紧锁,嘴巴却咧开'''
batch_size = 1
input_ids = jnp.array(tokenizer.encode(inp)).reshape(batch_size, -1)
data = {}
data['inputs'] = input_ids[:, :-1]
data["inputs_position"] = jnp.arange(data['inputs'].shape[1]).reshape(batch_size, -1)
data["inputs_segmentation"] = jnp.ones_like(data['inputs'])
data["targets"] = input_ids[:, 1:]


quant = None
devices_array = max_utils.create_device_mesh(config)
mesh = Mesh(devices_array, config.mesh_axes)
Transformer = models.Transformer
model = Transformer(config, mesh, quant=quant)

is_train = False
rng1, aqt_rng = jax.random.split(jax.random.key(9876))

logits, intermediate_outputs = model.apply(
      {'params': params},
      data["inputs"],
      data["inputs_position"],
      decoder_segment_ids=data["inputs_segmentation"],
      enable_dropout=config.enable_dropout if is_train else False,
      rngs={"dropout": rng1, "params": aqt_rng},
      mutable="intermediates",
  )
one_hot_targets = jax.nn.one_hot(data["targets"], config.vocab_size)
xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets, 0.0)
xent = nn.with_logical_constraint(xent, ("activation_embed_and_logits_batch", "activation_length"))