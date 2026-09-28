"""生成第 01–05 章的 Notebook。"""
from nbcommon import (md, code, blank, img, header, goals, part1, part2, part3, save,
                        testkit, test, summary_cell, deps)

# ══════════════════════════════════════════════════════════════
# 01 项目总览与 uv 环境
# ══════════════════════════════════════════════════════════════
c = [
    header(1, "项目总览与 uv 环境", "第一部分 · 基础", "01_overview.html",
           "这一章不写模型，只做两件事：**看清 MiniMind 的全貌**，以及**把环境搭好**。后面 14 章都会在这套环境里跑。"),
    goals([
        "理解 MiniMind 的完整训练流水线（分词 → 预训练 → SFT → 对齐/RL）",
        "掌握 `pathlib`、`dataclasses`、随机种子等工程基础",
        "会用 uv 创建环境、安装依赖、启动 JupyterLab",
        "能自己写出一个配置类并统计模型参数量",
    ]),
    md("""## 📖 背景：MiniMind 是什么

MiniMind 用极简代码实现了一个完整的语言模型训练链路，规模只有 **64M 参数**，
单张消费级显卡、几小时就能跑完。它的价值在于——**短到你能逐行读懂**。

| 阶段 | 脚本 | 做什么 |
|---|---|---|
| 分词 | train_tokenizer.py | 训练 6400 词表的 BPE |
| 预训练 | train_pretrain.py | 下一词预测，学语言规律 |
| 指令微调 | train_full_sft.py | 学会对话 / 问答 / 工具调用 |
| LoRA | train_lora.py | 冻结底座，只训低秩增量 |
| 蒸馏 | train_distillation.py | 用大模型当老师 |
| 偏好对齐 | train_dpo.py | chosen / rejected 对比 |
| 强化学习 | train_ppo / grpo.py | 用奖励优化策略 |
| 智能体 RL | train_agent.py | 多轮工具调用轨迹 |

> 记住主线：**预训练让模型会“接话”，SFT 让它会“听话”**，其余都是可选增强。

---

"""),
    img(
        ("../images/LLM-structure.jpg", "MiniMind 稠密模型结构（Transformer Decoder-Only，对齐 Qwen3）"),
        ("../images/LLM-structure-moe.jpg", "MiniMind MoE 模型结构（前馈层替换为专家混合）"),
        ("../images/dataset.jpg", "各阶段数据集的构成与推荐组合"),
        ("../images/gpt3_config.png", "GPT-3 参数配置参考（MiniMind 的缩放对照）"),
        ("../images/benchmark_radar.jpg", "MiniMind 在第三方评测上的表现"),
    ),
    img(
        ("../images/logo2.png", "MiniMind 项目标识"),
        ("../images/with_huggingface.png", "模型权重也发布在 Hugging Face 上，可直接下载对照"),
        ("../images/with_modelscope.png", "国内可通过 ModelScope 获取权重与数据集"),
    ),
    part1("下面这些是最基础的工程工具，写任何深度学习项目都会用到。"),
    md("### 1.1 检查环境：`sys` 与版本信息\n\n先确认 Python 和 PyTorch 的版本。深度学习对版本很敏感。"),
    code("""import sys, platform
print("Python :", sys.version.split()[0])
print("平台   :", platform.platform())

try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA 可用:", torch.cuda.is_available())
    print("设备   :", "cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    print("PyTorch 未安装 —— 请在项目目录执行：uv sync")"""),
    md("""### 1.2 路径处理：`pathlib.Path`

比字符串拼路径更安全，跨平台。深度学习项目里到处都要读写数据集、权重文件。"""),
    code("""from pathlib import Path

p = Path("..") / "notebooks"          # 用 / 拼路径
print("路径       :", p)
print("绝对路径   :", p.resolve())
print("是否存在   :", p.exists())
print("后缀       :", Path("model.pth").suffix)
print("文件名     :", Path("a/b/model.pth").stem)"""),
    md("""### 1.3 配置对象：`dataclasses.dataclass`

MiniMind 的 `MiniMindConfig` 就是一个配置容器。`dataclass` 让你少写一堆 `__init__`。"""),
    code("""from dataclasses import dataclass, field
import math

@dataclass
class DemoConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    vocab_size: int = 6400
    head_dim: int = field(init=False)

    def __post_init__(self):
        # 每个注意力头的维度 = hidden / heads
        self.head_dim = self.hidden_size // self.num_attention_heads

cfg = DemoConfig()
print(cfg)
print("head_dim =", cfg.head_dim)
# MiniMind 的 intermediate_size 会对齐到 64 的倍数，让矩阵乘法更快
print("intermediate_size =", math.ceil(768 * math.pi / 64) * 64)"""),
    md("""### 1.4 随机种子：让实验可复现

同样的代码跑两次结果不同，是深度学习里最让人头疼的事之一。固定种子能解决大部分问题。"""),
    code("""import random
import numpy as np
import torch

def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

setup_seed(42)
a = torch.randn(3)
setup_seed(42)
b = torch.randn(3)
print("两次随机结果是否一致:", torch.allclose(a, b))"""),
    part2("下面两件事是本章要你独立完成的。提示足够，答案不给。\n\n> 💡 每个留白代码块后面都跟了一个**测试块**。先写代码，再运行测试块；\n> 没写之前运行也不会报错，只会提示「未实现」。"),
    testkit(),
    md("""### 2.1 写一个 `MiniMindConfig`

参考上面的 `DemoConfig`，写出一个更完整的配置类。

**要求**：包含 `hidden_size=768`、`num_hidden_layers=8`、`num_attention_heads=8`、
`num_key_value_heads=4`、`vocab_size=6400`、`rope_theta=1e6`、`rms_norm_eps=1e-6`、
`tie_word_embeddings=True`，并在 `__post_init__` 里算出 `head_dim` 和 `intermediate_size`。"""),
    blank("MiniMindConfig", [
        "字段：hidden_size / num_hidden_layers / num_attention_heads / num_key_value_heads",
        "字段：vocab_size / rope_theta / rms_norm_eps / tie_word_embeddings",
        "在 __post_init__ 中计算 head_dim = hidden_size // num_attention_heads",
        "在 __post_init__ 中计算 intermediate_size = ceil(hidden_size * π / 64) * 64",
    ], [
        "`@dataclass`、`field(init=False)`",
        "`math.ceil`、`math.pi`",
    ]),
    test("MiniMindConfig", '''def _t():
    c = MiniMindConfig()
    assert c.hidden_size == 768, f"hidden_size 应为 768，得到 {c.hidden_size}"
    assert c.num_hidden_layers == 8, f"num_hidden_layers 应为 8，得到 {c.num_hidden_layers}"
    assert c.head_dim == 96, f"head_dim 应为 768 // 8 = 96，得到 {c.head_dim}"
    assert c.intermediate_size == 2432, f"intermediate_size 应为 ceil(768*pi/64)*64 = 2432，得到 {c.intermediate_size}"
    assert c.rope_theta == 1e6, f"rope_theta 应为 1e6，得到 {c.rope_theta}"
    assert c.tie_word_embeddings is True, "tie_word_embeddings 应为 True"

check("MiniMindConfig 字段与推导值", _t)'''),
    md("""### 2.2 参数量统计函数

写一个函数，统计一个 `nn.Module` 的总参数量（单位：百万 M），并分别打印每个子模块的参数量。

**提示**：用 `model.named_children()` 或 `model.named_modules()` 遍历，
每个参数张量的元素个数是 `p.numel()`。"""),
    blank("count_parameters", [
        "输入：一个 nn.Module",
        "返回：总参数量（整数，即所有参数元素个数之和）",
        "额外：打印每个直接子模块的参数量，以及总量（以 M 为单位，保留 2 位小数）",
    ], [
        "`sum(p.numel() for p in model.parameters())`",
        "`model.named_children()`",
    ]),
    test("count_parameters", '''def _t():
    from torch import nn
    m = nn.Sequential(nn.Linear(4, 8, bias=False), nn.Linear(8, 2, bias=False))
    n = count_parameters(m)
    assert n == 4 * 8 + 8 * 2, f"参数量应为 48，得到 {n}"

check("count_parameters 统计正确", _t)'''),
    md("""### 2.3 uv 常用命令（在终端执行，不是在这里）

```bash
uv sync                 # 按 pyproject.toml 创建/更新环境
uv run jupyter lab      # 在项目环境里启动 JupyterLab
uv add matplotlib       # 加一个依赖
uv add --dev pytest     # 加开发依赖
uv sync --extra train   # 安装可选依赖组（训练用）
uv lock --upgrade       # 升级依赖并重写锁文件
```

> 为什么用 uv：它比 pip 快一个数量级，并且 `uv.lock` 保证你和别人装出**完全一致**的版本。"""),
    summary_cell(),
    part3([
        "能说清 MiniMind 的 8 个训练阶段以及它们的先后顺序",
        "能解释为什么需要固定随机种子",
        "`MiniMindConfig` 的 `head_dim` 和 `intermediate_size` 算对了吗？",
        "会用 `uv sync` / `uv run` 管理本项目环境",
    ], "**思考题**：如果把 `num_attention_heads` 从 8 改成 16，`head_dim` 会变成多少？这会影响注意力的计算量吗？"),
]

# ══════════════════════════════════════════════════════════════
# 02 分词器与数据流水线
# ══════════════════════════════════════════════════════════════
c2 = [
    header(2, "分词器与数据流水线", "第一部分 · 基础", "02_tokenizer.html",
           "模型看不见文字，只能看见整数。这一章我们先把「文字 ↔ 整数」的桥梁搭好，再把数据组织成模型能吃的 batch。"),
    goals([
        "理解 BPE 子词分词的核心思想",
        "会用 `collections.Counter` 做统计、会手写一次 BPE 合并",
        "掌握 `tokenizers` 库训练一个小词表，以及 `AutoTokenizer` 的用法",
        "会用 `torch.utils.data.Dataset` + `DataLoader` 组装训练数据",
    ]),
    md("""## 📖 背景：为什么是子词（subword）

- 按**字符**切：序列太长，模型要记很久才能到下一个词。
- 按**词**切：词表爆炸，遇到没见过的词就崩（OOV）。
- 按**子词**切：折中。常见片段合成一个 token，罕见词拆成小片段。

MiniMind 用 **BPE**，词表只有 **6400**。此外还有几个特殊 token 只做结构标记：
`<|im_start|>` / `<|im_end|>`（对话边界）、`<think>`（思维链）、`<tool_call>`（工具调用）。

---

"""),
    img(
        ("../images/dataset.jpg", "MiniMind 各阶段数据集的构成——分词质量直接决定这些数据能被模型利用多少"),
    ),
    part1(),
    md("### 2.1 字符与编码：`ord` / `chr` / `encode`\n\n分词的最底层就是字节和字符。"),
    code("""s = "猫"
print("字符     :", s)
print("Unicode  :", hex(ord(s)))
print("UTF-8 字节:", list(s.encode("utf-8")))
print("转回字符 :", chr(ord(s)))

# 中文一个字通常占 3 个 UTF-8 字节
print("字节长度 :", len(s.encode("utf-8")))"""),
    md("""### 2.2 统计频次：`collections.Counter`

BPE 的第一步就是数「哪个符号对出现得最多」。"""),
    code("""from collections import Counter

words = ["low", "lower", "newest", "widest", "low"]
cnt = Counter(words)
print("词频:", dict(cnt))
print("最常见的 2 个:", cnt.most_common(2))

# 统计“相邻字符对”的频次
pairs = Counter()
for w, n in cnt.items():
    for i in range(len(w) - 1):
        pairs[w[i:i+2]] += n
print("相邻对频次:", dict(pairs))
print("最频繁的相邻对:", pairs.most_common(1))"""),
    md("""### 2.3 手写一次 BPE 合并

BPE 的规则只有一句：**反复把出现次数最多的相邻符号对合并成一个新符号**。"""),
    code("""def merge_pair(word_list, a, b):
    \"\"\"把 word_list 中所有相邻的 (a, b) 合并成 a+b\"\"\"
    merged = a + b
    out = []
    for w in word_list:
        new_w, i = [], 0
        while i < len(w):
            if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                new_w.append(merged); i += 2
            else:
                new_w.append(w[i]); i += 1
        out.append(new_w)
    return out

# 把每个词拆成字符
corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
words = [list(w) for w in corpus]
print("初始:", words)

words = merge_pair(words, "e", "s")     # 合并 e+s → es
print("合并 es:", words)
words = merge_pair(words, "es", "t")    # 再合并 es+t → est
print("合并 est:", words)"""),
    md("""### 2.4 用 `tokenizers` 训练一个小 BPE

真实项目不会手写 BPE，而是用 `tokenizers` 库。下面在一个小语料上训练一个迷你词表。"""),
    code("""try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False
    print("未安装 tokenizers —— 请先在项目目录执行 `uv sync`，然后重启内核。")

if HAS_TOKENIZERS:
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=120,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    )
    corpus_text = [
        "minimind is a tiny language model",
        "train a language model from scratch",
        "the cat sat on the mat",
        "a small model can still learn to chat",
    ]
    tok.train_from_iterator(corpus_text, trainer=trainer)

    enc = tok.encode("minimind is a tiny model")
    print("词表大小:", tok.get_vocab_size())
    print("tokens  :", enc.tokens)
    print("ids     :", enc.ids)"""),
    md("""### 2.5 加载 MiniMind 官方分词器（可选）

如果已经下载了 MiniMind 的 `tokenizer.json`，可以直接加载。否则跳过这一格。"""),
    code("""from pathlib import Path

# 官方分词器已放在 ../data/ 下，可直接使用
tok_path = Path("../data/tokenizer.json")

if not tok_path.exists():
    print("未找到 ../data/tokenizer.json —— 跳过。可从 MiniMind 仓库 model/ 目录复制过来。")
else:
    loaded = False
    try:
        from transformers import AutoTokenizer
        mm_tok = AutoTokenizer.from_pretrained("../data")
        ids = mm_tok("你好，MiniMind！").input_ids
        print("MiniMind 词表大小:", mm_tok.vocab_size)
        print("编码:", ids)
        print("解码:", mm_tok.decode(ids))
        loaded = True
    except ImportError:
        pass

    if not loaded:
        try:
            from tokenizers import Tokenizer
            mm_tok = Tokenizer.from_file(str(tok_path))
            enc = mm_tok.encode("你好，MiniMind！")
            print("MiniMind 词表大小:", mm_tok.get_vocab_size())
            print("tokens:", enc.tokens)
            print("ids   :", enc.ids)
            loaded = True
        except ImportError:
            print("未安装 transformers / tokenizers —— 请先执行 `uv sync` 再重启内核。")"""),
    md("""### 2.6 数据流水线：`Dataset` 与 `DataLoader`

`Dataset` 负责「第 i 个样本长什么样」，`DataLoader` 负责「怎么凑成一个 batch」。"""),
    code("""import torch
from torch.utils.data import Dataset, DataLoader

class ToyDataset(Dataset):
    \"\"\"把一段文本切成定长片段，做下一词预测\"\"\"
    def __init__(self, ids, max_length=8):
        self.ids = ids
        self.max_length = max_length

    def __len__(self):
        return len(self.ids) // self.max_length

    def __getitem__(self, index):
        chunk = self.ids[index * self.max_length:(index + 1) * self.max_length]
        input_ids = torch.tensor(chunk, dtype=torch.long)
        labels = input_ids.clone()          # 先复制，稍后错位
        return input_ids, labels

ids = list(range(1, 41))
ds = ToyDataset(ids, max_length=8)
loader = DataLoader(ds, batch_size=2, shuffle=False)
for x, y in loader:
    print("input:", x.tolist())
    print("label:", y.tolist())
    break"""),
    part2("本章的核心产出是两个东西：一个自己写的 BPE 合并循环，一个 `PretrainDataset`。\n\n> 💡 每个留白代码块后面都跟了一个**测试块**。先写代码，再运行测试块。"),
    testkit(),
    md("""### 2.7 从零实现 `train_bpe`

给定一个语料（`dict[str, int]`，词 → 频次）和目标合并次数 `num_merges`，
返回合并规则列表，并把语料逐步合并。

**算法**：
1. 把每个词拆成字符列表；
2. 统计所有相邻符号对的加权频次；
3. 取频次最高的对 (a, b)，合并成 a+b，记录规则；
4. 重复 num_merges 次。"""),
    blank("train_bpe", [
        "输入：corpus(dict 词→频次)、num_merges(int)",
        "输出：merges 列表 [(a, b), ...]",
        "每一步都要重新统计相邻对频次（因为合并后符号变了）",
        "频次相同时取字典序较小的一对，保证结果可复现",
    ], [
        "`list(word)` 把字符串拆成字符",
        "`collections.Counter`、`.most_common()`",
        "上面写过的 `merge_pair` 可以直接复用",
    ]),
    test("train_bpe", '''def _t():
    corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
    merges = train_bpe(corpus, 3)
    assert len(merges) == 3, f"应返回 3 条合并规则，得到 {len(merges)}"
    assert merges[0] == ("e", "s"), f"第一步应合并出现 9 次的 ('e', 's')，得到 {merges[0]}"
    assert merges[1] == ("es", "t"), f"第二步应合并出现 9 次的 ('es', 't')，得到 {merges[1]}"

check("train_bpe 合并顺序与频次统计", _t)'''),
    md("""### 2.8 实现 `PretrainDataset`

参考 MiniMind 的写法：读一批文本 → 分词 → 加 BOS/EOS → 补齐到 `max_length` →
**把 pad 位置的标签设为 `-100`**（这样交叉熵会忽略它们）。

**为什么是 -100**：`F.cross_entropy(..., ignore_index=-100)` 会跳过标签为 -100 的位置。"""),
    blank("PretrainDataset", [
        "继承 torch.utils.data.Dataset",
        "__init__ 接收 data_path / tokenizer / max_length（用 json 逐行读 jsonl 即可）",
        "__getitem__ 返回 (input_ids, labels)，两者都是 torch.long 张量",
        "序列结构：[BOS] + tokens + [EOS] + [PAD] * 补齐长度",
        "labels = input_ids.clone()，然后把 pad 位置改成 -100",
    ], [
        "`torch.tensor(x, dtype=torch.long)`",
        "`tokenizer(...).input_ids`",
        "`labels[input_ids == pad_id] = -100`",
        "`json.loads(line)` 逐行读取",
    ]),
    test("PretrainDataset", '''def _t():
    import json, tempfile, os

    class FakeTok:
        bos_token_id, eos_token_id, pad_token_id = 1, 2, 0
        def __call__(self, text, add_special_tokens=False, max_length=100, truncation=True):
            ids = [3 + (ord(ch) % 50) for ch in str(text)][:max_length]
            return type("Enc", (), {"input_ids": ids})()

    path = os.path.join(tempfile.gettempdir(), "mm_pretrain_ds_test.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": "hello world"}) + "\\n")

    ds = PretrainDataset(path, FakeTok(), max_length=16)
    x, y = ds[0]
    assert tuple(x.shape) == (16,), f"input_ids 形状应为 (16,)，得到 {tuple(x.shape)}"
    assert tuple(y.shape) == (16,), f"labels 形状应为 (16,)，得到 {tuple(y.shape)}"
    assert x[0].item() == 1, "序列应以 BOS(1) 开头"
    assert x[1].item() != 0, "第 2 个位置应是真实 token，不该是 pad"
    assert (y[x == 0] == -100).all(), "pad 位置的标签应全部为 -100"
    assert (y[x != 0] == x[x != 0]).all(), "非 pad 位置的标签应等于 input_ids 本身"

check("PretrainDataset 形状 / BOS / pad 标签", _t)'''),
    summary_cell(),
    part3([
        "能用自己的话说清 BPE 的合并规则",
        "`train_bpe` 在小语料上跑出合理结果了吗？",
        "`PretrainDataset` 返回的两个张量形状一致吗？pad 位置标签是 -100 吗？",
        "用 `DataLoader` 取一个 batch，形状是 `[batch, max_length]` 吗？",
    ], "**思考题**：如果词表太小，会发生什么？太大呢？MiniMind 为什么选 6400 这个数量级？"),
]

# ══════════════════════════════════════════════════════════════
# 03 RMSNorm
# ══════════════════════════════════════════════════════════════
c3 = [
    header(3, "RMSNorm 均方根归一化", "第二部分 · 模型", "03_rmsnorm.html",
           "从这一章开始进入模型。第一个零件是归一化——它保证深层网络里激活值不会越跑越偏。"),
    goals([
        "理解 LayerNorm 与 RMSNorm 的区别",
        "掌握 `torch.rsqrt`、`keepdim=True` 的广播规则",
        "理解为什么需要 `keepdim`，写错会怎样",
        "能从零实现 RMSNorm 并与 `nn.LayerNorm` 对照",
    ]),
    img(
        ("../images/LLM-structure.jpg", "RMSNorm 在整体架构中的位置：每个 Block 的注意力子层与前馈子层之前各有一个"),
    ),
    md("""## 📖 背景：为什么需要归一化

深度网络里，每一层的输出都会被下一层放大或缩小。层数一多，数值就会指数级漂移——
要么爆炸成 `inf`，要么消失成 0。归一化把每层的数值**拉回同一量级**，训练才稳得住。

| | LayerNorm | RMSNorm |
|---|---|---|
| 减均值 | ✅ | ❌ |
| 除标准差 | ✅ | 除**均方根** |
| 偏置 β | ✅ | ❌ |
| 速度 | 慢 | 快 |

RMSNorm 的公式只有一行：

$$\\text{RMS}(x)=\\sqrt{\\frac{1}{d}\\sum_i x_i^2+\\epsilon}, \\qquad \\text{RMSNorm}(x)=\\gamma \\odot \\frac{x}{\\text{RMS}(x)}$$

---

"""),
    part1(),
    md("### 3.1 幂与开方：`Tensor.pow`、`torch.sqrt`、`torch.rsqrt`"),
    code("""import torch

x = torch.tensor([4.0, 9.0, 16.0])
print("x^2        :", x.pow(2))
print("sqrt(x)    :", torch.sqrt(x))
print("rsqrt(x)   :", torch.rsqrt(x))          # 1/sqrt(x)，比 1/torch.sqrt 更快更稳
print("验证        :", torch.allclose(torch.rsqrt(x), 1 / torch.sqrt(x)))"""),
    md("""### 3.2 求均值与 `keepdim=True`

**这是本章最容易错的地方。** 看下面的形状变化。"""),
    code("""x = torch.randn(2, 5)          # 2 个样本，每个 5 维

m1 = x.mean(dim=-1)            # 形状 [2]
m2 = x.mean(dim=-1, keepdim=True)  # 形状 [2, 1]
print("x       :", x.shape)
print("mean    :", m1.shape, " keepdim:", m2.shape)

# 用 [2] 去减 [2,5] 会怎样？
try:
    print((x - m1).shape)
except RuntimeError as e:
    print("广播失败：", str(e).split("at")[0].strip())

print("keepdim 版本可以正确广播:", (x - m2).shape)"""),
    md("""### 3.3 可学习参数：`nn.Parameter`

`nn.Parameter` 是会被 `optimizer` 更新的张量。RMSNorm 的 `γ` 就是它。"""),
    code("""from torch import nn

class Tiny(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))   # 初始化为全 1

t = Tiny(4)
print("weight        :", t.weight)
print("需要梯度吗     :", t.weight.requires_grad)
print("会被列出为参数 :", [n for n, _ in t.named_parameters()])"""),
    md("""### 3.4 手动算一遍 RMSNorm

先不写类，用张量运算手推一遍，看清每一步的形状。"""),
    code("""x = torch.tensor([[2.0, -1.0, 3.0, -0.5]])
gamma = torch.tensor([1.2, 0.8, 1.0, 1.1])
eps = 1e-6

ms = x.pow(2).mean(dim=-1, keepdim=True)     # 均方
rms = torch.rsqrt(ms + eps)                  # 1/均方根
xn = x * rms                                 # 归一化
y = gamma * xn                               # 缩放

print("均方   :", ms.squeeze().item())
print("RMS    :", (1 / rms).squeeze().item())
print("归一化 :", xn.tolist())
print("输出   :", y.tolist())
print("归一化后的 RMS 应约等于 1:", xn.pow(2).mean().sqrt().item())"""),
    part2("模型部分的实现，只给提示。请自己写出来。\n\n> 💡 每个留白代码块后面都跟了一个**测试块**。先写代码，再运行测试块。"),
    testkit(),
    md("""### 3.5 实现 `RMSNorm` 类

**要求**：
- `__init__(self, dim, eps=1e-6)`：保存 `eps`，创建形状为 `dim` 的 `nn.Parameter`（初始全 1）
- `norm(x)`：返回 `x * rsqrt(mean(x²) + eps)`
- `forward(x)`：先算归一化，再乘 `weight`，**并保证输出 dtype 与输入一致**

**小坑**：MiniMind 里先 `.float()` 算归一化再 `.type_as(x)` 转回去——
因为 fp16 下平方容易溢出。"""),
    blank("RMSNorm", [
        "继承 nn.Module",
        "weight = nn.Parameter(torch.ones(dim))",
        "norm(x) = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)",
        "forward(x) = (weight * norm(x.float())).type_as(x)",
    ], [
        "`torch.rsqrt`",
        "`Tensor.mean(dim=-1, keepdim=True)`",
        "`Tensor.type_as(other)`",
        "`Tensor.float()`",
    ]),
    test("RMSNorm", '''def _t():
    torch.manual_seed(0)
    m = RMSNorm(8)
    x = torch.randn(2, 5, 8)
    y = m(x)
    assert tuple(y.shape) == (2, 5, 8), f"输出形状应与输入一致，得到 {tuple(y.shape)}"
    rms = y.pow(2).mean(-1).sqrt()
    close(rms.mean().item(), 1.0, tol=1e-2, msg="weight 初始为 1 时，归一化后每个向量的 RMS 应约等于 1")
    close(m(x * 7.3), y, tol=1e-3, msg="RMSNorm 应具有尺度不变性：RMSNorm(k*x) == RMSNorm(x)")
    assert torch.isfinite(m(x * 1e4)).all(), "输入放大 1e4 倍后出现了 inf/nan，检查是否先用 float 计算"

check("RMSNorm 形状 / 数值 / 尺度不变性", _t)'''),
    md("""### 3.6 验证：与 `nn.LayerNorm` 对比 + 尺度不变性

写完之后跑这两组验证：
1. 当 `weight` 全 1 且输入均值为 0 时，RMSNorm 与 LayerNorm 输出应该接近；
2. `RMSNorm(k * x) == RMSNorm(x)`（尺度不变性）。"""),
    blank("validate_rmsnorm", [
        "构造一个随机输入 x 和你的 RMSNorm 实例",
        "测试 1：比较 RMSNorm(x) 与 nn.LayerNorm(dim, elementwise_affine=False)(x) 的差异",
        "测试 2：验证 RMSNorm(x) 与 RMSNorm(10 * x) 是否几乎相同",
        "打印两种情况的误差",
    ], [
        "`torch.allclose(a, b, atol=1e-3)`",
        "`nn.LayerNorm(dim, elementwise_affine=False)`",
    ]),
    summary_cell(),
    part3([
        "能解释为什么 `keepdim=True` 不能省",
        "能说清 RMSNorm 比 LayerNorm 少了哪几步计算",
        "尺度不变性验证通过了吗？",
        "你的实现里 fp16 输入会溢出吗？加了 `.float()` 吗？",
    ], "**思考题**：如果 `weight` 初始化成全 0 而不是全 1，训练一开始会发生什么？"),
]

# ══════════════════════════════════════════════════════════════
# 04 RoPE
# ══════════════════════════════════════════════════════════════
c4 = [
    header(4, "RoPE 旋转位置编码", "第二部分 · 模型", "04_rope.html",
           "注意力本身不知道词的顺序。RoPE 用「旋转」这个几何操作，把位置信息优雅地注入到 Q 和 K 里。"),
    goals([
        "理解 RoPE 的几何直觉：位置 = 旋转角，内积只依赖相对距离",
        "掌握 `torch.outer`、`torch.cos/sin`、切片翻转 `rotate_half`",
        "掌握 `unsqueeze` 的广播维度对齐技巧",
        "能从零实现 `precompute_freqs_cis` 与 `apply_rotary_pos_emb`",
    ]),
    md("""## 📖 背景：位置信息怎么塞进注意力

把向量的每两个维度看成一个复数平面上的点。位置 `m` 的 token，第 `i` 组维度旋转 `m·θᵢ`：

$$\\theta_i = \\frac{1}{10000^{2i/d}}, \\qquad
\\begin{bmatrix} x' \\\\ y' \\end{bmatrix} =
\\begin{bmatrix} \\cos m\\theta_i & -\\sin m\\theta_i \\\\ \\sin m\\theta_i & \\cos m\\theta_i \\end{bmatrix}
\\begin{bmatrix} x \\\\ y \\end{bmatrix}$$

因为旋转是等距变换，两个向量都旋转后，它们的内积只依赖角度差 `(m−n)·θ`——**也就是相对位置**。

---

"""),
    img(
        ("../images/rope_ppl.png", "不同 RoPE 配置下的困惑度对比——YaRN 外推能让模型在更长上下文上保持稳定"),
    ),
    part1(),
    md("### 4.1 `torch.arange` 与 `torch.outer`\n\n`outer(t, f)` 会算出所有位置 × 所有频率的组合矩阵。"),
    code("""import torch, math

t = torch.arange(6).float()              # 位置 0..5
f = 1.0 / (10000 ** (torch.arange(0, 8, 2).float() / 8))   # 4 个频率
print("位置 t :", t)
print("频率 f :", f)

freqs = torch.outer(t, f)                # [6, 4]
print("freqs 形状:", freqs.shape)
print(freqs[:2])"""),
    md("""### 4.2 `torch.cos` / `torch.sin` 与 `torch.cat`

RoPE 把 cos/sin 各复制一份再拼接，好让它们能直接和 head_dim 维度对齐。"""),
    code("""freqs = torch.randn(6, 4)
cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
print("cos 形状:", cos.shape, " (4 -> 8，因为拼了两份)")"""),
    md("""### 4.3 `rotate_half`：切片翻转

RoPE 的旋转可以写成 `x·cos + rotate_half(x)·sin`。`rotate_half` 把后半段取负放到前面。"""),
    code("""def rotate_half(x):
    # x 最后一维是 d，切成两半
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)

x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
print("原始      :", x.tolist())
print("rotate_half:", rotate_half(x).tolist(), "  <- 期望 [-3, -4, 1, 2]")"""),
    md("""### 4.4 `unsqueeze` 广播对齐

`cos` 的形状是 `[seq, head_dim]`，而 `q` 是 `[batch, seq, heads, head_dim]`。
要在 `seq` 与 `heads` 之间插一个维度才能广播。"""),
    code("""q = torch.randn(2, 6, 8, 16)      # [B, S, H, D]
cos = torch.randn(6, 16)          # [S, D]

# 直接在 dim=1 插入 -> [S, 1, D]，可以和 [B, S, H, D] 广播
print("unsqueeze(1):", cos.unsqueeze(1).shape)

# 验证：乘法能跑通
out = q * cos.unsqueeze(1)
print("广播相乘结果:", out.shape)"""),
    part2("RoPE 有两个函数要写：预计算频率表、把旋转应用到 Q/K。\n\n> 💡 每个留白代码块后面都跟了一个**测试块**。先写代码，再运行测试块。"),
    testkit(),
    md("""### 4.5 实现 `precompute_freqs_cis`

**要求**：
1. 生成频率 `freqs = 1 / (theta ** (arange(0, dim, 2) / dim))`
2. `freqs = outer(arange(end), freqs)` 得到 `[end, dim/2]`
3. 算 `cos` 和 `sin`，各复制一份 `cat` 成 `dim` 维
4. 返回 `(freqs_cos, freqs_sin)`

（进阶：可选实现 YaRN 的 `rope_scaling` 分支，网页里有公式。）"""),
    blank("precompute_freqs_cis", [
        "签名：precompute_freqs_cis(dim, end=32768, rope_base=1e6)",
        "freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:dim//2].float() / dim))",
        "freqs = torch.outer(torch.arange(end).float(), freqs)",
        "freqs_cos = torch.cat([cos, cos], dim=-1)；sin 同理",
        "返回 (freqs_cos, freqs_sin)，形状都是 [end, dim]",
    ], [
        "`torch.arange`、`torch.outer`",
        "`torch.cos`、`torch.sin`、`torch.cat`",
    ]),
    test("precompute_freqs_cis", '''def _t():
    cos, sin = precompute_freqs_cis(dim=8, end=16, rope_base=1e6)
    assert tuple(cos.shape) == (16, 8), f"cos 形状应为 (16, 8)，得到 {tuple(cos.shape)}"
    assert tuple(sin.shape) == (16, 8), f"sin 形状应为 (16, 8)，得到 {tuple(sin.shape)}"
    close(cos[0], torch.ones(8), tol=1e-5, msg="位置 0 的旋转角为 0，cos 应全为 1")
    close(sin[0], torch.zeros(8), tol=1e-5, msg="位置 0 的旋转角为 0，sin 应全为 0")
    close(cos[:, :4], cos[:, 4:], tol=1e-6, msg="cos 前后两半应相同（cat 了两份）")
    close(sin[:, :4], sin[:, 4:], tol=1e-6, msg="sin 前后两半应相同（cat 了两份）")
    close(cos ** 2 + sin ** 2, torch.ones(16, 8), tol=1e-4, msg="cos² + sin² 应恒为 1")

check("precompute_freqs_cis 形状与边界值", _t)'''),
    md("""### 4.6 实现 `apply_rotary_pos_emb`

**要求**：给定 `q, k, cos, sin`，返回旋转后的 `q_embed, k_embed`。
公式：`x_embed = x * cos + rotate_half(x) * sin`。

**注意**：`cos`/`sin` 需要 `unsqueeze` 到合适的维度才能和 q/k 广播。"""),
    blank("apply_rotary_pos_emb", [
        "内部先定义 rotate_half(x)",
        "q_embed = q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1)",
        "k_embed 同理",
        "最后 .to(q.dtype) 保证类型一致",
    ], [
        "上面写过的 `rotate_half`",
        "`Tensor.unsqueeze`",
    ]),
    test("apply_rotary_pos_emb", '''def _t():
    torch.manual_seed(0)
    cos, sin = precompute_freqs_cis(dim=8, end=8, rope_base=1e6)
    q = torch.randn(1, 8, 2, 8)
    k = torch.randn(1, 8, 2, 8)
    qe, ke = apply_rotary_pos_emb(q, k, cos, sin)
    assert tuple(qe.shape) == tuple(q.shape), f"q 形状应保持不变，得到 {tuple(qe.shape)}"
    assert tuple(ke.shape) == tuple(k.shape), f"k 形状应保持不变，得到 {tuple(ke.shape)}"
    close(qe[:, 0], q[:, 0], tol=1e-5, msg="位置 0 旋转角为 0，输出应等于输入")
    close(qe.norm(dim=-1), q.norm(dim=-1), tol=1e-4, msg="旋转是等距变换，向量范数应保持不变")
    close(ke.norm(dim=-1), k.norm(dim=-1), tol=1e-4)

check("apply_rotary_pos_emb 形状 / 位置0不变 / 保范数", _t)'''),
    md("""### 4.7 验证相对位置不变性

这是 RoPE 最核心的性质，务必自己验证一次：
构造两个向量 q、k，分别旋转 m 和 n 位，内积应该只依赖 `(m−n)`。"""),
    blank("verify_rope", [
        "固定 q、k 两个向量（head_dim=8）",
        "取 (m, n) = (0, 0)、(1, 1)、(2, 2) → 内积应该相同",
        "取 (m, n) = (0, 3)、(2, 5)、(10, 13) → 内积应该相同（相对距离都是 3）",
        "打印各组的点积，确认同一相对距离结果一致",
    ], [
        "用你的 `precompute_freqs_cis` 取出对应位置的 cos/sin 切片",
        "`torch.dot` 或 `(a * b).sum()`",
    ]),
    summary_cell(),
    part3([
        "能画出「位置 → 旋转角」的几何图景",
        "`rotate_half` 的输出顺序对吗？（`[-后半, 前半]`）",
        "`unsqueeze(1)` 的作用说清楚了吗？",
        "相对位置不变性验证通过了吗？",
    ], "**思考题**：为什么不同维度要用不同的 θ，而不是所有维度共用一个？"),
]

# ══════════════════════════════════════════════════════════════
# 05 Attention
# ══════════════════════════════════════════════════════════════
c5 = [
    header(5, "因果自注意力与 GQA / KV Cache", "第二部分 · 模型", "05_attention.html",
           "注意力是 Transformer 的心脏。这一章要写的是全项目最核心的一段代码。"),
    goals([
        "理解缩放点积注意力的四步流程",
        "掌握 `matmul` 批量矩阵乘、`masked_fill`、`softmax`",
        "理解因果掩码为什么用上三角",
        "理解 GQA 的 `repeat_kv` 广播技巧，以及 KV Cache 的作用",
    ]),
    img(
        ("../images/LLM-structure.jpg", "注意力在整体架构中的位置：每个 Block 的核心就是它（8 个 Q 头 / 4 个 KV 头）"),
    ),
    md("""## 📖 背景：四步走完注意力

```python
scores = (Q @ K.T) / sqrt(head_dim)   # 1. 打分
scores = scores + causal_mask          # 2. 遮住未来
attn   = softmax(scores, dim=-1)       # 3. 归一化
out    = attn @ V                      # 4. 加权求和
```

**因果**：位置 t 只能看到 ≤ t 的信息，所以要把注意力矩阵的**上三角**设成 −inf。
**GQA**：8 个 Q 头共享 4 组 KV，KV 显存减半。
**KV Cache**：生成第 t 个词时复用前 t−1 个词的 K/V，每步只算 1 个新 token。

---

"""),
    part1(),
    md("### 5.1 批量矩阵乘：`@` / `torch.matmul`\n\n注意力里所有的乘法都是批量矩阵乘，形状要对齐。"),
    code("""import torch

Q = torch.randn(2, 8, 6, 16)    # [B, H, S, D]
K = torch.randn(2, 8, 6, 16)

# 只要最后两维做矩阵乘，前面维度自动广播
scores = Q @ K.transpose(-2, -1)
print("scores 形状:", scores.shape, " <- [B, H, S, S]")"""),
    md("""### 5.2 因果掩码：`masked_fill` + `triu`

`triu(diagonal=1)` 取上三角（不含对角线），正是「不能看未来」要遮住的部分。"""),
    code("""S = 5
mask = torch.triu(torch.ones(S, S), diagonal=1).bool()
print("上三角布尔掩码:")
print(mask.int())

scores = torch.randn(1, 1, S, S)
scores = scores.masked_fill(mask, float("-inf"))
print("\\n遮罩后的最后一行:", scores[0, 0, -1].tolist(), " <- 只有最后一个是有限值")"""),
    md("""### 5.3 `softmax` 与数值稳定性

`−inf` 经过 `exp` 会变成 0，正好实现「权重为 0」。"""),
    code("""x = torch.tensor([[1.0, 2.0, float("-inf"), 3.0]])
print("softmax:", torch.softmax(x, dim=-1).tolist(), " <- -inf 位置变成 0")"""),
    md("""### 5.4 `repeat_kv`：GQA 的关键技巧

`[B, S, kv_heads, D]` 要变成 `[B, S, q_heads, D]`，让每 `n_rep` 个 Q 头共享一个 KV 头。
用 `expand` + `reshape` 完成，**不复制数据**。"""),
    code("""x = torch.randn(1, 6, 2, 4)          # kv_heads = 2
n_rep = 2                            # 每个 kv 头被 2 个 q 头共享

bs, slen, kvh, d = x.shape
y = x[:, :, :, None, :].expand(bs, slen, kvh, n_rep, d).reshape(bs, slen, kvh * n_rep, d)
print("原始:", x.shape, " -> 重复后:", y.shape)
print("验证前两组是否相同:", torch.equal(y[:, :, 0], y[:, :, 1]))"""),
    md("""### 5.5 内置加速：`F.scaled_dot_product_attention`

PyTorch 2.x 提供了融合实现的 SDPA，内部会自动选 FlashAttention / 内存高效核。"""),
    code("""import torch.nn.functional as F

q = torch.randn(1, 4, 6, 16)
k = torch.randn(1, 4, 6, 16)
v = torch.randn(1, 4, 6, 16)

out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print("SDPA 输出形状:", out.shape)"""),
    part2("本章要写的是全项目最核心的一段代码。请务必自己实现一遍。\n\n> 💡 每个留白代码块后面都跟了一个**测试块**。先写代码，再运行测试块。"),
    testkit(),
    md("""### 5.6 实现 `repeat_kv`

先写这个小工具——GQA 要靠它把 KV 头「复制」到和 Q 头一样多。

输入 `[B, S, kv_heads, D]` 和 `n_rep`，输出 `[B, S, kv_heads*n_rep, D]`。"""),
    blank("repeat_kv", [
        "如果 n_rep == 1，直接返回原张量",
        "否则用 x[:, :, :, None, :].expand(...) 再 reshape",
        "保证结果中每 n_rep 个相邻头的数据相同",
    ], ["`Tensor.expand`、`Tensor.reshape`"]),
    test("repeat_kv", '''def _t():
    x = torch.randn(2, 6, 2, 4)
    y = repeat_kv(x, 2)
    assert tuple(y.shape) == (2, 6, 4, 4), f"形状应为 (2, 6, 4, 4)，得到 {tuple(y.shape)}"
    same(y[:, :, 0], y[:, :, 1], msg="第 0、1 个头应完全相同")
    same(y[:, :, 2], y[:, :, 3], msg="第 2、3 个头应完全相同")
    assert tuple(repeat_kv(x, 1).shape) == tuple(x.shape), "n_rep=1 时形状不应改变"

check("repeat_kv 形状与分组", _t)'''),
    md("""### 5.7 实现 `causal_attention`

**要求**：实现一个函数，输入已经投影好的 Q / K / V，返回注意力输出。
**注意张量布局和 `repeat_kv` 保持一致**（头在第 2 维，不在第 1 维）：

```python
def causal_attention(q, k, v, n_rep=1, is_causal=True):
    # q:  [B, S, H,   D]
    # k:  [B, S, Hkv, D]      H == Hkv * n_rep
    # v:  [B, S, Hkv, D]
    # 返回 [B, S, H, D]
```

步骤：
1. 先用 `repeat_kv` 把 k/v 的头数扩到 H
2. `transpose(1, 2)` 变成 `[B, H, S, D]`，再算 `scores = q @ kᵀ / sqrt(D)`
3. 因果掩码 → `softmax` → `@ v`
4. 再 `transpose(1, 2)` 变回 `[B, S, H, D]`"""),
    blank("causal_attention", [
        "签名：causal_attention(q, k, v, n_rep=1, is_causal=True)",
        "输入 q: [B, S, H, D]，k/v: [B, S, Hkv, D]；输出 [B, S, H, D]",
        "先用 repeat_kv 把 k/v 的头数从 Hkv 扩到 H",
        "transpose(1, 2) 得到 [B, H, S, D]，scores = q @ k.transpose(-2, -1) / sqrt(D)",
        "is_causal=True 时把上三角（对角线以上）设成 -inf",
        "softmax 后与 v 相乘，再 transpose(1, 2) 还原布局",
    ], [
        "`Tensor.transpose`、`@` 或 `torch.matmul`",
        "`torch.triu`、`Tensor.masked_fill`",
        "`torch.softmax`、`math.sqrt`",
        "上面写过的 `repeat_kv`",
    ]),
    test("causal_attention", '''def _t():
    torch.manual_seed(0)
    B, S, H, D = 1, 5, 2, 4
    q = torch.randn(B, S, H, D); k = torch.randn(B, S, H, D); v = torch.randn(B, S, H, D)
    out = causal_attention(q, k, v)
    assert tuple(out.shape) == (B, S, H, D), f"输出形状应为 {(B, S, H, D)}，得到 {tuple(out.shape)}"
    q2 = q.clone(); q2[:, -1] += 5.0
    close(causal_attention(q2, k, v)[:, :-1], out[:, :-1], tol=1e-5,
          msg="只改了最后一个位置的 q，却影响了前面的输出 —— 因果掩码不对")
    ref = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                         is_causal=True).transpose(1, 2)
    close(out, ref, tol=1e-4, msg="与 F.scaled_dot_product_attention(is_causal=True) 不一致")
    q3 = torch.randn(1, 5, 4, 4); k3 = torch.randn(1, 5, 2, 4); v3 = torch.randn(1, 5, 2, 4)
    assert tuple(causal_attention(q3, k3, v3, n_rep=2).shape) == (1, 5, 4, 4), "n_rep=2（GQA）时形状不对"

check("causal_attention 形状 / 因果性 / 与 SDPA 一致", _t)'''),
    md("""### 5.8 验证因果性

写一个测试：把序列**最后一个位置**的输入改掉，前面位置的输出应该**完全不变**（因为看不到未来）。"""),
    blank("verify_causal", [
        "构造输入 x: [1, 6, 32]",
        "跑一遍得到 out1",
        "把 x[:, -1, :] 随机改掉，再跑一遍得到 out2",
        "断言 out1[:, :-1, :] 与 out2[:, :-1, :] 完全一致",
    ], ["`torch.allclose`"]),
    summary_cell(),
    part3([
        "能说清 scores 的形状是 `[B, H, S, S]`",
        "因果掩码为什么用 `triu(diagonal=1)`？",
        "`repeat_kv` 有没有真的复制数据？（expand 是视图）",
        "因果性验证通过了吗？",
    ], "**思考题**：KV Cache 让推理复杂度从 O(t²) 降到 O(t)，那训练时为什么不用它？"),
]

for name, cells in [
    ("01_overview.ipynb", c),
    ("02_tokenizer.ipynb", c2),
    ("03_rmsnorm.ipynb", c3),
    ("04_rope.ipynb", c4),
    ("05_attention.ipynb", c5),
]:
    save(name, cells)
