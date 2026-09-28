"""生成第 06–10 章的 Notebook。"""
from nbcommon import (md, code, blank, img, header, goals, part1, part2, part3, save,
                        testkit, test, summary_cell, deps)

# ══════════════════════════════════════════════════════════════
# 06 SwiGLU
# ══════════════════════════════════════════════════════════════
c = [
    header(6, "SwiGLU 前馈网络", "第二部分 · 模型", "06_swiglu.html",
           "注意力负责 token 之间的交流，前馈网络负责单个 token 内部的加工。SwiGLU 用一个「门」决定哪些信息通过。"),
    goals([
        "理解 GLU（门控线性单元）的结构",
        "掌握 `nn.Linear(bias=False)`、`nn.SiLU`、逐元素乘",
        "会从 `transformers` 的 `ACT2FN` 查激活函数",
        "能从零实现 `FeedForward`（SwiGLU）",
    ]),
    img(
        ("../images/LLM-structure.jpg", "前馈网络在整体架构中的位置：每个 Block 里注意力之后就是它"),
    ),
    md("""## 📖 背景：门控前馈

传统 FFN 是 `d → 4d → d`（两个矩阵）。SwiGLU 变成三个矩阵，中间多了一条「门控」分支：

```python
gate = SiLU(W_gate @ x)     # 门：决定放行多少
up   =        W_up   @ x     # 内容：决定传什么
h    = gate * up             # 逐元素相乘 = 门控
out  = W_down @ h            # 降回 hidden_size
```

因为多了一个矩阵，中间维度要缩小到约 `8/3·d` 才能保持参数量相当。
MiniMind 用 `ceil(d·π/64)·64`，768 维时得到 2432（对齐 64 的倍数，矩阵乘更快）。

---

"""),
    part1(),
    md("### 6.1 `nn.Linear` 与 `bias=False`\n\n现代 LLM 的线性层几乎都不带偏置——省参数，且归一化已经处理了平移。"),
    code("""import torch
from torch import nn

lin = nn.Linear(4, 8, bias=False)
print("weight 形状:", lin.weight.shape)      # [out, in]
print("有 bias 吗  :", lin.bias)

x = torch.randn(2, 4)
print("输出形状    :", lin(x).shape)"""),
    md("""### 6.2 SiLU 与 `nn.SiLU`

`SiLU(x) = x · sigmoid(x)`，比 ReLU 平滑，负半轴不会硬性归零。"""),
    code("""import torch.nn.functional as F

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
silu_manual = x * torch.sigmoid(x)
silu_nn = nn.SiLU()(x)

print("手动 SiLU:", silu_manual.tolist())
print("nn.SiLU  :", silu_nn.tolist())
print("一致吗    :", torch.allclose(silu_manual, silu_nn))
print("对比 ReLU :", F.relu(x).tolist(), " <- 负半轴直接变 0")"""),
    md("""### 6.3 用 `ACT2FN` 查表

`transformers` 提供了「激活函数名 → 函数」的映射表，方便从配置字符串拿到激活函数。"""),
    code("""try:
    from transformers.activations import ACT2FN
    print("可用激活函数:", list(ACT2FN.keys()))
    print("silu ->", ACT2FN["silu"])
except ImportError:
    print("未安装 transformers，可用 uv add transformers 安装")"""),
    md("""### 6.4 逐元素乘：门控的本质

`gate * up` 是逐元素相乘，不是矩阵乘。这是「门」能选择性放行的关键。"""),
    code("""gate = torch.tensor([[0.9, 0.1, 0.05]])
up   = torch.tensor([[2.0, 2.0, 2.0]])
print("gate   :", gate.tolist())
print("up     :", up.tolist())
print("门控结果:", (gate * up).tolist(), " <- 门值小的地方被压下去了")"""),
    part2("模型部分只给提示。请自己写出 SwiGLU 前馈层。\n\n> 💡 每个留白代码块后面都跟了一个**测试块**。先写代码，再运行测试块。"),
    testkit(),
    md("""### 6.5 实现 `FeedForward`

**要求**：
- `__init__(self, config, intermediate_size=None)`：创建 `gate_proj`、`up_proj`（`hidden → intermediate`）
  和 `down_proj`（`intermediate → hidden`），都不带 bias
- 从 `config.hidden_act` 用 `ACT2FN` 取激活函数
- `forward(x)`：返回 `down_proj(act(gate_proj(x)) * up_proj(x))`

**小坑**：`gate_proj` 和 `up_proj` 的输出维度必须一致才能相乘。"""),
    blank("FeedForward", [
        "继承 nn.Module",
        "intermediate_size 默认取 config.intermediate_size",
        "三个 nn.Linear(..., bias=False)：gate_proj / up_proj / down_proj",
        "self.act_fn = ACT2FN[config.hidden_act]",
        "forward: down_proj(act_fn(gate_proj(x)) * up_proj(x))",
    ], [
        "`nn.Linear(in_features, out_features, bias=False)`",
        "`ACT2FN`",
    ]),
    test("FeedForward", '''def _t():
    torch.manual_seed(0)

    class Cfg:
        hidden_size = 16
        intermediate_size = 32
        hidden_act = "silu"

    ffn = FeedForward(Cfg())
    x = torch.randn(2, 7, 16)
    y = ffn(x)
    assert tuple(y.shape) == (2, 7, 16), f"输出形状应为 (2, 7, 16)，得到 {tuple(y.shape)}"
    n = sum(p.numel() for p in ffn.parameters())
    assert n == 3 * 16 * 32, f"参数量应为 3×16×32 = {3 * 16 * 32}，得到 {n}"
    ref = ffn.down_proj(F.silu(ffn.gate_proj(x)) * ffn.up_proj(x))
    close(y, ref, tol=1e-5, msg="与 down_proj(silu(gate_proj(x)) * up_proj(x)) 不一致")

check("FeedForward 形状 / 参数量 / 与公式一致", _t)'''),
    md("""### 6.6 验证形状与参数量

写完之后：
1. 输入 `[B, S, hidden]`，输出形状应该不变；
2. 打印参数量，和 `2 × d × intermediate + d × intermediate` 对一下。"""),
    blank("verify_ffn", [
        "构造一个 config 对象（含 hidden_size / intermediate_size / hidden_act）",
        "实例化你的 FeedForward",
        "输入 torch.randn(2, 10, hidden_size)，确认输出形状相同",
        "统计参数量并与理论值比较",
    ], ["`sum(p.numel() for p in ffn.parameters())`"]),
    summary_cell(),
    part3([
        "能解释 gate / up / down 三个矩阵各自的作用",
        "知道为什么中间维度是 8/3·d 而不是 4d",
        "能说清 SiLU 比 ReLU 好在哪里",
        "参数量与理论值对上了吗？",
    ], "**思考题**：如果把 `gate_proj` 和 `up_proj` 合并成一个矩阵（一次投影到 2×intermediate 再切开），计算效率会变吗？"),
]

# ══════════════════════════════════════════════════════════════
# 07 MoE
# ══════════════════════════════════════════════════════════════
c2 = [
    header(7, "MoE 混合专家", "第二部分 · 模型", "07_moe.html",
           "MoE 把前馈层拆成多个「专家」，每个 token 只激活其中一两个——参数量上去了，计算量却几乎不变。"),
    goals([
        "理解稀疏激活：用参数换算力",
        "掌握 `torch.topk`、`F.one_hot`、`index_add_`",
        "理解负载均衡辅助损失的作用",
        "能从零实现 `MOEFeedForward`",
    ]),
    md("""## 📖 背景：稀疏激活

```python
scores   = softmax(router(x))          # 每个 token 对 N 个专家打分
topk_w, topk_i = topk(scores, k=1)     # 只选分数最高的 k 个
y = Σ weight_i * expert_i(x)           # 只调用被选中的专家
```

稠密模型每个 token 用全部参数；MoE 每个 token 只用一小部分。
所以 MoE 的**总参数量**可以是稠密的 3 倍，但**单 token 激活参数**几乎一样。

问题也随之而来：如果所有 token 都涌向同一个专家，其它专家就「饿死」了。
所以需要**负载均衡辅助损失**。

---

"""),
    img(
        ("../images/LLM-structure-moe.jpg", "MiniMind MoE 结构：前馈层被替换为多个专家 + 一个路由器（Qwen3-MoE 风格，无共享专家）"),
    ),
    part1(),
    md("### 7.1 `torch.topk`：取分数最高的 k 个"),
    code("""import torch

scores = torch.tensor([[0.1, 0.6, 0.2, 0.1],
                       [0.4, 0.1, 0.4, 0.1]])
w, idx = torch.topk(scores, k=1, dim=-1, sorted=False)
print("top1 权重:", w.tolist())
print("top1 索引:", idx.tolist())

w2, idx2 = torch.topk(scores, k=2, dim=-1, sorted=True)
print("top2 索引:", idx2.tolist())"""),
    md("""### 7.2 `F.one_hot`：把索引变成独热

用来统计「每个专家被选了多少次」。"""),
    code("""import torch.nn.functional as F

idx = torch.tensor([[0], [2], [1], [0]])       # 4 个 token 选了哪个专家
oh = F.one_hot(idx, num_classes=4).float()     # [4, 1, 4]
print("one_hot 形状:", oh.shape)
load = oh.mean(0)                              # 每个专家被选中的比例
print("负载分布:", load.squeeze().tolist())"""),
    md("""### 7.3 `index_add_`：按需累加

MoE 的循环里，我们只对「选中了当前专家」的那些 token 做计算，再把结果累加回原位置。"""),
    code("""y = torch.zeros(4, 3)                    # 4 个 token，每个 3 维
token_idx = torch.tensor([0, 3])         # 这两个 token 属于本专家
vals = torch.ones(2, 3) * 5.0
y.index_add_(0, token_idx, vals)         # 按索引累加
print(y)"""),
    md("""### 7.4 布尔掩码分派

`topk_idx == i` 得到一个布尔张量，标出哪些 token 选了专家 i。"""),
    code("""topk_idx = torch.tensor([[0], [2], [0], [1]])
i = 0
mask = (topk_idx == i)
print("mask 形状:", mask.shape)
print("哪些 token 选了专家 0:", mask.any(dim=-1).nonzero().flatten().tolist())"""),
    *deps("""### 7.5 前置依赖：`FeedForward`

MoE 的「专家」就是一个前馈层，也就是第 6 章你写过的 `FeedForward`。
下面给一份**等价的参考实现**——不是让你抄，而是让本章可以独立测试。
（如果你第 6 章已经写好了，可以直接换成你自己的版本。）""",
    '''from torch import nn

class FeedForward(nn.Module):
    """参考实现，等价于第 6 章的 SwiGLU 前馈层"""
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        d = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, d, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, d, bias=False)
        self.down_proj = nn.Linear(d, config.hidden_size, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))'''),
    part2("MoE 的实现里有一个容易被忽略的细节：反向传播的梯度路径。请仔细看提示。\n\n> 💡 每个留白代码块后面都跟了一个**测试块**。先写代码，再运行测试块。"),
    testkit(),
    md("""### 7.6 实现 `MOEFeedForward`

**要求**：
1. `self.gate = nn.Linear(hidden_size, num_experts, bias=False)`（路由器）
2. `self.experts = nn.ModuleList([FeedForward(config) for _ in range(num_experts)])`
3. `forward(x)`：
   - 把 `[B, S, d]` 展平成 `[B*S, d]`
   - 算 router 分数 → softmax → `topk`
   - 对每个专家：找出选了它的 token，算输出并**按权重累加**回 `y`
   - reshape 回 `[B, S, d]`
4. 训练时计算 `aux_loss`（负载均衡），否则返回 0

**关键细节**：MiniMind 在 `num_experts_per_tok == 1` 时用「直通估计」——
前向让权重恒为 1.0，但用 `top1 - top1.detach() + 1.0` 保留梯度路径。
这样单专家路由时权重不会退化。"""),
    blank("MOEFeedForward", [
        "继承 nn.Module",
        "gate = nn.Linear(hidden_size, num_experts, bias=False)",
        "experts = nn.ModuleList([FeedForward(config, moe_intermediate_size) ...])",
        "forward: 展平 -> softmax(gate(x)) -> topk -> 逐专家 index_add_ 累加",
        "k==1 时权重用 top1 - top1.detach() + 1.0 保持梯度",
        "训练时计算 aux_loss = (load * prob).sum() * num_experts * coef",
    ], [
        "`torch.topk`、`F.softmax`",
        "`torch.zeros_like`、`Tensor.index_add_`",
        "`F.one_hot`、`Tensor.nonzero`",
    ]),
    test("MOEFeedForward", '''def _t():
    torch.manual_seed(0)

    class Cfg:
        hidden_size = 16
        intermediate_size = 32
        moe_intermediate_size = 32
        num_experts = 4
        num_experts_per_tok = 1
        norm_topk_prob = True
        router_aux_loss_coef = 5e-4

    moe = MOEFeedForward(Cfg())
    x = torch.randn(2, 6, 16)
    y = moe(x)
    assert tuple(y.shape) == (2, 6, 16), f"输出形状应为 (2, 6, 16)，得到 {tuple(y.shape)}"
    assert torch.isfinite(y).all(), "输出出现了 inf/nan"

    xf, yf = x.reshape(-1, 16), y.reshape(-1, 16)
    top1 = F.softmax(moe.gate(xf), dim=-1).argmax(-1)
    for i in range(xf.shape[0]):
        e = top1[i].item()
        close(yf[i], moe.experts[e](xf[i]), tol=1e-4,
              msg=f"token {i} 选了专家 {e}，输出应等于该专家的输出（k=1 时权重为 1）")

check("MOEFeedForward 形状 / 单专家路由", _t)'''),
    md("""### 7.7 实现辅助损失 `aux_loss`

**要求**：输入 router 分数和 topk 索引，返回负载均衡损失。

```python
load = one_hot(topk_idx, num_experts).float().mean(0)   # 实际负载
prob = scores.mean(0)                                    # 平均路由概率
aux  = (load * prob).sum() * num_experts * coef
```

**直觉**：当负载完全均匀时，`load ≈ prob ≈ 1/N`，`sum = N·(1/N)² = 1/N`，
乘 `N` 后得到 1。所以 aux_loss 越接近 1 表示越均衡。"""),
    blank("aux_loss", [
        "输入：scores [B*S, N]、topk_idx [B*S, k]、num_experts、coef",
        "load = F.one_hot(topk_idx, N).float().mean(0)",
        "prob = scores.mean(0)",
        "返回 (load * prob).sum() * num_experts * coef",
    ], ["`F.one_hot`", "`Tensor.mean`"]),
    test("aux_loss", '''def _t():
    balanced = torch.tensor([[0.7, 0.1, 0.1, 0.1],
                             [0.1, 0.7, 0.1, 0.1],
                             [0.1, 0.1, 0.7, 0.1],
                             [0.1, 0.1, 0.1, 0.7]])
    idx_b = balanced.argmax(-1, keepdim=True)
    loss_b = aux_loss(balanced, idx_b, num_experts=4, coef=1.0)
    close(loss_b.item(), 1.0, tol=1e-5, msg="负载与概率都均匀时，aux_loss 应为 1.0")

    skewed = torch.tensor([[0.9, 0.03, 0.03, 0.04]] * 4)
    idx_s = skewed.argmax(-1, keepdim=True)
    loss_s = aux_loss(skewed, idx_s, num_experts=4, coef=1.0)
    assert loss_s.item() > loss_b.item(), \\
        f"负载集中时 aux_loss 应更大（这正是它要惩罚的），得到 {loss_s.item():.3f}"

check("aux_loss 负载越均匀值越接近 1", _t)'''),
    summary_cell(),
    part3([
        "能解释 MoE 为什么「参数多但算力省」",
        "负载均衡损失接近 1 意味着什么？",
        "你的实现里被选中的 token 输出正确累加了吗？",
        "k=1 时的直通估计你理解了吗？",
    ], "**思考题**：如果路由器对所有 token 都给出相同分数，会发生什么？这时 aux_loss 是多少？"),
]

# ══════════════════════════════════════════════════════════════
# 08 Block & Model
# ══════════════════════════════════════════════════════════════
c3 = [
    header(8, "Transformer Block 与模型组装", "第二部分 · 模型", "08_block_model.html",
           "把 RMSNorm、注意力、SwiGLU 拼成一个 Block，堆 8 层，加上词嵌入和输出头——一个完整的 MiniMind 就成型了。"),
    goals([
        "理解 Pre-Norm 残差块的结构",
        "掌握 `nn.ModuleList`、`nn.Embedding`、`register_buffer`",
        "理解权重共享（weight tying）",
        "能从零组装完整模型并算交叉熵损失",
    ]),
    md("""## 📖 背景：一个 Block 的骨架

```python
x = x + Attn( RMSNorm(x) )      # 注意力子层 + 残差
x = x + FFN ( RMSNorm(x) )      # 前馈子层 + 残差
```

**Pre-Norm**：先归一化再进子层。比 Post-Norm 更容易训练深层网络。
**残差**：`x + f(x)`，让梯度有一条「高速公路」直达浅层。

完整模型：`Embedding → 8 × Block → RMSNorm → lm_head → logits`。

---

"""),
    img(
        ("../images/LLM-structure.jpg", "把前面各章的零件拼起来，就是这张图——Embedding、8 个 Block、RMSNorm、输出头"),
    ),
    part1(),
    md("### 8.1 `nn.Embedding`：查表"),
    code("""import torch
from torch import nn

emb = nn.Embedding(100, 8)          # 词表 100，每词 8 维
ids = torch.tensor([[1, 5, 9], [2, 3, 4]])
print("输入形状:", ids.shape, " 输出形状:", emb(ids).shape)
print("等价于直接索引:", torch.allclose(emb(ids), emb.weight[ids]))"""),
    md("""### 8.2 `nn.ModuleList`：动态数量的层\n\n堆 N 层用 `ModuleList`，它会被正确注册为子模块（普通 list 不会）。"""),
    code("""class Stack(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(n)])

s = Stack(3)
print("子模块数量:", len(s.layers))
print("参数会被 optimizer 看到:", len(list(s.parameters())) == 6)"""),
    md("""### 8.3 `register_buffer`：不参与训练的常量\n\nRoPE 的 cos/sin 表是常量，用 buffer 注册（跟着 `.to(device)` 走，但不更新）。"""),
    code("""class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("freqs", torch.ones(4, 8), persistent=False)

m = M()
print("在 state_dict 里吗:", "freqs" in m.state_dict())   # persistent=False -> 不在
print("在 parameters 里吗:", len(list(m.parameters())))    # 0
print("buffer:", m.freqs.shape)"""),
    md("""### 8.4 权重共享（weight tying）

输入嵌入和输出投影共用同一个矩阵。MiniMind 默认开启。"""),
    code("""class Tied(nn.Module):
    def __init__(self, vocab, d):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.lm_head.weight = self.embed.weight      # 共享！

t = Tied(100, 8)
print("共享后是同一个张量:", t.lm_head.weight is t.embed.weight)
print("参数量（应只有 100*8）:", sum(p.numel() for p in t.parameters()))"""),
    md("""### 8.5 交叉熵与 `ignore_index`

`labels` 里为 `-100` 的位置会被跳过——这正是 SFT 里「只算回答部分」的实现方式。"""),
    code("""import torch.nn.functional as F

logits = torch.randn(1, 3, 10)                  # [B, L, vocab]
labels = torch.tensor([[3, -100, 7]])           # 中间位置不计损失

loss = F.cross_entropy(logits.view(-1, 10), labels.view(-1), ignore_index=-100)
print("loss =", loss.item())

# 只有 2 个位置参与计算
print("有效标签数:", (labels != -100).sum().item())"""),
    *deps("""### 8.6 前置依赖：`RMSNorm` / `Attention` / `FeedForward`

组装需要前面几章的零件。下面给一份**精简的参考实现**（注意力这里用标准多头、签名简化成 `forward(x)`），
目的是让本章可以独立测试「组装」这件事本身。如果你前面几章已经写好了，欢迎换成你自己的版本。""",
    '''class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return (self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))).type_as(x)

class Attention(nn.Module):
    """精简版：标准多头注意力，签名 forward(x)"""
    def __init__(self, config):
        super().__init__()
        self.n_heads, self.head_dim = config.num_attention_heads, config.head_dim
        d = self.n_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, d, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, d, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, d, bias=False)
        self.o_proj = nn.Linear(d, config.hidden_size, bias=False)
    def forward(self, x):
        B, S, _ = x.shape
        q, k, v = (p(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
                   for p in (self.q_proj, self.k_proj, self.v_proj))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(B, S, -1))

class FeedForward(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        d = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, d, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, d, bias=False)
        self.down_proj = nn.Linear(d, config.hidden_size, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))'''),
    part2("现在把所有零件组装起来。这是本章最重要的一段代码。\n\n> 💡 每个留白代码块后面都跟了一个**测试块**。先写代码，再运行测试块。"),
    testkit(),
    md("""### 8.7 实现 `MiniMindBlock`

**要求**：
- `input_layernorm` / `post_attention_layernorm`：两个 RMSNorm
- `self_attn`：注意力（用上面的参考实现，签名是 `forward(x)`）
- `mlp`：`FeedForward` 或 `MOEFeedForward`（看 `config.use_moe`）
- `forward(self, x)`：

```python
residual = x
x = self.self_attn(self.input_layernorm(x))
x = x + residual
x = x + self.mlp(self.post_attention_layernorm(x))
return x
```"""),
    blank("MiniMindBlock", [
        "签名：MiniMindBlock(layer_id, config)，forward(self, x) 返回张量",
        "input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)",
        "post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)",
        "self_attn = Attention(config)",
        "mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)",
        "forward 里注意残差相加的时机",
    ], ["复用上面的 RMSNorm / Attention / FeedForward", "`nn.Module`"]),
    test("MiniMindBlock", '''def _t():
    torch.manual_seed(0)

    class Cfg:
        hidden_size = 16
        intermediate_size = 32
        num_attention_heads = 2
        head_dim = 8
        use_moe = False
        rms_norm_eps = 1e-6

    blk = MiniMindBlock(0, Cfg())
    x = torch.randn(2, 5, 16)
    y = blk(x)
    assert tuple(y.shape) == (2, 5, 16), f"输出形状应为 (2, 5, 16)，得到 {tuple(y.shape)}"
    with torch.no_grad():
        blk.self_attn.o_proj.weight.zero_()
        blk.mlp.down_proj.weight.zero_()
    close(blk(x), x, tol=1e-6,
          msg="把两个子层的输出投影置零后，残差应让输出严格等于输入 —— 说明残差连接不对")

check("MiniMindBlock 形状 / 残差连接", _t)'''),
    md("""### 8.8 实现 `MiniMindForCausalLM`（组装 + 损失）

**要求**：写出这个类。

```python
class MiniMindForCausalLM(nn.Module):
    def __init__(self, config):
        # embed_tokens: nn.Embedding(vocab_size, hidden_size)
        # layers: nn.ModuleList([MiniMindBlock(i, config) for i in range(num_hidden_layers)])
        # norm: RMSNorm(hidden_size, eps=config.rms_norm_eps)
        # lm_head: nn.Linear(hidden_size, vocab_size, bias=False)
        # 若 config.tie_word_embeddings：lm_head.weight = embed_tokens.weight

    def forward(self, input_ids, labels=None):
        # 返回 (logits, loss)；没传 labels 时 loss 为 None
```

**为什么要错位**：位置 t 的输出要预测位置 t+1 的词。

```python
x, y = logits[:, :-1, :], labels[:, 1:]
loss = F.cross_entropy(x.reshape(-1, vocab_size), y.reshape(-1), ignore_index=-100)
```"""),
    blank("MiniMindForCausalLM", [
        "__init__ 里搭好 embed_tokens / layers / norm / lm_head",
        "tie_word_embeddings 为 True 时，让 lm_head.weight 指向 embed_tokens.weight",
        "forward(self, input_ids, labels=None) 返回 (logits, loss)",
        "hidden = embed_tokens(input_ids)，逐个过 Block，最后过 norm",
        "logits = lm_head(hidden)，形状 [B, S, vocab_size]",
        "有 labels 时错位一位算交叉熵（ignore_index=-100），否则 loss = None",
    ], [
        "`nn.Embedding`、`nn.ModuleList`、`nn.Linear`",
        "`F.cross_entropy(..., ignore_index=-100)`",
        "`Tensor.reshape(-1, vocab_size)`",
    ]),
    test("MiniMindForCausalLM", '''def _t():
    torch.manual_seed(0)

    class Cfg:
        vocab_size = 64
        hidden_size = 16
        num_hidden_layers = 2
        intermediate_size = 32
        num_attention_heads = 2
        head_dim = 8
        use_moe = False
        rms_norm_eps = 1e-6
        tie_word_embeddings = True

    model = MiniMindForCausalLM(Cfg())
    ids = torch.randint(0, 64, (2, 6))
    out = model(ids)
    logits = out[0] if isinstance(out, tuple) else out
    assert tuple(logits.shape) == (2, 6, 64), f"logits 形状应为 (2, 6, 64)，得到 {tuple(logits.shape)}"
    assert model.lm_head.weight is model.embed_tokens.weight, \\
        "tie_word_embeddings=True 时，lm_head.weight 应就是 embed_tokens.weight 本身"
    out2 = model(ids, labels=ids.clone())
    loss = out2[1] if isinstance(out2, tuple) else getattr(out2, "loss", None)
    assert loss is not None, "传了 labels 却没有返回 loss"
    assert loss.dim() == 0 and torch.isfinite(loss), f"loss 应是有限的标量，得到 {loss}"
    loss.backward()
    assert model.embed_tokens.weight.grad is not None, "loss.backward() 之后嵌入层应该有梯度"

check("MiniMindForCausalLM 前向 / 权重共享 / 损失可反传", _t)'''),
    md("""### 8.9 验证参数量 ≈ 64M

组装完成后，数一下参数量，应该接近 64M（6400 词表 + 8 层 + 768 维）。"""),
    blank("verify_model", [
        "实例化完整的模型",
        "统计总参数量（M）",
        "构造 input_ids = torch.randint(0, 6400, (1, 16))，跑一次前向",
        "确认 logits 形状是 [1, 16, 6400]",
        "给它 labels 跑一次，确认能算出 loss",
    ], ["`sum(p.numel() for p in model.parameters()) / 1e6`"]),
    summary_cell(),
    part3([
        "能画出完整的模型数据流图",
        "Pre-Norm 与 Post-Norm 的区别说清楚了吗？",
        "权重共享省了多少参数？",
        "loss 的错位对齐做对了吗？（`logits[:-1]` vs `labels[1:]`）",
        "参数量接近 64M 吗？",
    ], "**思考题**：如果去掉所有残差连接，训练会发生什么？试试把残差去掉跑几步，观察 loss。"),
]

# ══════════════════════════════════════════════════════════════
# 09 Pretrain
# ══════════════════════════════════════════════════════════════
c4 = [
    header(9, "预训练 Pretrain", "第三部分 · 训练", "09_pretrain.html",
           "模型写好了，现在让它真正开始学习。预训练就是在大规模文本上反复做「下一词预测」。"),
    goals([
        "掌握 `optim.AdamW`、`GradScaler`、`autocast`、`clip_grad_norm_`",
        "理解梯度累积与有效 batch size 的关系",
        "能实现余弦学习率调度",
        "能写出完整的训练循环 `train_epoch`",
    ]),
    md("""## 📖 背景：训练循环的五个动作

```python
for input_ids, labels in loader:
    with autocast():                     # 1. 混合精度前向
        loss = model(input_ids, labels=labels).loss / accumulation_steps
    scaler.scale(loss).backward()        # 2. 反向传播
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), 1.0)   # 3. 梯度裁剪
    scaler.step(optimizer); scaler.update()    # 4. 更新参数
    optimizer.zero_grad(set_to_none=True)      # 5. 清空梯度
```

**梯度累积**：显存不够放不下大 batch 时，把多个小 batch 的梯度攒起来再更新一次，
等效于大 batch。注意 loss 要先除以累积步数。

**余弦学习率**：MiniMind 用 `lr_t = lr · (0.1 + 0.45·(1 + cos(π·step/total)))`。

---

"""),
    img(
        ("../images/pretrain_loss.jpg", "MiniMind 预训练损失曲线——loss 稳定下降即训练正常"),
    ),
    part1(),
    md("### 9.1 `optim.AdamW`"),
    code("""import torch
from torch import nn, optim

model = nn.Linear(4, 4)
opt = optim.AdamW(model.parameters(), lr=1e-3)

loss = model(torch.randn(2, 4)).sum()
loss.backward()
print("更新前 weight[0,0]:", model.weight[0, 0].item())
opt.step()
opt.zero_grad(set_to_none=True)
print("更新后 weight[0,0]:", model.weight[0, 0].item())
print("梯度已清空:", model.weight.grad is None)"""),
    md("""### 9.2 梯度裁剪：`clip_grad_norm_`

梯度突然变得很大（梯度爆炸）是训练崩溃的常见原因。裁剪把梯度的总范数限制在阈值内。"""),
    code("""model = nn.Linear(4, 4)
model.weight.grad = torch.full_like(model.weight, 100.0)   # 人为造一个巨大的梯度
norm_before = model.weight.grad.norm().item()
total = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"裁剪前范数: {norm_before:.1f}  裁剪后范数: {model.weight.grad.norm().item():.3f}")"""),
    md("""### 9.3 混合精度：`autocast` + `GradScaler`

用 bf16/fp16 计算，显存减半、速度更快。`GradScaler` 负责在 fp16 下防止梯度下溢。"""),
    code("""device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Linear(8, 8).to(device)
scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
opt = optim.AdamW(model.parameters(), lr=1e-3)

x = torch.randn(4, 8, device=device)
with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=torch.bfloat16):
    loss = model(x).pow(2).mean()
scaler.scale(loss).backward()
scaler.step(opt)
scaler.update()
print(f"混合精度前向成功，loss={loss.item():.4f}（设备 {device}）")"""),
    md("""### 9.4 余弦学习率调度"""),
    code("""import math

def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))

total = 1000
lrs = [get_lr(s, total, 5e-4) for s in range(total)]
print(f"初始 lr = {lrs[0]:.6f}   结束 lr = {lrs[-1]:.6f}   比值 = {lrs[-1]/lrs[0]:.2f}")

try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 3))
    plt.plot(lrs)
    plt.xlabel("step"); plt.ylabel("lr"); plt.title("Cosine LR Schedule")
    plt.grid(alpha=.3); plt.tight_layout(); plt.show()
except ImportError:
    print("未安装 matplotlib（`uv sync` 会装上）。用文本方式看曲线形状：")
    for i in range(0, total, 100):
        bar = "█" * int(lrs[i] / max(lrs) * 40)
        print(f"  step {i:4d}: {bar} {lrs[i]:.6f}")"""),
    md("""### 9.5 梯度累积\n\n累积 `n` 步再更新，等效 batch = `batch_size × n`。"""),
    code("""model = nn.Linear(4, 4)
opt = optim.AdamW(model.parameters(), lr=1e-2)
accumulation_steps = 4
before = model.weight[0, 0].item()

for i in range(accumulation_steps):
    loss = model(torch.randn(2, 4)).pow(2).mean() / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        opt.step(); opt.zero_grad(set_to_none=True)

print("参数已更新:", before != model.weight[0, 0].item())"""),
    part2("现在写出完整的训练循环。\n\n> 💡 每个留白代码块后面都跟了一个**测试块**。先写代码，再运行测试块。"),
    testkit(),
    md("""### 9.6 实现 `train_epoch`

**要求**：按上面「五个动作」的顺序实现一个 epoch 的训练，并每隔 `log_interval` 打印 loss。

**细节**：
- loss 要除以 `accumulation_steps` 再 backward
- 每步更新学习率：`param_group['lr'] = get_lr(...)`
- 只有累积到 `accumulation_steps` 时才 step + zero_grad
- **混合精度要能在 CPU 上跑**：用 `torch.autocast(device_type=..., enabled=...)`，
  或者设备是 CPU 时用 `contextlib.nullcontext()`"""),
    blank("train_epoch", [
        "签名：train_epoch(epoch, loader, iters, optimizer, scaler, model, args)",
        "循环里：取 batch -> to(args.device) -> 更新 lr -> 前向 -> loss / accumulation_steps",
        "scaler.scale(loss).backward()",
        "到累积点时：unscale_ -> clip_grad_norm_(..., args.grad_clip) -> step -> update -> zero_grad",
        "按 args.log_interval 打印 loss 和 lr",
        "autocast 要能在 CPU 上安全退化（nullcontext 或 enabled=False）",
    ], [
        "`torch.autocast`、`torch.cuda.amp.GradScaler`、`contextlib.nullcontext`",
        "`torch.nn.utils.clip_grad_norm_`",
        "`optimizer.param_groups`",
    ]),
    test("train_epoch", '''def _t():
    torch.manual_seed(0)
    from torch import nn, optim
    from torch.utils.data import DataLoader, TensorDataset

    class TinyLM(nn.Module):
        def __init__(self, vocab=16, d=8):
            super().__init__()
            self.emb = nn.Embedding(vocab, d)
            self.head = nn.Linear(d, vocab, bias=False)
        def forward(self, input_ids, labels=None):
            h = self.head(self.emb(input_ids))
            loss = None
            if labels is not None:
                x, y = h[:, :-1, :], labels[:, 1:]
                loss = F.cross_entropy(x.reshape(-1, h.size(-1)), y.reshape(-1), ignore_index=-100)
            return type("Out", (), {"loss": loss})()

    class Args:
        accumulation_steps = 2
        grad_clip = 1.0
        log_interval = 1000
        device = "cpu"
        dtype = "bfloat16"
        learning_rate = 1e-3
        epochs = 1

    model = TinyLM()
    ids = torch.randint(0, 16, (16, 10))
    loader = DataLoader(TensorDataset(ids, ids.clone()), batch_size=4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    before = [p.detach().clone() for p in model.parameters()]
    train_epoch(0, loader, len(loader), optimizer, scaler, model, Args())
    after = [p.detach() for p in model.parameters()]
    assert any(not torch.allclose(b, a) for b, a in zip(before, after)), \\
        "跑完一个 epoch 参数没变化 —— 检查 optimizer.step() 有没有被调用"

check("train_epoch 能跑通并更新参数", _t)'''),
    md("""### 9.7 实现 `get_lr`

把上面演示的余弦调度写成函数。"""),
    blank("get_lr", [
        "签名：get_lr(current_step, total_steps, lr)",
        "返回 lr * (0.1 + 0.45 * (1 + cos(pi * current_step / total_steps)))",
    ], ["`math.cos`、`math.pi`"]),
    test("get_lr", '''def _t():
    close(get_lr(0, 100, 5e-4), 5e-4, tol=1e-12, msg="起始学习率应等于 lr")
    close(get_lr(100, 100, 5e-4), 0.1 * 5e-4, tol=1e-12, msg="结束时学习率应衰减到 0.1·lr")
    mid = get_lr(50, 100, 5e-4)
    assert 0.1 * 5e-4 < mid < 5e-4, f"中点学习率应介于两者之间，得到 {mid}"
    assert get_lr(0, 100, 1e-3) > get_lr(50, 100, 1e-3) > get_lr(100, 100, 1e-3), "学习率应单调递减"

check("get_lr 余弦调度", _t)'''),
    summary_cell(),
    part3([
        "能说清梯度累积为什么能省显存",
        "为什么 loss 要先除以 accumulation_steps？",
        "梯度裁剪应该放在 scaler.step 之前还是之后？（提示：unscale 之后）",
        "余弦调度的起点和终点分别是多少？",
    ], "**思考题**：如果学习率恒定不衰减，训练后期 loss 会怎样？为什么余弦衰减有效？"),
]

# ══════════════════════════════════════════════════════════════
# 10 SFT
# ══════════════════════════════════════════════════════════════
c5 = [
    header(10, "指令微调 SFT", "第三部分 · 训练", "10_sft.html",
           "预训练后的模型会「接话」但不会「听话」。SFT 教它遵守对话格式，关键技巧是——损失只算回答部分。"),
    goals([
        "理解 chat template 的作用",
        "掌握 `apply_chat_template` 的用法",
        "理解 loss mask 与 `ignore_index=-100`",
        "能实现 `generate_labels`（只对 assistant 回答计损失）",
    ]),
    md("""## 📖 背景：为什么要有 loss mask

如果把用户的问题也拿来算损失，模型会去「学习生成用户的问题」——这是浪费甚至有害的。

正确做法：把问题部分的标签设成 `-100`，交叉熵会自动忽略。

```python
labels = [-100, -100, ..., 回答的 token id, ...]
loss = F.cross_entropy(logits, labels, ignore_index=-100)
```

MiniMind 的实现思路：先分词出 `bos_id = tokenizer("<|im_start|>assistant\\n")`，
然后扫描序列，找到每个 `bos_id` 的位置，从它之后开始把标签设为原文 id，直到遇到 `eos_id`。

---

"""),
    img(
        ("../images/sft_loss.jpg", "MiniMind SFT 阶段损失曲线"),
    ),
    part1(),
    md("### 10.1 对话数据长什么样"),
    code("""import json

conversations = [
    {"role": "system", "content": "你是一个知识丰富的AI，尽力为用户提供准确的信息。"},
    {"role": "user", "content": "1+1等于几？"},
    {"role": "assistant", "content": "1+1=2。"},
]
print(json.dumps(conversations, ensure_ascii=False, indent=2))"""),
    md("""### 10.2 `apply_chat_template`

分词器自带模板，会把对话列表渲染成带特殊 token 的字符串。"""),
    code("""from pathlib import Path

tok_path = Path("../data")
has_tok = (tok_path / "tokenizer_config.json").exists()

if has_tok:
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(tok_path))
        prompt = tok.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
        print(prompt)
        print("--- 编码后 ---")
        print(tok(prompt).input_ids[:30])
    except ImportError:
        has_tok = False
        print("未安装 transformers —— 请先执行 `uv sync`。")

if not has_tok:
    # 没有官方分词器时，用一段手工模板演示同样的结构
    prompt = ("<|im_start|>system\\n你是一个AI助手<|im_end|>\\n"
              "<|im_start|>user\\n1+1等于几？<|im_end|>\\n"
              "<|im_start|>assistant\\n1+1=2。<|im_end|>\\n")
    print("（未找到官方分词器，使用手工模板演示）\\n")
    print(prompt)"""),
    md("""### 10.3 构造 loss mask 的基本工具"""),
    code("""n = 10
labels = [-100] * n
print("初始全掩码:", labels)

# 假设从第 6 个位置开始是回答
for j in range(6, 10):
    labels[j] = j + 100        # 用假 id 代替
print("只标记回答部分:", labels)"""),
    md("""### 10.4 子序列查找：找到 `bos_id` 出现的位置

这是 `generate_labels` 的核心——在 token 序列里找「assistant 开始」标记。"""),
    code("""ids = [1, 5, 6, 7, 2, 1, 8, 9, 2, 3]
bos_id = [1]
eos_id = [2]

i, spans = 0, []
while i < len(ids):
    if ids[i:i + len(bos_id)] == bos_id:
        start = i + len(bos_id)
        end = start
        while end < len(ids) and ids[end:end + len(eos_id)] != eos_id:
            end += 1
        spans.append((start, end + len(eos_id)))
        i = end + len(eos_id)
    else:
        i += 1
print("回答区间:", spans, " <- 每个区间内的 token 才计损失")"""),
    part2("SFT 的核心就是 `generate_labels`。请自己写出来。\n\n> 💡 每个留白代码块后面都跟了一个**测试块**。先写代码，再运行测试块。"),
    testkit(),
    md("""### 10.5 实现 `generate_labels`

**要求**：写一个**独立函数**（方便单独测试），把一条已经拼好模板并分词的序列变成 loss mask 标签。

```python
def generate_labels(input_ids, bos_id, eos_id, max_length):
    # input_ids 是 Python list；返回同长度的 labels，默认全 -100
```

规则：
- 每当匹配到子序列 `bos_id`（即 `<|im_start|>assistant\\n`），就从它**之后**开始，
  把标签设为对应的 `input_ids`，直到遇到 `eos_id`（含 eos）
- 边界不要越界；超过 `max_length` 的部分截断
- 循环要能处理**多轮对话**（跳过一段后继续往后扫描）

**验证**：打印出哪些位置计入了损失，确认只有 assistant 的回答部分。"""),
    blank("generate_labels", [
        "签名：generate_labels(input_ids, bos_id, eos_id, max_length)",
        "labels = [-100] * len(input_ids)",
        "用 while 扫描，用切片比较匹配 bos_id 子序列",
        "匹配到后，从 start 开始把 input_ids 复制进 labels，直到遇到 eos_id",
        "跳过整个区间后，i 前进到 eos 之后继续扫描下一轮",
    ], [
        "列表切片比较 `input_ids[i:i+len(bos_id)] == bos_id`",
        "`min(end + len(eos_id), max_length)`",
    ]),
    test("generate_labels", '''def _t():
    ids = [1, 5, 6, 7, 2, 1, 8, 9, 2, 3]
    labels = generate_labels(ids, bos_id=[1], eos_id=[2], max_length=len(ids))
    assert len(labels) == len(ids), f"labels 长度应与 input_ids 一致，得到 {len(labels)}"
    assert labels[0] == -100, "标记之外的 token 应为 -100"
    assert labels[1:5] == [5, 6, 7, 2], f"第一个回答区间应为 [5, 6, 7, 2]，得到 {labels[1:5]}"
    assert labels[5] == -100, "bos_id 自身不应计入损失"
    assert labels[6:9] == [8, 9, 2], f"第二个回答区间应为 [8, 9, 2]，得到 {labels[6:9]}"
    assert labels[9] == -100, "尾部未匹配的 token 应为 -100"

check("generate_labels 只标记 assistant 回答", _t)'''),
    md("""### 10.6 实现 `SFTDataset`

**要求**：写出这个 Dataset 类。

```python
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        # 用 json 逐行读 jsonl，每行形如 {"conversations": [...]}
        # 预先算好 self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\\n",
        #                                   add_special_tokens=False).input_ids
        #        self.eos_id = tokenizer(f"{tokenizer.eos_token}\\n",
        #                                add_special_tokens=False).input_ids

    def __getitem__(self, index):
        # 1. 取出 conversations，套模板得到 prompt
        # 2. input_ids = tokenizer(prompt).input_ids[:max_length]
        # 3. 补齐 pad 到 max_length
        # 4. labels = generate_labels(input_ids, self.bos_id, self.eos_id, max_length)
        # 5. 返回两个 torch.long 张量
```

> `generate_labels` 是上面那节写的**模块级函数**，直接调用即可。"""),
    blank("SFTDataset", [
        "继承 torch.utils.data.Dataset",
        "__init__ 里用 json 逐行读 jsonl，并预先算好 self.bos_id / self.eos_id",
        "__getitem__ 里套模板（可自己拼一个简单模板，或用 apply_chat_template）",
        "分词、截断到 max_length、补齐 pad",
        "调用上面写好的 generate_labels 得到 labels",
        "返回 (input_ids, labels)，都是 torch.long 张量",
    ], [
        "`tokenizer.apply_chat_template(convs, tokenize=False)`",
        "`tokenizer(prompt).input_ids`",
        "上面写好的 `generate_labels`",
    ]),
    test("SFTDataset", '''def _t():
    import json, tempfile, os

    class FakeTok:
        pad_token_id = 0
        bos_token, eos_token = "<A>", "<E>"
        def apply_chat_template(self, convs, tokenize=False, add_generation_prompt=False):
            out = ""
            for m in convs:
                if m["role"] == "assistant":
                    out += "<A>assistant\\n" + m["content"] + "<E>\\n"
                else:
                    out += m["content"]
            return out
        def __call__(self, text, add_special_tokens=True, max_length=None, truncation=False):
            ids, i = [], 0
            while i < len(text):
                if text.startswith("<A>", i):
                    ids.append(1); i += 3
                elif text.startswith("<E>", i):
                    ids.append(2); i += 3
                else:
                    ids.append(3 + (ord(text[i]) % 20)); i += 1
            return type("Enc", (), {"input_ids": ids[:max_length] if max_length else ids})()

    path = os.path.join(tempfile.gettempdir(), "mm_sft_ds_test.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"conversations": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]}, ensure_ascii=False) + "\\n")

    ds = SFTDataset(path, FakeTok(), max_length=32)
    x, y = ds[0]
    assert tuple(x.shape) == (32,), f"input_ids 形状应为 (32,)，得到 {tuple(x.shape)}"
    assert tuple(y.shape) == (32,), f"labels 形状应为 (32,)，得到 {tuple(y.shape)}"
    valid = y != -100
    assert (~valid).any(), "问题部分应该被掩码成 -100"
    assert valid.any(), "assistant 的回答部分应该计入损失"
    same(y[valid], x[valid], msg="计入损失的标签应等于 input_ids 本身")
    assert valid.sum().item() >= 2, f"计入损失的 token 太少（{valid.sum().item()} 个），掩码可能不对"

check("SFTDataset 只对 assistant 回答计损失", _t)'''),
    summary_cell(),
    part3([
        "能解释为什么问题部分要设成 -100",
        "`bos_id` 为什么包含 `assistant\\n` 而不只是 `<|im_start|>`？",
        "你的 `generate_labels` 在 eos 处正确闭合了吗？",
        "一条样本里大约有多少比例的 token 参与损失？",
    ], "**思考题**：如果多轮对话里每一轮 assistant 都要计损失，你的实现能处理吗？（提示：`while` 循环会继续往后扫描）"),
]

for name, cells in [
    ("06_swiglu.ipynb", c),
    ("07_moe.ipynb", c2),
    ("08_block_model.ipynb", c3),
    ("09_pretrain.ipynb", c4),
    ("10_sft.ipynb", c5),
]:
    save(name, cells)
