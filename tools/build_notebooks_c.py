"""生成第 11–15 章的 Notebook。"""
from nbcommon import (md, code, blank, img, header, goals, part1, part2, part3, save,
                        testkit, test, summary_cell, deps)

# ══════════════════════════════════════════════════════════════
# 11 LoRA
# ══════════════════════════════════════════════════════════════
c = [
    header(11, "LoRA 低秩适配", "第三部分 · 训练", "11_lora.html",
           "想给模型加一项新技能，不必重训全部参数。LoRA 冻结底座，只训练一条低秩小分支。"),
    goals([
        "理解低秩分解为什么有效",
        "掌握 `setattr` 动态挂模块、`named_modules` 遍历",
        "理解 A/B 的初始化策略",
        "能从零实现 `LoRA`、`apply_lora` 与权重合并",
    ]),
    img(
        ("../images/LLM-structure.jpg", "LoRA 挂在这些层上：架构里所有 in_features == out_features 的方阵线性层（Q/K/V/O 投影、FFN 三投影）"),
    ),
    md("""## 📖 背景：只训练 1% 的参数

```python
h = W @ x  +  (α/r) · B @ A @ x
      ↑              ↑
   冻结不变      只训练这两个小矩阵

A: (in_features, r)   高斯初始化
B: (r, out_features)  零初始化 → 初始时 B@A = 0，不破坏原模型
```

为什么有效？因为微调时权重的**变化量 ΔW 本身就是低秩的**。
用一个 `d×r` 加一个 `r×d` 的小矩阵（r 远小于 d）就足够表达。

**B 为什么零初始化**：这样训练开始时 `W + B@A = W`，模型行为与原来完全一致，
不会有「一开始就被随机扰动」的问题。

---

"""),
    part1(),
    md("### 11.1 `nn.Linear` 的 `in_features` / `out_features`"),
    code("""import torch
from torch import nn

lin = nn.Linear(768, 768, bias=False)
print("in_features :", lin.in_features)
print("out_features:", lin.out_features)
print("weight 形状 :", lin.weight.shape)
print("参数量      :", lin.weight.numel())"""),
    md("""### 11.2 低秩分解：`B @ A` 的形状

`A` 把 768 维压到 r 维，`B` 再把它升回 768 维。"""),
    code("""d, r = 768, 16
A = nn.Linear(d, r, bias=False)      # [16, 768]
B = nn.Linear(r, d, bias=False)      # [768, 16]

x = torch.randn(2, d)
print("x          :", x.shape)
print("A(x)       :", A(x).shape)
print("B(A(x))    :", B(A(x)).shape)
print("增量矩阵形状:", (B.weight @ A.weight).shape)   # [768, 768]
print("LoRA 参数量 :", A.weight.numel() + B.weight.numel())
print("全参数量    :", d * d)"""),
    md("""### 11.3 初始化：A 高斯、B 零

这是 LoRA 能「无损挂载」的关键。"""),
    code("""A = nn.Linear(768, 16, bias=False)
B = nn.Linear(16, 768, bias=False)
A.weight.data.normal_(mean=0.0, std=0.02)
B.weight.data.zero_()

x = torch.randn(2, 768)
delta = B(A(x))
print("B@A 初始输出是否全为 0:", torch.allclose(delta, torch.zeros_like(delta)))"""),
    md("""### 11.4 `setattr` 动态挂模块

LoRA 要在运行时给已有的 `nn.Linear` 挂一个 `lora` 子模块。"""),
    code("""class Holder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False)

h = Holder()
setattr(h.proj, "lora", nn.Linear(8, 2, bias=False))     # 动态挂载
print("挂上了吗:", hasattr(h.proj, "lora"))
print("会被注册为子模块:", "proj.lora" in [n for n, _ in h.named_modules()])"""),
    md("""### 11.5 替换 forward：让原输出加上 LoRA 增量

用一个闭包保存原 forward，再替换掉。"""),
    code("""class Holder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False)

h = Holder()
lora = nn.Linear(8, 8, bias=False)     # 简化：直接用满秩代替 A@B
setattr(h.proj, "lora", lora)

original_forward = h.proj.forward
def forward_with_lora(x, layer1=original_forward, layer2=lora):
    return layer1(x) + layer2(x)
h.proj.forward = forward_with_lora

x = torch.randn(2, 8)
print("原输出 + LoRA 增量:", h.proj(x).shape)
print("差值 == lora(x):", torch.allclose(h.proj(x) - original_forward(x), lora(x), atol=1e-5))"""),
    md("""### 11.6 权重合并\n\n推理时把 `B@A` 加回 `W`，就没有额外计算开销了。"""),
    code("""d, r = 8, 2
W = nn.Linear(d, d, bias=False)
A = nn.Linear(d, r, bias=False)
B = nn.Linear(r, d, bias=False)
A.weight.data.normal_(0, 0.1)

W_merged = W.weight.data.clone() + (B.weight.data @ A.weight.data)
print("合并前后形状一致:", W_merged.shape == W.weight.shape)
print("合并后参数总量不变:", W_merged.numel())"""),
    part2("LoRA 的三个部分都要自己写：类定义、挂载、合并。\n\n> 💡 每个留白代码块后面都跟了一个**测试块**。先写代码，再运行测试块。"),
    testkit(),
    md("""### 11.7 实现 `LoRA` 类

**要求**：
- `__init__(self, in_features, out_features, rank)`
- `self.A = nn.Linear(in_features, rank, bias=False)`，权重用 `normal_(0, 0.02)` 初始化
- `self.B = nn.Linear(rank, out_features, bias=False)`，权重**零初始化**
- `forward(x)` 返回 `B(A(x))`"""),
    blank("LoRA", [
        "继承 nn.Module",
        "A: in_features -> rank（高斯初始化 std=0.02）",
        "B: rank -> out_features（零初始化）",
        "forward(x) = self.B(self.A(x))",
    ], ["`nn.Linear(..., bias=False)`", "`Tensor.data.normal_` / `.zero_()`"]),
    test("LoRA", '''def _t():
    torch.manual_seed(0)
    lora = LoRA(16, 16, rank=4)
    assert tuple(lora.A.weight.shape) == (4, 16), f"A 形状应为 (4, 16)，得到 {tuple(lora.A.weight.shape)}"
    assert tuple(lora.B.weight.shape) == (16, 4), f"B 形状应为 (16, 4)，得到 {tuple(lora.B.weight.shape)}"
    close(lora(torch.randn(3, 16)), torch.zeros(3, 16), tol=1e-8,
          msg="B 零初始化时 LoRA 初始输出必须为 0，否则挂载会破坏原模型")
    n = sum(p.numel() for p in lora.parameters())
    assert n == 4 * 16 + 16 * 4, f"参数量应为 {4 * 16 + 16 * 4}，得到 {n}"

check("LoRA 形状 / 零初始化 / 参数量", _t)'''),
    md("""### 11.8 实现 `apply_lora`

**要求**：遍历 `model.named_modules()`，对每个 `nn.Linear` 且 `in_features == out_features` 的层，
挂上 `LoRA`，并把它的 forward 替换成「原输出 + LoRA 输出」。

**注意**：闭包要用默认参数绑定 `original_forward` 和 `lora`，
否则循环变量会被后续迭代覆盖（经典的 Python 闭包陷阱）。"""),
    blank("apply_lora", [
        "签名：apply_lora(model, rank=16)",
        "遍历 model.named_modules()",
        "判断 isinstance(module, nn.Linear) and module.in_features == module.out_features",
        "创建 LoRA 并 setattr(module, 'lora', lora)",
        "替换 module.forward 为 forward_with_lora(x, layer1=原forward, layer2=lora)",
    ], [
        "`model.named_modules()`",
        "`setattr`",
        "默认参数绑定闭包变量",
    ]),
    test("apply_lora", '''def _t():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8, bias=False), nn.Linear(8, 4, bias=False))
    x = torch.randn(2, 8)
    before = model(x).detach().clone()

    apply_lora(model, rank=2)
    assert hasattr(model[0], "lora"), "in_features == out_features 的层应该被挂上 lora"
    assert not hasattr(model[1], "lora"), "in_features != out_features 的层不该被挂 lora"
    close(model(x), before, tol=1e-6,
          msg="挂载 LoRA 后原模型的输出变了 —— 检查 B 是不是零初始化")

    model(x).sum().backward()
    assert model[0].lora.A.weight.grad is not None, "LoRA 的 A 应该能收到梯度"

check("apply_lora 挂载 / 不改变原输出", _t)'''),
    md("""### 11.9 实现权重合并 `merge_lora`

**要求**：对每个带 `.lora` 的 `nn.Linear`，执行 `W += B.weight @ A.weight`，
然后把所有 `.lora.` 相关的键从 state_dict 里剔除，保存到 `save_path`。

**提示**：`state_dict = {k: v for k, v in model.state_dict().items() if '.lora.' not in k}`"""),
    blank("merge_lora", [
        "签名：merge_lora(model, save_path)",
        "遍历 named_modules，找到有 .lora 属性的 nn.Linear",
        "module.weight.data += module.lora.B.weight.data @ module.lora.A.weight.data",
        "导出 state_dict 时过滤掉 '.lora.' 键",
        "torch.save 到 save_path",
    ], ["`torch.save`", "字典推导式过滤键"]),
    test("merge_lora", '''def _t():
    import tempfile, os
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8, bias=False))
    apply_lora(model, rank=2)
    with torch.no_grad():
        model[0].lora.A.weight.normal_(0, 0.1)
        model[0].lora.B.weight.normal_(0, 0.1)

    x = torch.randn(2, 8)
    with_lora = model(x).detach().clone()
    expected_W = (model[0].weight.detach()
                  + model[0].lora.B.weight.detach() @ model[0].lora.A.weight.detach())

    path = os.path.join(tempfile.gettempdir(), "mm_lora_merged.pth")
    merge_lora(model, path)
    sd = torch.load(path, map_location="cpu")
    assert not any(".lora." in k for k in sd), f"合并后的权重里不该再有 lora 参数：{list(sd.keys())}"
    merged_W = sd["0.weight"]
    close(merged_W, expected_W, tol=1e-4, msg="合并后的权重应等于 W + B@A")
    close(x @ merged_W.T, with_lora, tol=1e-4, msg="用合并权重算出的输出应与原来一致")

check("merge_lora 权重合并正确", _t)'''),
    summary_cell(),
    part3([
        "能解释为什么 B 要零初始化",
        "挂载后原模型的行为会变吗？（不应该变）",
        "参数量降到多少？",
        "合并后的权重与「W + BA」一致吗？",
    ], "**思考题**：rank 越大效果一定越好吗？为什么？"),
]

# ══════════════════════════════════════════════════════════════
# 12 Distillation
# ══════════════════════════════════════════════════════════════
c2 = [
    header(12, "知识蒸馏", "第三部分 · 训练", "12_distill.html",
           "让一个小模型去模仿一个大模型。关键是：老师输出的完整概率分布里藏着「暗知识」。"),
    goals([
        "理解黑盒蒸馏与白盒蒸馏的区别",
        "掌握 `F.kl_div`、`F.log_softmax` 与温度缩放",
        "理解为什么 KL 里 input 传 log-prob、target 传概率",
        "能实现组合了 CE 与 KL 的蒸馏损失",
    ]),
    img(
        ("../images/LLM-structure.jpg", "学生模型就是这个架构；老师通常是一个结构相同但规模更大的模型，用它的输出分布来指导学生"),
    ),
    md("""## 📖 背景：软标签里的暗知识

假设正确答案是「猫」。硬标签只说「猫=1，其它=0」。
但老师的软标签可能是「猫 0.7，狗 0.2，兔子 0.05」——
这告诉学生**「狗比兔子更接近猫」**。这种相似性信息是硬标签完全没有的。

```python
loss = α · CE(student_logits, hard_label)
     + (1-α) · T² · KL( softmax(teacher/T) ‖ softmax(student/T) )
```

乘 `T²` 是为了补偿温度把梯度变小的问题。温度 `T > 1` 会把分布摊平，让暗知识更明显。

---

"""),
    part1(),
    md("### 12.1 温度缩放：`logits / T`"),
    code("""import torch
import torch.nn.functional as F

logits = torch.tensor([[3.2, 2.1, 1.4, 0.2, -0.6, -1.1]])

for T in [1.0, 2.0, 4.0]:
    p = F.softmax(logits / T, dim=-1)
    print(f"T={T}: {[round(x, 3) for x in p.squeeze().tolist()]}")
print("\\n观察：T 越大分布越平，低概率词越明显")"""),
    md("""### 12.2 `F.log_softmax`：更稳定的 log 概率"""),
    code("""x = torch.tensor([[3.2, 2.1, 1.4]])
print("log_softmax  :", F.log_softmax(x, dim=-1))
print("log(softmax) :", torch.log(F.softmax(x, dim=-1)))
print("两者一致:", torch.allclose(F.log_softmax(x, -1), torch.log(F.softmax(x, -1)), atol=1e-6))"""),
    md("""### 12.3 `F.kl_div` 的参数约定（最容易错的地方）

`F.kl_div(input, target)` 计算的是 `KL(target ‖ exp(input))`。

- `input`：**log 概率**（用 `log_softmax`）
- `target`：**概率**（用 `softmax`）

```python
kl = F.kl_div(F.log_softmax(student/T, -1), F.softmax(teacher/T, -1), reduction="batchmean")
```"""),
    code("""student = torch.tensor([[2.4, 1.6, 0.9, 0.5, -0.2, -0.5]])
teacher = torch.tensor([[3.2, 2.1, 1.4, 0.2, -0.6, -1.1]])
T = 2.0

kl = F.kl_div(F.log_softmax(student / T, dim=-1),
              F.softmax(teacher / T, dim=-1),
              reduction="batchmean") * (T ** 2)
print("KL(teacher ‖ student) =", kl.item())

# 手工验证
p = F.softmax(teacher / T, -1)
q = F.softmax(student / T, -1)
manual = (p * (p.log() - q.log())).sum()
print("手工计算            =", manual.item())"""),
    md("""### 12.4 组合损失：CE + KL"""),
    code("""def distill_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    ce = F.cross_entropy(student_logits, labels, ignore_index=-100)
    kl = F.kl_div(F.log_softmax(student_logits / T, -1),
                  F.softmax(teacher_logits / T, -1),
                  reduction="batchmean") * (T ** 2)
    return alpha * ce + (1 - alpha) * kl, ce.item(), kl.item()

s = torch.randn(4, 100)
t = torch.randn(4, 100)
y = torch.randint(0, 100, (4,))
loss, ce, kl = distill_loss(s, t, y)
print(f"总损失={loss.item():.4f}  CE={ce:.4f}  KL={kl:.4f}")"""),
    part2("现在自己实现完整的蒸馏损失。\n\n> 💡 每个留白代码块后面都跟了一个**测试块**。先写代码，再运行测试块。"),
    testkit(),
    md("""### 12.5 实现 `distillation_loss`

**要求**：
1. 算硬标签交叉熵：`F.cross_entropy(student_logits, labels, ignore_index=-100)`
2. 算软标签 KL：
   - `F.log_softmax(student_logits / T, dim=-1)` 作为 input
   - `F.softmax(teacher_logits / T, dim=-1)` 作为 target
   - `reduction="batchmean"`，再乘 `T ** 2`
3. 返回 `alpha * ce + (1 - alpha) * kl`

**验证**：当学生 logits 完全等于老师时，KL 应该接近 0。"""),
    blank("distillation_loss", [
        "签名：distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5)",
        "ce = F.cross_entropy(student_logits, labels, ignore_index=-100)",
        "kl = F.kl_div(log_softmax(student/T), softmax(teacher/T), reduction='batchmean') * T**2",
        "return alpha * ce + (1 - alpha) * kl",
        "（可选）同时返回 ce 和 kl 的数值方便观察",
    ], [
        "`F.cross_entropy`、`F.kl_div`",
        "`F.log_softmax`、`F.softmax`",
    ]),
    test("distillation_loss", '''def _t():
    torch.manual_seed(0)
    s = torch.randn(4, 50)
    t = torch.randn(4, 50)
    y = torch.randint(0, 50, (4,))
    out = distillation_loss(s, t, y, T=2.0, alpha=0.5)
    loss = out[0] if isinstance(out, tuple) else out
    assert torch.isfinite(loss) and loss.dim() == 0, f"应返回有限标量，得到 {loss}"

    ce = F.cross_entropy(s, y, ignore_index=-100)
    out2 = distillation_loss(s, s.clone(), y, T=2.0, alpha=0.5)
    loss2 = out2[0] if isinstance(out2, tuple) else out2
    close(loss2.item(), (0.5 * ce).item(), tol=1e-4,
          msg="学生与老师完全一致时 KL 应为 0，loss 应等于 alpha * CE")

check("distillation_loss 数值 / 学生=老师时 KL 归零", _t)'''),
    md("""### 12.6 实现「黑盒蒸馏」的数据构造

黑盒蒸馏不需要老师的 logits，只需要老师生成的文本。写一个函数：
给定一批 prompt，用老师模型生成回答，组装成 SFT 格式的数据。

（没有真实老师模型时，用假数据模拟即可，重点是理解流程。）"""),
    blank("build_blackbox_data", [
        "签名：build_blackbox_data(prompts, teacher, tokenizer)",
        "对每个 prompt 调用 teacher.generate(...)，再用 tokenizer.decode 转成文本",
        "把 (prompt, teacher_output) 组成 {'conversations': [{'role': 'user', ...}, {'role': 'assistant', ...}]}",
        "返回与 prompts 等长的样本列表（每项都含 'conversations'）",
    ], ["`teacher.generate`", "`tokenizer.decode`"]),
    test("build_blackbox_data", '''def _t():
    class FakeTeacher:
        def generate(self, *a, **kw):
            return torch.zeros(1, 1, dtype=torch.long)
    class FakeTok:
        def apply_chat_template(self, convs, tokenize=False, add_generation_prompt=True):
            return "PROMPT:" + convs[-1]["content"]
        def decode(self, ids, skip_special_tokens=True):
            return "老师的回答"

    data = build_blackbox_data(["1+1=?"], FakeTeacher(), FakeTok())
    assert isinstance(data, list) and len(data) == 1, f"应返回与 prompts 等长的列表，得到 {type(data)}"
    sample = data[0]
    assert "conversations" in sample, f"每个样本应有 conversations 字段，得到 {list(sample.keys())}"
    roles = [m["role"] for m in sample["conversations"]]
    assert "assistant" in roles, f"样本里应包含 assistant 回复，得到 {roles}"

check("build_blackbox_data 结构", _t)'''),
    summary_cell(),
    part3([
        "能说清「暗知识」到底指什么",
        "温度 T 增大时分布怎么变？",
        "KL 里为什么 input 是 log 概率、target 是概率？",
        "T² 缩放为什么必要？",
        "学生等于老师时 KL 接近 0 吗？",
    ], "**思考题**：蒸馏和 SFT 的区别到底在哪？如果老师的输出本身就是错的，蒸馏会怎样？"),
]

# ══════════════════════════════════════════════════════════════
# 13 DPO
# ══════════════════════════════════════════════════════════════
c3 = [
    header(13, "DPO 直接偏好优化", "第三部分 · 训练", "13_dpo.html",
           "DPO 把 RLHF 那一整套复杂流程，化简成一个二分类式的损失函数。只用现成的好坏答案数据就能对齐。"),
    goals([
        "理解 DPO 损失公式的每一项",
        "掌握用 `log_softmax` + `gather` 取特定 token 的 log 概率",
        "掌握用 loss_mask 把逐 token logprob 汇总成整句 logprob",
        "能实现完整的 `dpo_loss`",
    ]),
    img(
        ("../images/dataset.jpg", "DPO 用的是偏好数据集 dpo.jsonl（chosen / rejected 成对出现），就在这张数据构成图里"),
    ),
    md("""## 📖 背景：一个损失函数搞定偏好对齐

```python
ratio_w = logπ(y_w|x) − logπ_ref(y_w|x)     # 好答案相对参考模型的提升
ratio_l = logπ(y_l|x) − logπ_ref(y_l|x)     # 坏答案的提升

loss = −log σ( β · (ratio_w − ratio_l) )
```

直观理解：**让「好答案」相对参考模型被抬得更高，让「坏答案」被压得更低。**

`π_ref` 是冻结的 SFT 模型。减掉它的 log 概率，相当于只奖励「比原来更好的变化」，
防止模型为了刷分而偏离语言本身。

---

"""),
    part1(),
    md("### 13.1 取特定位置、特定 token 的 log 概率\n\n`log_softmax` + `gather` 是标准组合。"),
    code("""import torch
import torch.nn.functional as F

logits = torch.randn(2, 5, 10)              # [B, L, vocab]
labels = torch.randint(0, 10, (2, 5))       # 每个位置的目标 token

logp = F.log_softmax(logits, dim=-1)                    # [B, L, vocab]
picked = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)   # [B, L]
print("logits :", logits.shape)
print("logp   :", logp.shape)
print("picked :", picked.shape, " <- 每个位置目标 token 的 log 概率")

# 验证：手工取出第一个位置
print("手工验证:", logp[0, 0, labels[0, 0]].item(), "==", picked[0, 0].item())"""),
    md("""### 13.2 用 mask 汇总整句的 log 概率\n\n回答只占序列的一部分，要用 mask 把非回答位置去掉。"""),
    code("""mask = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 0, 1, 1]])      # 1 = 属于回答
seq_logp = (picked * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
print("每条回答的平均 log 概率:", seq_logp.tolist())"""),
    md("""### 13.3 `F.logsigmoid` 与 sigmoid"""),
    code("""x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
print("sigmoid    :", torch.sigmoid(x).tolist())
print("logsigmoid :", F.logsigmoid(x).tolist())
print("-log(sigmoid(x)):", (-F.logsigmoid(x)).tolist())"""),
    md("""### 13.4 手推一遍 DPO 损失"""),
    code("""beta = 0.1
# 策略模型的 log 概率
logp_w, logp_l = -1.5, -2.8
# 参考模型的 log 概率
ref_w, ref_l = -3.4, -3.2

ratio_w = logp_w - ref_w
ratio_l = logp_l - ref_l
margin = beta * (ratio_w - ratio_l)
loss = -F.logsigmoid(torch.tensor(margin))

print(f"ratio_w = {ratio_w:.3f}   ratio_l = {ratio_l:.3f}")
print(f"margin  = {margin:.3f}")
print(f"loss    = {loss.item():.4f}")
print(f"隐式偏好准确率 σ(margin) = {torch.sigmoid(torch.tensor(margin)).item():.4f}")"""),
    part2("现在实现完整的 DPO 损失。\n\n> 💡 每个留白代码块后面都跟了一个**测试块**。先写代码，再运行测试块。"),
    testkit(),
    md("""### 13.5 实现 `compute_logprob`

**要求**：给定 `logits [B, L, V]`、`labels [B, L]`、`mask [B, L]`，
返回每条序列的 log 概率（标量，形状 `[B]`）。

步骤：
1. `logp = F.log_softmax(logits, dim=-1)`
2. `picked = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)`
3. `(picked * mask).sum(-1) / mask.sum(-1).clamp(min=1)`

**注意**：labels 里为 -100 的位置要先用 mask 归零，否则 `gather` 会越界。"""),
    blank("compute_logprob", [
        "签名：compute_logprob(logits, labels, mask)",
        "logits: [B, L, V]，labels: [B, L]，mask: [B, L]（1=计入）",
        "先把 labels 中 -100 的位置替换成 0（避免 gather 越界）",
        "logp = F.log_softmax(logits, -1)",
        "picked = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)",
        "返回 (picked * mask).sum(-1) / mask.sum(-1).clamp(min=1)",
    ], ["`F.log_softmax`", "`Tensor.gather`、`Tensor.unsqueeze`、`Tensor.squeeze`", "`Tensor.clamp`"]),
    test("compute_logprob", '''def _t():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 10)
    labels = torch.randint(0, 10, (2, 5))
    mask = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 0, 1, 1]])

    out = compute_logprob(logits, labels, mask)
    assert tuple(out.shape) == (2,), f"应返回形状 (2,) 的每条序列 log 概率，得到 {tuple(out.shape)}"
    logp = F.log_softmax(logits, -1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    ref = (logp * mask).sum(-1) / mask.sum(-1).clamp(min=1)
    close(out, ref, tol=1e-5, msg="与 (logp * mask).sum(-1) / mask.sum(-1) 不一致")

    labels2 = labels.clone(); labels2[:, 0] = -100
    compute_logprob(logits, labels2, mask)   # 不应抛异常

check("compute_logprob 形状 / 数值 / 容错 -100", _t)'''),
    md("""### 13.6 实现 `dpo_loss`

**要求**：输入策略模型和参考模型对 chosen / rejected 的 log 概率，返回 DPO 损失。

```
ratio_w = logp_w - ref_logp_w
ratio_l = logp_l - ref_logp_l
loss = -F.logsigmoid(beta * (ratio_w - ratio_l)).mean()
```

**验证**：如果策略模型与参考模型完全一致，loss 应该等于 `log(2) ≈ 0.693`。"""),
    blank("dpo_loss", [
        "签名：dpo_loss(logp_w, logp_l, ref_logp_w, ref_logp_l, beta=0.1)",
        "四个输入都是形状 [B] 的张量",
        "ratio_w = logp_w - ref_logp_w；ratio_l = logp_l - ref_logp_l",
        "margin = beta * (ratio_w - ratio_l)",
        "return -F.logsigmoid(margin).mean()",
        "（可选）同时返回 chosen/rejected 的隐式奖励供观察",
    ], ["`F.logsigmoid`", "`Tensor.mean`"]),
    test("dpo_loss", '''def _t():
    w, l = torch.tensor([-1.5]), torch.tensor([-2.8])
    rw, rl = torch.tensor([-3.4]), torch.tensor([-3.2])

    out = dpo_loss(w, l, rw, rl, beta=0.1)
    loss = out[0] if isinstance(out, tuple) else out
    ref = -F.logsigmoid(0.1 * ((w - rw) - (l - rl))).mean()
    close(loss, ref, tol=1e-5, msg="与 -log σ(β·(Δw − Δl)) 不一致")

    out2 = dpo_loss(rw, rl, rw, rl, beta=0.1)
    loss2 = out2[0] if isinstance(out2, tuple) else out2
    close(loss2.item(), float(torch.log(torch.tensor(2.0))), tol=1e-5,
          msg="策略模型与参考模型完全一致时 margin=0，loss 应为 log(2) ≈ 0.693")

check("dpo_loss 数值 / 策略=参考时等于 log2", _t)'''),
    summary_cell(),
    part3([
        "能解释为什么减去参考模型的 log 概率",
        "β 变大时损失和梯度会怎样变化？",
        "策略=参考时 loss 是多少？验证了吗？",
        "`gather` 之前为什么要把 -100 替换掉？",
    ], "**思考题**：DPO 只能利用静态的偏好数据。如果模型在训练中生成了一条比数据里更好的回答，DPO 能利用它吗？"),
]

# ══════════════════════════════════════════════════════════════
# 14 RL
# ══════════════════════════════════════════════════════════════
c4 = [
    header(14, "强化学习 GRPO / PPO / CISPO", "第三部分 · 训练", "14_rl.html",
           "DPO 只能被动学现成数据。强化学习让模型自己生成、自己评分、自己改进。MiniMind 把四种算法统一成一个公式。"),
    goals([
        "理解「策略项 × 优势项 × 正则项」的统一视角",
        "掌握 `torch.distributions.Categorical` 采样与 log_prob",
        "掌握 `torch.clamp` 实现 PPO 裁剪",
        "能实现 GRPO 的优势归一化与策略损失",
    ]),
    md("""## 📖 背景：统一视角

```
J = E[ Φ(r_t, A_t) ]  −  β · KL(π ‖ π_ref)
      └─ 策略项 ─┘        └─ 正则项 ─┘
                ↑
          优势项 A_t：这一步比平均好多少
```

| 算法 | 策略项 Φ | 优势 A_t | 模型数 |
|---|---|---|---|
| DPO | log r_w − log r_l | 隐式 | 1（2 次前向） |
| PPO | min(r·A, clip(r)·A) | Critic / GAE | 2 |
| GRPO | min(r·A, clip(r)·A) | (R−μ)/σ | 1 |
| CISPO | clip(r)·A·logπ | (R−μ)/σ | 1 |

**GRPO 最巧妙的地方**：不需要 Critic。对同一个问题采样一组回答，
用组内奖励的均值/标准差做归一化，就得到了优势。

---

"""),
    img(
        ("../images/grpo_loss.jpg", "MiniMind GRPO 训练曲线：奖励稳定上升"),
        ("../images/ppo_loss.jpg", "MiniMind PPO 训练曲线：需要 actor + critic 双网络协同"),
    ),
    part1(),
    md("### 14.1 采样：`torch.distributions.Categorical`"),
    code("""import torch
from torch.distributions import Categorical

logits = torch.tensor([[2.0, 1.0, 0.1, 0.1]])
dist = Categorical(logits=logits)
samples = dist.sample((5,))               # 采样 5 次
print("采样结果:", samples.tolist())
print("对应 log 概率:", dist.log_prob(samples).tolist())
print("概率分布:", dist.probs.tolist())"""),
    md("""### 14.2 `log_prob` 与 `gather` 取特定动作"""),
    code("""import torch.nn.functional as F

logits = torch.randn(2, 4)                    # [B, num_actions]
actions = torch.tensor([1, 3])                # 实际采取的动作

logp = F.log_softmax(logits, dim=-1)
picked = logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
print("动作的 log 概率:", picked.tolist())"""),
    md("""### 14.3 组内归一化 → 优势"""),
    code("""rewards = torch.tensor([0.9, 0.2, 0.6, 0.1, 0.8, 0.3, 0.5, 0.4])
mu, sd = rewards.mean(), rewards.std()
advantages = (rewards - mu) / (sd + 1e-8)
print(f"μ = {mu:.3f}   σ = {sd:.3f}")
print("优势:", [round(a, 3) for a in advantages.tolist()])
print("优势均值≈0:", round(advantages.mean().item(), 6), " 标准差≈1:", round(advantages.std().item(), 4))"""),
    md("""### 14.4 PPO 裁剪：`torch.clamp`"""),
    code("""eps = 0.2
ratio = torch.tensor([0.5, 0.9, 1.0, 1.1, 1.5, 2.0])
A = 1.0                                    # 正优势

clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
obj = torch.minimum(ratio * A, clipped * A)
print("ratio  :", ratio.tolist())
print("clipped:", clipped.tolist())
print("目标   :", obj.tolist(), " <- ratio 超过 1.2 后不再增长")"""),
    md("""### 14.5 `detach` 的重要性

计算 `r = π_new / π_old` 时，分母必须 `detach`，否则梯度会走错方向。"""),
    code("""logp_new = torch.tensor([-1.0], requires_grad=True)
logp_old = torch.tensor([-1.2])

r_wrong = torch.exp(logp_new - logp_old)              # 分母带梯度
r_right = torch.exp(logp_new - logp_old.detach())     # 分母 detach

print("两种写法数值相同:", torch.allclose(r_wrong, r_right))
print("但对 logp_old 的梯度不同（正确写法里 logp_old 不应有梯度）")"""),
    md("""### 14.6 KL 正则：用 log 概率近似"""),
    code("""logp = torch.tensor([[-1.0, -2.0, -3.0]])
logp_ref = torch.tensor([[-1.2, -1.8, -3.1]])

# 无偏估计：exp(logp_ref - logp) - (logp_ref - logp) - 1
diff = logp_ref - logp
kl = torch.exp(diff) - diff - 1
print("近似 KL:", kl.tolist(), " <- 每项都 >= 0")"""),
    part2("现在实现 GRPO 的核心部分。\n\n> 💡 每个留白代码块后面都跟了一个**测试块**。先写代码，再运行测试块。"),
    testkit(),
    md("""### 14.7 实现 `compute_advantage`

**要求**：输入一组奖励 `[G]`，返回归一化后的优势 `[G]`。

```python
A = (R - R.mean()) / (R.std() + 1e-8)
```"""),
    blank("compute_advantage", [
        "签名：compute_advantage(rewards)",
        "rewards: 形状 [G] 的张量（G = 组内采样数）",
        "返回 (rewards - rewards.mean()) / (rewards.std() + 1e-8)",
    ], ["`Tensor.mean`、`Tensor.std`"]),
    test("compute_advantage", '''def _t():
    r = torch.tensor([0.9, 0.2, 0.6, 0.1, 0.8, 0.3, 0.5, 0.4])
    a = compute_advantage(r)
    assert tuple(a.shape) == tuple(r.shape), f"优势应与奖励同形状，得到 {tuple(a.shape)}"
    close(a.mean().item(), 0.0, tol=1e-5, msg="优势的均值应约为 0")
    assert abs(a.std().item() - 1.0) < 0.1, f"优势的标准差应约为 1，得到 {a.std().item():.3f}"
    assert a.max() > 0 and a.min() < 0, "奖励高于/低于平均的样本应分别得到正/负优势"

check("compute_advantage 组内归一化", _t)'''),
    md("""### 14.8 实现 `grpo_loss`

**要求**：
1. 计算概率比 `ratio = exp(logp_new - logp_old.detach())`
2. 计算 `surr1 = ratio * A`，`surr2 = clamp(ratio, 1-ε, 1+ε) * A`
3. 策略损失 `policy_loss = -min(surr1, surr2).mean()`
4. 加上 KL 正则项（可选，用上面的近似公式）
5. 返回总损失

**注意**：`logp_old` 一定要 `detach()`。"""),
    blank("grpo_loss", [
        "签名：grpo_loss(logp_new, logp_old, advantages, eps=0.2, kl_coef=0.0, logp_ref=None)",
        "ratio = torch.exp(logp_new - logp_old.detach())",
        "surr1 = ratio * advantages；surr2 = clamp(ratio, 1-eps, 1+eps) * advantages",
        "policy_loss = -torch.min(surr1, surr2).mean()",
        "若有 logp_ref，加上 kl_coef * KL（用 exp(diff) - diff - 1 近似）",
        "返回总损失",
    ], [
        "`torch.exp`、`torch.clamp`、`torch.minimum`/`torch.min`",
        "`Tensor.detach`",
    ]),
    test("grpo_loss", '''def _t():
    torch.manual_seed(0)
    logp_new = torch.randn(8, requires_grad=True)
    logp_old = torch.randn(8, requires_grad=True)
    adv = torch.randn(8)

    out = grpo_loss(logp_new, logp_old, adv)
    loss = out[0] if isinstance(out, tuple) else out
    assert torch.isfinite(loss) and loss.dim() == 0, f"应返回有限标量，得到 {loss}"
    loss.backward()
    assert logp_new.grad is not None and torch.isfinite(logp_new.grad).all(), "梯度应能传回 logp_new"
    assert logp_old.grad is None, "logp_old 不该有梯度 —— 计算 ratio 时忘记 .detach() 了"

    z = torch.zeros(8, requires_grad=True)
    out2 = grpo_loss(z, torch.zeros(8), adv)
    loss2 = out2[0] if isinstance(out2, tuple) else out2
    close(loss2.item(), (-adv.mean()).item(), tol=1e-5,
          msg="ratio=1 时 min(r·A, clip(r)·A) 应等于 A，loss 应为 -mean(A)")

check("grpo_loss 数值 / ratio=1 特例 / detach 与反传", _t)'''),
    md("""### 14.9 实现 `cispo_loss`（进阶）

CISPO 与 GRPO 的差别只在策略项：它**不裁剪 ratio 本身，而是把 ratio 当作权重**，
再乘上 `log π`。

```python
ratio_clipped = clamp(ratio, 0, eps_high)      # 注意下界是 0
policy_loss = -(ratio_clipped * advantages * logp_new).mean()
```

**思考**：这样在 ratio 被裁剪时，梯度依然能通过 `logp_new` 流动，这是 CISPO 的设计意图。"""),
    blank("cispo_loss", [
        "签名：cispo_loss(logp_new, logp_old, advantages, eps=0.2)",
        "ratio = torch.exp(logp_new - logp_old.detach())",
        "ratio_clipped = torch.clamp(ratio, 0, 1 + eps)",
        "policy_loss = -(ratio_clipped * advantages * logp_new).mean()",
        "返回 policy_loss",
    ], ["`torch.clamp`、`Tensor.mean`"]),
    test("cispo_loss", '''def _t():
    torch.manual_seed(0)
    logp_new = torch.randn(8, requires_grad=True)
    logp_old = torch.randn(8)
    adv = torch.randn(8)

    out = cispo_loss(logp_new, logp_old, adv, eps=0.2)
    loss = out[0] if isinstance(out, tuple) else out
    assert torch.isfinite(loss) and loss.dim() == 0, f"应返回有限标量，得到 {loss}"
    loss.backward()
    assert logp_new.grad is not None and torch.isfinite(logp_new.grad).all(), "梯度应能传回 logp_new"

    big = torch.full((4,), 5.0, requires_grad=True)      # ratio = e^5，远超 1+eps
    out2 = cispo_loss(big, torch.zeros(4), torch.ones(4), eps=0.2)
    loss2 = out2[0] if isinstance(out2, tuple) else out2
    loss2.backward()
    assert big.grad.abs().sum() > 0, \\
        "ratio 被裁剪后梯度消失了 —— CISPO 的设计意图正是让梯度继续通过 logp_new 流动"

check("cispo_loss 数值 / 裁剪后仍有梯度", _t)'''),
    summary_cell(),
    part3([
        "能用统一公式说出四种算法的区别",
        "GRPO 为什么不需要 Critic？",
        "PPO 的 clip 在 ratio 超过上界后做了什么？",
        "`logp_old.detach()` 忘了写会怎样？",
        "优势归一化后均值≈0、标准差≈1，验证了吗？",
    ], "**思考题**：如果一组回答的奖励全都相同（σ=0），优势会变成什么？这时 GRPO 还有梯度吗？"),
]

# ══════════════════════════════════════════════════════════════
# 15 Agentic RL
# ══════════════════════════════════════════════════════════════
c5 = [
    header(15, "Agentic RL 智能体强化学习", "第三部分 · 训练", "15_agent_rl.html",
           "最后一章。前面模型只会「一次说完」，现在要让它学会「多轮行动」：调工具、看结果、再决定下一步。"),
    goals([
        "理解多轮轨迹与单轮回答的区别",
        "掌握 `json`、`re` 解析工具调用与答案",
        "理解延迟奖励与多组件奖励设计",
        "能实现工具调用解析与奖励计算",
    ]),
    md("""## 📖 背景：奖励不只来自答案

```python
R(τ) = R_answer + R_tool + R_format − R_unfinished
        │          │         │          └ 没给出最终答案要扣分
        │          │         └ 格式错误要扣分
        │          └ 工具调用是否合法、是否有用
        └ 最终答案对不对（可能延迟到最后才结算）
```

与 GRPO 的区别：GRPO 优化一条回答，Agentic RL 优化**一整条轨迹**——
包括「想 → 调工具 → 看结果 → 再想 → 答」的每一步。

**延迟奖励**：调工具这一动作本身可能没有对错，它的价值取决于最终能不能解出题。
所以奖励要等到轨迹结束才结算，再沿着整条轨迹回传。

---

"""),
    img(
        ("../images/rl-structure.jpg", "MiniMind 的 RL 结构：训练侧、轨迹侧、rollout 侧解耦"),
        ("../images/agent_rl_loss.jpg", "Agentic RL 训练曲线"),
        ("../images/agent_webui.jpg", "基于 Agentic RL 训练结果的 WebUI：支持思考展示、工具选择与多轮 Tool Use"),
    ),
    part1(),
    md("### 15.1 解析工具调用：`json`"),
    code("""import json

tool_call_text = '''<tool_call>
{"name": "calculator", "arguments": {"expression": "3*12-5"}}
</tool_call>'''

# 提取标签内的 JSON
start = tool_call_text.find("<tool_call>") + len("<tool_call>")
end = tool_call_text.find("</tool_call>")
payload = json.loads(tool_call_text[start:end].strip())
print("工具名  :", payload["name"])
print("参数    :", payload["arguments"])
print("表达式  :", payload["arguments"]["expression"])"""),
    md("""### 15.2 正则提取：`re`"""),
    code("""import re

text = "答案是 <answer>31</answer>，用了 <tool_call>calculator</tool_call> 工具。"

ans = re.search(r"<answer>(.*?)</answer>", text)
tools = re.findall(r"<tool_call>(.*?)</tool_call>", text)
print("答案:", ans.group(1) if ans else None)
print("工具:", tools)"""),
    md("""### 15.3 校验数学答案"""),
    code("""def check_math(pred: str, gt: str) -> bool:
    def norm(s):
        s = str(s).strip().replace(",", "")
        try:
            return round(float(s), 4)
        except ValueError:
            return s
    return norm(pred) == norm(gt)

print(check_math("31", "31"))          # True
print(check_math("31.0", "31"))        # True（数值归一化）
print(check_math("32", "31"))          # False"""),
    md("""### 15.4 安全求值一个简单表达式\n\n真实项目里应该用受限的解析器，这里演示用 `eval` + 白名单的思路。"""),
    code("""import re

def safe_eval(expr: str):
    # 只允许数字和 + - * / ( ) . 空格
    if not re.fullmatch(r"[0-9+\\-*/(). ]+", expr):
        return None
    try:
        return eval(expr, {"__builtins__": {}}, {})
    except Exception:
        return None

print("3*12-5 =", safe_eval("3*12-5"))
print("非法输入:", safe_eval("__import__('os').system('ls')"))"""),
    part2("现在实现智能体的解析与奖励计算。\n\n> 💡 每个留白代码块后面都跟了一个**测试块**。先写代码，再运行测试块。"),
    testkit(),
    md("""### 15.5 实现 `parse_tool_call`

**要求**：从模型输出文本里提取第一个工具调用，返回 `(name, arguments)`；
如果没有或 JSON 非法，返回 `(None, None)`。

**提示**：先找 `<tool_call>...</tool_call>`，再 `json.loads`，用 `try/except` 兜住解析失败。"""),
    blank("parse_tool_call", [
        "签名：parse_tool_call(text)",
        "用正则或 find 提取 <tool_call> 与 </tool_call> 之间的内容",
        "json.loads 解析，取出 name 和 arguments",
        "解析失败或无标签时返回 (None, None)",
    ], ["`re.search`", "`json.loads`", "`try / except json.JSONDecodeError`"]),
    test("parse_tool_call", '''def _t():
    text = \'<tool_call>{"name": "calculator", "arguments": {"expression": "3*12-5"}}</tool_call>\'
    name, args = parse_tool_call(text)
    assert name == "calculator", f"应解析出工具名 calculator，得到 {name}"
    assert args == {"expression": "3*12-5"}, f"参数解析错误，得到 {args}"

    n2, a2 = parse_tool_call("直接回答，没有工具")
    assert n2 is None and a2 is None, f"没有工具调用时应返回 (None, None)，得到 {(n2, a2)}"

    n3, _ = parse_tool_call("<tool_call>{这不是合法 JSON}</tool_call>")
    assert n3 is None, "JSON 非法时应返回 None，而不是抛异常"

check("parse_tool_call 正常 / 缺失 / 非法 JSON", _t)'''),
    md("""### 15.6 实现 `compute_reward`

**要求**：给定一条轨迹的文本和标准答案，返回总奖励。

```python
R = R_answer + R_tool + R_format − R_unfinished
```

规则：
- `R_answer`：`<answer>` 里的答案与 `gt_answer` 数值相等给 **1.0**，否则 0
- `R_tool`：解析出至少一个**合法**的 `<tool_call>` 给 **0.3**，否则 0
- `R_format`：有 `<answer>` 标签、且工具调用 JSON 合法给 **0.1**，否则 0
- `R_unfinished`：整条轨迹里没有 `<answer>` 扣 **0.5**

返回 `(total, detail_dict)`，明细 dict 方便调试。"""),
    blank("compute_reward", [
        "签名：compute_reward(text, gt_answer)",
        "用上面写好的 parse_tool_call 判断有没有合法工具调用",
        "用 re.search 找 <answer>...</answer>，用 check_math 判对错",
        "R_answer 正确 +1.0；R_tool 有合法工具调用 +0.3",
        "R_format 有 <answer> 且 JSON 合法 +0.1；R_unfinished 没答案 -0.5",
        "返回 (total, detail_dict)",
    ], ["上面写好的 `parse_tool_call`", "演示过的 `check_math`（含数值归一化）", "`re.search`"]),
    test("compute_reward", '''def _t():
    good = (\'<tool_call>{"name": "calculator", "arguments": {"expression": "3*12-5"}}</tool_call>\\n\'
            "<answer>31</answer>")
    out = compute_reward(good, "31")
    assert isinstance(out, tuple) and len(out) == 2, "应返回 (总奖励, 明细 dict)"
    total, detail = out
    assert isinstance(detail, dict), "第二个返回值应该是明细 dict"
    close(total, 1.0 + 0.3 + 0.1, tol=1e-6,
          msg="完整且正确的轨迹应得 1.0 + 0.3 + 0.1 = 1.4")

    total2, _ = compute_reward("我不知道", "31")
    assert total2 < 0, f"没给出 <answer> 应扣 R_unfinished，得到 {total2}"

    total3, _ = compute_reward("<answer>32</answer>", "31")
    assert 0 < total3 < 1.0, f"答案错误时不该拿到 R_answer，得到 {total3}"

    total4, _ = compute_reward("<answer>1,000</answer>", "1000")
    assert total4 >= 1.0, f"1,000 与 1000 数值相等应判对，得到 {total4}"

check("compute_reward 各分量", _t)'''),
    md("""### 15.7 实现一个简化的单轮 `grpo_step`（进阶）

真实的 Agentic RL 需要与环境交互，比较复杂。这里先用**离线轨迹 + 规则奖励**
把「采样 → 打分 → 归一化 → 更新」这条链路跑通。

**要求**：
1. 对同一个 prompt 用当前策略采样 G 条回答
2. 用 `compute_reward` 给每条打分
3. 用 `compute_advantage`（第 14 章）归一化
4. 用 `grpo_loss`（第 14 章）算损失并反向

**说明**：如果模型没训练好、采不出合法格式，可以先用固定的假 logits 模拟，
重点是把流程走通。

> ⚠️ 这一节是**进阶内容，没有自动测试**——真实的多轮 rollout 需要与环境交互，
> 没法在这里自动验证。请自己打印每条轨迹的奖励、优势与损失，观察它们是否合理。"""),
    blank("grpo_step", [
        "签名：grpo_step(model, tokenizer, prompt, G=4, ...)",
        "采样 G 条回答（model.generate 或 Categorical 采样）",
        "对每条算 compute_reward，得到 rewards [G]",
        "advantages = compute_advantage(rewards)",
        "算 logp_new / logp_old，调用 grpo_loss，反向 + 更新",
    ], [
        "第 14 章的 `compute_advantage` 和 `grpo_loss`",
        "`torch.distributions.Categorical`",
        "`optimizer.step()` / `zero_grad()`",
    ]),
    summary_cell(),
    md("""## 🎓 结语

到这里，你已经把 MiniMind 的**全部零件**亲手实现了一遍：

- **模型**：RMSNorm → RoPE → 注意力/GQA → SwiGLU → MoE → Block → 完整模型
- **训练**：预训练 → SFT → LoRA → 蒸馏 → DPO → GRPO/PPO/CISPO → Agentic RL

**下一步建议**：
1. 把这 15 章所有留白代码块都自己写一遍；
2. 对照官方 `model/model_minimind.py` 逐行 diff，看差异在哪；
3. 改一个超参（层数 / head 数 / 词表大小）重新训练，观察 loss 与生成质量的变化；
4. 把小模型部署起来（`web_demo.py`），亲自和它对话。"""),
    part3([
        "能说清多轮轨迹与单轮回答在奖励设计上的区别",
        "「延迟奖励」为什么必要？",
        "`compute_reward` 的四个分量各自防止了什么坏行为？",
        "整个项目的 15 个知识点，你能独立讲出来几个？",
    ], "**思考题**：如果只保留 `R_answer` 而不要 `R_format`，模型会学到什么奇怪的输出格式？"),
]

for name, cells in [
    ("11_lora.ipynb", c),
    ("12_distill.ipynb", c2),
    ("13_dpo.ipynb", c3),
    ("14_rl.ipynb", c4),
    ("15_agent_rl.ipynb", c5),
]:
    save(name, cells)
