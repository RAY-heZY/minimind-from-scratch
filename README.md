# MiniMind 从零学习项目

**🔗 在线看动画网页：<https://RAY-heZY.github.io/minimind-from-scratch/>**

> 把 MiniMind 拆成 **15 个知识点**，每个知识点独立成章。
> 每章 = **一个原理动画网页** + **一个引导式 Jupyter Notebook**。
> 模型部分**只给文字提示**，代码由你自己从零写出来。

---

## 这是什么

[MiniMind](https://github.com/jingyaogong/minimind) 是一个用极低成本（约 3 元、2 小时）从零训练
64M 参数小语言模型的开源项目。它最大的教学价值是——**短到你能逐行读懂**。

本项目把它拆开、摊平，变成一条你可以自己走完的学习路径：

- **看**：每章一个 HTML 网页，用交互动画讲清原理（拖滑块、点播放，看得见数据怎么流动）。
- **学**：每章一个 Notebook，**第一部分**带你熟悉本章要用到的 Python / PyTorch 函数（可直接运行）。
- **写**：Notebook **第二部分**只给文字提示，**留白代码块**让你自己实现。
- **验**：Notebook **第三部分**是自测清单，用来确认你真的写对了。

> ⚠️ 模型部分（第 3–8 章）**不给答案**。这是刻意的——自己写一遍，比读十遍强。

---

## 课程地图（15 章）

### 第一部分 · 基础
| 章 | 标题 | 配套网页 / Notebook |
|---|---|---|
| 01 | 项目总览与 uv 环境 | `01_overview.html` · `notebooks/01_overview.ipynb` |
| 02 | 分词器与数据流水线 | `02_tokenizer.html` · `notebooks/02_tokenizer.ipynb` |

### 第二部分 · 模型
| 章 | 标题 | 知识点 |
|---|---|---|
| 03 | RMSNorm 均方根归一化 | 归一化、尺度不变性、`rsqrt` |
| 04 | RoPE 旋转位置编码 | 相对位置、旋转、YaRN 外推 |
| 05 | 因果自注意力与 GQA / KV Cache | 缩放点积、因果掩码、分组查询 |
| 06 | SwiGLU 前馈网络 | 门控、SiLU、8/3·d |
| 07 | MoE 混合专家 | 路由、Top-K、负载均衡 |
| 08 | Transformer Block 与模型组装 | Pre-Norm 残差、权重共享、交叉熵 |

### 第三部分 · 训练
| 章 | 标题 | 知识点 |
|---|---|---|
| 09 | 预训练 Pretrain | 下一词预测、余弦调度、梯度累积 |
| 10 | 指令微调 SFT | chat template、loss mask |
| 11 | LoRA 低秩适配 | 冻结底座、低秩增量、权重合并 |
| 12 | 知识蒸馏 | 黑盒 / 白盒、温度、KL |
| 13 | DPO 直接偏好优化 | chosen/rejected、log-odds |
| 14 | 强化学习 GRPO / PPO / CISPO | 策略项 × 优势项 × 正则项 |
| 15 | Agentic RL 智能体强化学习 | 多轮轨迹、延迟奖励、工具调用 |

---

## 快速开始

环境用 **uv** 管理，依赖声明在 `pyproject.toml`，精确版本锁在 `uv.lock`。

```bash
cd minimind-from-scratch

# 1. 创建隔离环境并安装依赖
uv sync

# 2. 启动 JupyterLab，打开 notebooks/ 下任意一章
uv run jupyter lab

# 3. 只想看动画网页？用浏览器打开 index.html 即可
#    （也可以起一个本地服务：uv run python -m http.server 8000）
```

### 在线访问

网页已托管在 GitHub Pages：**https://RAY-heZY.github.io/minimind-from-scratch/**

> 每页右上角的「配套 Notebook ↗」会跳到 GitHub 上对应的 `.ipynb`，
> 可以直接在线阅读（GitHub 能渲染 Notebook），或 clone 下来本地跑。
> 改仓库地址后，用 `python tools/set_repo.py <owner>/<repo>` 可一次性更新所有链接。

### 关于算力

- **原理演示全部使用小张量，CPU 就能跑通**（第 1–2 部分）。
- 真正训练 64M 模型建议单卡 3090 / 4090。
- 训练章节（09–15）的 Notebook 以「讲清算法 + 留白实现」为主，
  能不能真跑取决于你的硬件，但**读代码 + 写代码**不受影响。

---

## 目录结构

```
minimind-from-scratch/          ← 仓库根目录 = 站点根目录（GitHub Pages 直接托管）
├── index.html                # 课程主页
├── 01_overview.html          # 15 个原理动画网页（都在根目录）
├── …
├── 15_agent_rl.html
├── assets/
│   ├── style.css             # 浅色主题样式
│   ├── common.js             # 动画播放器 / SVG 助手
│   └── chapters.js           # 章节清单 + GitHub 仓库坐标（侧边栏 & 翻页）
├── .nojekyll                 # 关闭 Jekyll，让 Pages 原样发布
├── pyproject.toml            # uv 依赖声明
├── uv.lock                   # 锁定的精确版本
├── README.md
├── data/
│   └── tokenizer.json        # MiniMind 官方分词器（词表 6400）
├── images/                   # MiniMind 官方配图（结构图 / 损失曲线 / RL 结构）
├── notebooks/                # 15 个引导式 Notebook
│   ├── 01_overview.ipynb
│   ├── …
│   └── 15_agent_rl.ipynb
└── tools/                    # 构建与校验脚本
    ├── nbcommon.py           # Notebook 生成工具（含 testkit / test 单元）
    ├── build_notebooks_*.py  # 生成各章 Notebook
    ├── validate_notebooks.py # 校验 JSON / nbformat / 代码语法
    ├── run_notebooks.py      # 顺序执行所有非留白单元
    ├── verify_tests.py       # 用「答案实现」反向校验测试本身
    ├── set_repo.py           # 一次性把所有页面链接指向你的 GitHub 仓库
    └── check_html_js.mjs     # 校验网页内联 JS 语法
```

---

## 每章 Notebook 的结构

所有章节都遵循同一个三段式结构：

```
# 第 NN 章 · 标题
> 📄 配套动画网页：../NN_xxx.html

## 🎯 本章目标

## 🖼 官方配图
   MiniMind 官方架构图 / 损失曲线，同样嵌进 Notebook，方便对照

# 第一部分 · 基础函数演示
   可直接运行的代码：先熟悉本章要用的 Python / PyTorch API

# 第二部分 · 动手实现（只给提示，留白给你）
   自测工具 check() / close() / same() / summary()
   每节：文字提示 + 【要求】+【可用函数】+ 空白代码块 + **测试块**

# 第三部分 · 自测与思考
   检查清单 + 思考题
```

### 写完了怎么知道对不对？——每个留白后面都有测试块

第二部分共 **30 个测试块**，紧跟在对应该留白的后面。直接运行它，会得到三种结果之一：

```
✅ 通过   RMSNorm 形状 / 数值 / 尺度不变性
❌ 失败   RMSNorm 形状 / 数值 / 尺度不变性 —— 输出形状应与输入一致，得到 (2, 5, 8)
⏸  未实现 RMSNorm 形状 / 数值 / 尺度不变性 —— name 'RMSNorm' is not defined
```

- **还没写就运行也不会报错**，只会提示「未实现」，不会打断后面的单元格。
- 每个测试都带**具体断言信息**（形状不对、数值不接近、性质不成立…），直接告诉你错在哪。
- 测试只检查**行为契约**（形状、性质、与公式对照），不限定你的写法。
- 每章末尾有 `summary()` 汇总：通过几个、失败几个、还没写几个。

> 测试的设计原则：**只测「对不对」，不测「像不像我写的」。** 只要你的实现行为正确，换个写法一样能过。


**建议的学习顺序**：先看网页动画 → 再跑第一部分 → 然后**合上参考、自己写第二部分** → 最后对照第三部分自测。

---

## 校验与重新生成

```bash
cd tools

# 校验所有 Notebook（JSON 合法性、nbformat 结构、代码语法）
uv run python validate_notebooks.py

# 顺序执行所有「非留白」单元，确认它们真能跑通
uv run python run_notebooks.py

# 校验所有网页的内联 JS 语法（需要 node）
node check_html_js.mjs

# 重新生成全部 Notebook（改内容后运行）
uv run python build_notebooks_a.py   # 第 01–05 章
uv run python build_notebooks_b.py   # 第 06–10 章
uv run python build_notebooks_c.py   # 第 11–15 章
```

> **30 个测试块**都经过「答案实现」反向校验：用一份正确的参考实现跑一遍所有测试，
> 确认每个测试都能通过（否则说明测试本身写错了）。测试只断言行为，不限定写法。

---

## 学完之后

1. 把这 15 章所有留白代码块都自己写一遍；
2. 对照官方 `model/model_minimind.py` 逐行 diff，看差异在哪；
3. 改一个超参（层数 / head 数 / 词表大小）重新训练，观察 loss 与生成质量的变化；
4. 把小模型部署起来（官方 `web_demo.py`），亲自和它对话。

---

## 致谢与许可

- 模型架构、官方配图与分词器来源：[jingyaogong/minimind](https://github.com/jingyaogong/minimind)，
  以 **Apache License 2.0** 授权。本仓库内的 `images/` 与 `data/tokenizer.json` 即来自该项目，
  遵循同一许可，版权归原作者所有。
- 本项目的网页、Notebook 与工具链为教学用途重新组织编写，同样以 **Apache License 2.0** 发布。
- 上游许可证全文见 <https://github.com/jingyaogong/minimind/blob/master/LICENSE>。
