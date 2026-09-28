"""Notebook 生成公共工具：把结构化内容写成标准 .ipynb。

设计约定（每个章节的 Notebook 都遵循）：
  第一部分 · 基础函数演示   —— 可运行代码，先熟悉本章要用到的 Python / PyTorch API
  第二部分 · 动手实现       —— 只给文字提示 + 空白代码块，学习者自己从零写
  第三部分 · 自测与思考     —— 检查清单
"""
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks")
os.makedirs(OUT, exist_ok=True)

KERNEL = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {
        "name": "python",
        "version": "3.12",
        "mimetype": "text/x-python",
        "file_extension": ".py",
        "pygments_lexer": "ipython3",
        "nbconvert_exporter": "python",
    },
}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(keepends=True)}


def img(*pairs):
    """插入官方配图。用法：img(("../images/x.jpg", "说明"), ...)

    路径相对 notebooks/ 目录，因此在 JupyterLab 里可以直接渲染。
    """
    s = "## 🖼 官方配图\n\n"
    for rel, cap in pairs:
        s += f"![{cap}]({rel})\n\n*{cap}*\n\n"
    s += "---\n"
    return md(s)


TESTKIT = r'''# ════════════════════ 自测工具 ════════════════════
# 每个留白代码块后面都跟一个测试块。先在上面写代码，再运行测试块。
# 没写之前运行也不会报错，只会提示「未实现」。
import torch
import torch.nn.functional as F

_PASS, _FAIL, _SKIP = [], [], []

def check(name, fn):
    """运行一个测试。捕获异常并打印结果，不会中断后面的测试。"""
    try:
        fn()
        print(f"✅ 通过   {name}")
        _PASS.append(name)
    except NameError as e:
        print(f"⏸  未实现 {name} —— {e}")
        _SKIP.append(name)
    except AssertionError as e:
        print(f"❌ 失败   {name} —— {e}")
        _FAIL.append(name)
    except Exception as e:
        print(f"❌ 出错   {name} —— {type(e).__name__}: {e}")
        _FAIL.append(name)

def close(a, b, tol=1e-4, msg=""):
    """断言两个张量 / 数字足够接近"""
    a, b = torch.as_tensor(a).float().reshape(-1), torch.as_tensor(b).float().reshape(-1)
    assert a.numel() == b.numel(), f"元素个数不一致 {a.numel()} vs {b.numel()} {msg}"
    assert torch.allclose(a, b, atol=tol), f"数值不接近（容差 {tol}）{msg}"

def same(a, b, msg=""):
    """断言两个张量完全相同"""
    a, b = torch.as_tensor(a), torch.as_tensor(b)
    assert a.shape == b.shape, f"形状不一致 {tuple(a.shape)} vs {tuple(b.shape)} {msg}"
    assert torch.equal(a, b), f"内容不相同 {msg}"

def summary():
    print()
    print(f"通过 {len(_PASS)} · 失败 {len(_FAIL)} · 未实现 {len(_SKIP)}")
    if _FAIL: print("需要修的：", "、".join(_FAIL))
    if _SKIP: print("还没写的：", "、".join(_SKIP))
    if _PASS and not _FAIL and not _SKIP:
        print("本章全部通过 🎉")

print("自测工具已就绪 —— 每写完一个留白，就运行它下面的测试块。")'''


def testkit():
    """放在第二部分开头：定义 check / close / same / summary。"""
    return code(TESTKIT)


def test(title, body):
    """生成一个测试单元，紧跟在对应该留白的代码块之后。"""
    return code(f"# ════════════ 测试：{title} ════════════\n# 先在上面把代码写出来，再运行这一格。\n{body}")


def summary_cell():
    return code("summary()")


def deps(intro, code_str):
    """本章依赖的、来自前面章节的参考实现（保证本章可以独立测试）。"""
    return md(intro) if code_str is None else [md(intro), code(code_str)]


def blank(title, requirements, hints=None, extra=None):
    """生成一个“留白”代码块：只有注释形式的提示，学习者自己补全。"""
    lines = [f"# {'='*8} 你的实现：{title} {'='*8}", "#", "# 【要求】"]
    for r in requirements:
        lines.append(f"#   - {r}")
    if hints:
        lines.append("#")
        lines.append("# 【可用到的函数 / 方法】")
        for h in hints:
            lines.append(f"#   - {h}")
    if extra:
        lines.append("#")
        lines.append(f"# 【补充】{extra}")
    lines += ["#", "# 提示：先想清楚输入输出的形状，再动手。", "#", "# ↓↓↓ 在这里写你的代码 ↓↓↓", "", "", ""]
    return code("\n".join(lines))


def header(n, title, part, web, lead):    return md(
        f"""# 第 {n} 章 · {title}

> 📄 配套动画网页：`../{web}`　（先用浏览器打开，把原理动画看一遍再来写代码）

**所属部分**：{part}

{lead}

---
"""
    )


def goals(items, prerequisites=None):
    s = "## 🎯 本章目标\n\n"
    for i in items:
        s += f"- {i}\n"
    if prerequisites:
        s += "\n**前置知识**：" + "、".join(prerequisites) + "\n"
    s += "\n---\n"
    return md(s)


def part1(intro=None):
    s = "# 第一部分 · 基础函数演示\n\n> 这一部分**可以直接运行**。先把本章要用到的 Python / PyTorch 函数和方法摸熟，后面自己写的时候才不会卡在 API 上。\n"
    if intro:
        s += f"\n{intro}\n"
    return md(s)


def part2(intro=None):
    s = "# 第二部分 · 动手实现（只给提示，留白给你）\n\n> 下面每一节都**没有答案**。请根据文字提示 + 第一部分演示过的函数，自己从零写出来。\n> 写不出来时，先回去看第一部分的演示，再对照网页里的动画想清楚数据形状。\n"
    if intro:
        s += f"\n{intro}\n"
    return md(s)


def part3(checks, extra=None):
    s = "# 第三部分 · 自测与思考\n\n"
    for c in checks:
        s += f"- [ ] {c}\n"
    if extra:
        s += f"\n{extra}\n"
    return md(s)


def save(name, cells):
    nb = {"cells": cells, "metadata": KERNEL, "nbformat": 4, "nbformat_minor": 5}
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("written:", path, f"({len(cells)} cells)")
