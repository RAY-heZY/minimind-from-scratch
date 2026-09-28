"""用「答案实现」反向校验所有测试块。

为什么需要它：留白代码块后面的测试块，只有在对的实现上跑过一遍，
才能确认「测试本身是对的」。否则学习者写对了却过不了测试，比没有测试更糟。

用法
----
1. 写一个 answers.py（**放在项目目录之外**，避免把答案泄漏给学习者）：

       ANSWERS = {
           "03_rmsnorm.ipynb": r'''
               from torch import nn
               class RMSNorm(nn.Module):
                   def __init__(self, dim, eps=1e-6):
                       super().__init__()
                       self.eps = eps
                       self.weight = nn.Parameter(torch.ones(dim))
                   def forward(self, x):
                       return (self.weight * (x * torch.rsqrt(
                           x.pow(2).mean(-1, keepdim=True) + self.eps))).type_as(x)
           ''',
           "05_attention.ipynb": r'''
               ...
           ''',
       }

2. 运行：

       python verify_tests.py <notebooks_dir> <answers.py>

退出码 0 表示全部通过；非 0 会列出失败/未实现的测试名与断言信息。

判定方式
--------
逐个 notebook 执行「答案实现 → 所有非留白单元」，
最后检查自测工具维护的 `_FAIL` / `_SKIP` 两个列表：
两者都为空才算通过。
"""
import contextlib
import glob
import importlib.util
import io
import json
import os
import sys


def load_answers(path):
    spec = importlib.util.spec_from_file_location("answers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ANSWERS


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    nb_dir, answers_path = sys.argv[1], sys.argv[2]
    answers = load_answers(answers_path)

    problems = []
    for path in sorted(glob.glob(os.path.join(nb_dir, "*.ipynb"))):
        name = os.path.basename(path)
        nb = json.load(open(path, encoding="utf-8"))
        ns = {"__name__": "__main__"}

        if name not in answers:
            problems.append((name, "NO ANSWER", "answers.py 里缺少这一章的参考实现"))
            continue
        try:
            exec(compile(answers[name], f"{name}:ANSWER", "exec"), ns)
        except Exception as e:
            problems.append((name, "ANSWER ERROR", f"{type(e).__name__}: {e}"))
            continue

        for ci, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            if "你的实现" in src:          # 跳过留白单元
                continue
            is_test = "测试：" in src
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    exec(compile(src, f"{name}:{ci}", "exec"), ns)
            except Exception as e:
                problems.append((name, f"cell{ci}", f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"))
            if is_test:
                for line in buf.getvalue().splitlines():
                    if "❌" in line or "⏸" in line:
                        problems.append((name, "detail", line.strip()))

        fails, skips, passes = ns.get("_FAIL", []), ns.get("_SKIP", []), ns.get("_PASS", [])
        flag = "OK " if (not fails and not skips) else "!! "
        print(f"{flag}{name:24s} pass={len(passes)} fail={len(fails)} skip={len(skips)}")
        problems += [(name, "TEST FAIL", x) for x in fails]
        problems += [(name, "TEST SKIP", x) for x in skips]

    print()
    if problems:
        print("PROBLEMS:")
        for a, b, c in problems:
            print(f"  {a} [{b}] {c}")
        sys.exit(1)
    print("ALL TESTS PASS WITH REFERENCE IMPLEMENTATIONS ✅")


if __name__ == "__main__":
    main()
