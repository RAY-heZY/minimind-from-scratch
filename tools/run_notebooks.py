"""顺序执行每个 Notebook 的「演示」代码单元（跳过留白单元），报告失败。"""
import json, os, glob, sys, io, traceback, contextlib

os.environ["MPLBACKEND"] = "Agg"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebooks")
os.chdir(OUT)                      # 模拟 Jupyter 的工作目录

files = sorted(glob.glob("*.ipynb"))
fails = []
for path in files:
    nb = json.load(open(path, encoding="utf-8"))
    g = {"__name__": "__main__"}
    ok = 0
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "你的实现" in src:
            continue               # 留白单元不执行
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(src, f"{path}:{i}", "exec"), g)
            ok += 1
        except Exception as e:
            fails.append((path, i, type(e).__name__, str(e).splitlines()[0][:160]))
    print(f"{path:26s} demo cells passed: {ok}")

print()
if fails:
    print("FAILURES:")
    for p, i, t, m in fails:
        print(f"  {p} cell {i}: {t}: {m}")
else:
    print("ALL DEMO CELLS EXECUTED SUCCESSFULLY ✅")
sys.exit(1 if fails else 0)
