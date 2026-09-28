"""校验生成的 Notebook：JSON 合法性、nbformat 结构、代码单元语法。"""
import json, os, glob, sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks")
files = sorted(glob.glob(os.path.join(OUT, "*.ipynb")))
bad = 0
total_code = 0
total_md = 0
total_blank = 0

for path in files:
    name = os.path.basename(path)
    try:
        nb = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        print(f"[JSON ERROR] {name}: {e}"); bad += 1; continue

    if nb.get("nbformat") != 4:
        print(f"[FORMAT ERROR] {name}: nbformat != 4"); bad += 1
    for k in ("cells", "metadata", "nbformat", "nbformat_minor"):
        if k not in nb:
            print(f"[FORMAT ERROR] {name}: missing key {k}"); bad += 1

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] not in ("markdown", "code"):
            print(f"[CELL ERROR] {name} cell {i}: unknown type"); bad += 1; continue
        src = "".join(cell["source"])
        if cell["cell_type"] == "code":
            total_code += 1
            if "你的实现" in src:
                total_blank += 1
            if "execution_count" not in cell or "outputs" not in cell:
                print(f"[CELL ERROR] {name} cell {i}: code cell missing execution_count/outputs"); bad += 1
            try:
                compile(src, f"{name}:cell{i}", "exec")
            except SyntaxError as e:
                print(f"[SYNTAX ERROR] {name} cell {i}: {e}"); bad += 1
        else:
            total_md += 1

print()
print(f"notebooks : {len(files)}")
print(f"code cells: {total_code}  (其中留白待实现 {total_blank})")
print(f"md cells  : {total_md}")
print("RESULT    :", "ALL OK ✅" if bad == 0 else f"{bad} PROBLEM(S) ❌")
sys.exit(1 if bad else 0)
