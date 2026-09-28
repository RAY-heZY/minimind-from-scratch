"""把站点里的 GitHub 链接指向你的仓库。

用法
----
    python tools/set_repo.py <owner>/<repo> [branch]

例：
    python tools/set_repo.py RAY-heZW/minimind-from-scratch
    python tools/set_repo.py RAY-heZW/minimind-from-scratch master

作用
----
1. 更新 assets/chapters.js 里的 REPO / BRANCH 常量（侧边栏的「本章 Notebook」用）
2. 重写每个页面右上角 topbar 里的导航：
   - 章节页：课程主页 + GitHub + 「配套 Notebook ↗」（指向 GitHub 上的 .ipynb）
   - index.html：开始学习 + GitHub
可重复运行（幂等）。
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def update_chapters_js(repo, branch):
    path = os.path.join(ROOT, "assets", "chapters.js")
    src = open(path, encoding="utf-8").read()
    src = re.sub(r'const REPO = "[^"]*";', f'const REPO = "{repo}";', src, count=1)
    src = re.sub(r'const BRANCH = "[^"]*";', f'const BRANCH = "{branch}";', src, count=1)
    open(path, "w", encoding="utf-8").write(src)
    print(f"  updated assets/chapters.js  ->  REPO={repo}  BRANCH={branch}")


def update_html_nav(path, repo, branch):
    name = os.path.basename(path)
    src = open(path, encoding="utf-8").read()
    repo_url = f"https://github.com/{repo}"

    if name == "index.html":
        nav = (
            '<nav>\n'
            '    <a href="01_overview.html">开始学习 →</a>\n'
            f'    <a href="{repo_url}" target="_blank" rel="noopener">GitHub ↗</a>\n'
            '  </nav>'
        )
    else:
        nb = "notebooks/" + name.replace(".html", ".ipynb")
        nb_url = f"{repo_url}/blob/{branch}/{nb}"
        nav = (
            '<nav>\n'
            '    <a href="index.html">课程主页</a>\n'
            f'    <a href="{repo_url}" target="_blank" rel="noopener">GitHub ↗</a>\n'
            f'    <a href="{nb_url}" target="_blank" rel="noopener">配套 Notebook ↗</a>\n'
            '  </nav>'
        )

    new, n = re.subn(r"<nav>.*?</nav>", nav, src, count=1, flags=re.S)
    if n == 0:
        print(f"  !! {name}: 没找到 <nav>，跳过")
        return False
    open(path, "w", encoding="utf-8").write(new)
    return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    repo = sys.argv[1].strip().strip("/")
    branch = sys.argv[2].strip() if len(sys.argv) > 2 else "main"

    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        print(f"仓库格式应为 <owner>/<repo>，收到：{repo!r}")
        sys.exit(2)

    print(f"repo = {repo}  branch = {branch}")
    update_chapters_js(repo, branch)

    htmls = sorted(f for f in os.listdir(ROOT) if f.endswith(".html"))
    done = 0
    for f in htmls:
        if update_html_nav(os.path.join(ROOT, f), repo, branch):
            done += 1
    print(f"  updated {done}/{len(htmls)} html files")
    print("done.")


if __name__ == "__main__":
    main()
