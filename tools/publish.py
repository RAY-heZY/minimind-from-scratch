"""一键把本项目发布到 GitHub，并开启 GitHub Pages 托管。

用法
----
    python tools/publish.py                     # 用 assets/chapters.js 里已配置的 owner/repo
    python tools/publish.py RAY-heZY/mm-course  # 指定仓库
    python tools/publish.py --private           # 建私有仓库（注意：私有仓库的 Pages 需要付费版）
    python tools/publish.py --no-pages          # 只推代码，不开 Pages

前提
----
1. 已安装 gh CLI
2. 已登录：`gh auth login`（需要 repo + workflow 权限；开 Pages 需要 repo 权限）

脚本会做四件事
--------------
1. 检查 gh 登录状态
2. 仓库不存在则创建（public），已存在则直接推送
3. push 当前分支到 origin
4. 开启 GitHub Pages：source = 当前分支 / 根目录，并打印站点地址
"""
import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, check=True, capture=True):
    r = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=capture)
    if check and r.returncode != 0:
        msg = (r.stderr or r.stdout or "").strip()
        raise SystemExit(f"命令失败：{' '.join(cmd)}\n{msg}")
    return r


def gh_ok():
    r = subprocess.run(["gh", "auth", "status"], text=True, capture_output=True)
    return r.returncode == 0


def repo_from_config():
    path = os.path.join(ROOT, "assets", "chapters.js")
    src = open(path, encoding="utf-8").read()
    m = re.search(r'const REPO = "([^"]+)"', src)
    return m.group(1) if m else None


def branch_name():
    r = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, text=True, capture_output=True)
    return r.stdout.strip() or "main"


def repo_exists(full):
    r = subprocess.run(["gh", "repo", "view", full], text=True, capture_output=True)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", nargs="?", help="owner/repo，省略则读 assets/chapters.js")
    ap.add_argument("--private", action="store_true", help="创建私有仓库")
    ap.add_argument("--no-pages", action="store_true", help="不开启 GitHub Pages")
    args = ap.parse_args()

    if not gh_ok():
        raise SystemExit("gh 未登录。请先运行：gh auth login")

    full = args.repo or repo_from_config()
    if not full or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", full):
        raise SystemExit(f"仓库名无效：{full!r}。用法：python tools/publish.py <owner>/<repo>")

    owner, name = full.split("/", 1)
    branch = branch_name()
    print(f"仓库   : {full}")
    print(f"分支   : {branch}")
    print(f"可见性 : {'private' if args.private else 'public'}")

    # 1) 确保仓库存在
    if repo_exists(full):
        print("→ 仓库已存在，跳过创建")
    else:
        print("→ 创建仓库…")
        vis = "--private" if args.private else "--public"
        run(["gh", "repo", "create", full, vis, "--source=.", "--remote=origin", "--push"])
        print("→ 已创建并推送")

    # 2) 确保 remote 正确并推送
    remotes = run(["git", "remote"]).stdout.split()
    if "origin" not in remotes:
        run(["git", "remote", "add", "origin", f"https://github.com/{full}.git"])
    else:
        run(["git", "remote", "set-url", "origin", f"https://github.com/{full}.git"])

    print("→ 推送…")
    run(["git", "push", "-u", "origin", branch])

    # 3) 开启 Pages
    if args.no_pages:
        print("→ 已跳过 Pages（--no-pages）")
    else:
        print("→ 开启 GitHub Pages（source = {branch} / 根目录）…")
        r = subprocess.run(
            ["gh", "api", "-X", "POST", f"repos/{full}/pages",
             "-f", f"source[branch]={branch}", "-f", "source[path]=/"],
            cwd=ROOT, text=True, capture_output=True,
        )
        if r.returncode != 0:
            err = (r.stderr or r.stdout).strip()
            if "already" in err.lower() or "409" in err:
                print("   Pages 之前已开启，跳过")
            else:
                print("   !! 自动开启失败，请手动到 Settings → Pages 里选择"
                      f"「Deploy from a branch: {branch} / (root)」\n   {err[:300]}")
        else:
            print("   Pages 已开启")

    site = f"https://{owner}.github.io/{name}/"
    print()
    print("完成 ✅")
    print(f"  仓库     : https://github.com/{full}")
    print(f"  站点地址 : {site}   （首次部署约需 1 分钟）")
    print()
    print("如果站点地址和上面的不一致，请运行一次以同步所有页面链接：")
    print(f"  python tools/set_repo.py {full} {branch}")
    print("  然后 git commit -am 'fix repo links' && git push")


if __name__ == "__main__":
    main()
