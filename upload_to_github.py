# -*- coding: utf-8 -*-
"""用GitHub Git Data API上传项目代码到仓库"""
import os, json, base64, subprocess, urllib.request

REPO = "xjcway123/multi-agent-research-pipeline"
PROJECT_DIR = r"C:\Users\15820\Documents\multi-agent-pipeline-main"
BRANCH = "main"

# 获取token
token = subprocess.check_output([r"C:\Program Files\GitHub CLI\gh.exe", "auth", "token"], text=True).strip()
headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github+json",
    "Content-Type": "application/json",
}

def api(method, path, data=None):
    url = f"https://api.github.com/{path}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"API Error {e.code}: {e.read().decode()[:300]}")
        raise

# 收集需要上传的文件
EXCLUDE_DIRS = {'.git', '__pycache__', '.venv', 'venv', 'node_modules', '.pytest_cache'}
EXCLUDE_FILES = {'.env', '.env.local', 'test_pipeline.py'}
files = []
for root, dirs, filenames in os.walk(PROJECT_DIR):
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
    for fname in filenames:
        if fname in EXCLUDE_FILES:
            continue
        fpath = os.path.join(root, fname)
        rel_path = os.path.relpath(fpath, PROJECT_DIR).replace('\\', '/')
        with open(fpath, 'rb') as f:
            content = f.read()
        files.append((rel_path, content))

print(f"待上传文件: {len(files)} 个")

# 创建blobs
tree_items = []
for rel_path, content in files:
    is_binary = any(b in content[:8000] for b in [b'\x00', b'\xff\xd8\xff', b'\x89PNG'])
    if is_binary:
        b64 = base64.b64encode(content).decode()
        blob = api("POST", f"repos/{REPO}/git/blobs", {"content": b64, "encoding": "base64"})
    else:
        text = content.decode('utf-8', errors='replace')
        blob = api("POST", f"repos/{REPO}/git/blobs", {"content": text, "encoding": "utf-8"})
    tree_items.append({"path": rel_path, "mode": "100644", "type": "blob", "sha": blob["sha"]})
    print(f"  blob: {rel_path}")

# 创建tree
tree = api("POST", f"repos/{REPO}/git/trees", {"tree": tree_items})
print(f"tree SHA: {tree['sha']}")

# 检查是否已有commit（新仓库可能没有）
try:
    ref = api("GET", f"repos/{REPO}/git/ref/heads/{BRANCH}")
    parent_sha = ref["object"]["sha"]
    print(f"已有commit: {parent_sha}")
except:
    parent_sha = None
    print("新仓库，创建初始commit")

# 创建commit
commit_data = {"message": "init: 4 Agent协作研究报告生成系统", "tree": tree["sha"]}
if parent_sha:
    commit_data["parents"] = [parent_sha]
commit = api("POST", f"repos/{REPO}/git/commits", commit_data)
print(f"commit SHA: {commit['sha']}")

# 更新或创建分支引用
if parent_sha:
    api("PATCH", f"repos/{REPO}/git/refs/heads/{BRANCH}", {"sha": commit["sha"], "force": True})
else:
    api("POST", f"repos/{REPO}/git/refs", {"ref": f"refs/heads/{BRANCH}", "sha": commit["sha"]})

print(f"\n✓ 上传完成！仓库地址: https://github.com/{REPO}")
