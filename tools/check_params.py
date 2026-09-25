#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""参数有效性门禁 (A2)

断言：课件里提到的每个命令行参数 / 生成参数都**真实存在**。

真值集来源（全部实测，不靠记忆）：
  1) `transformers --help` 及其子命令的 --help（CLI 选项）
  2) GenerationConfig 的字段名（生成参数）
  3) PreTrainedModel.generate / from_pretrained 的形参名
  4) config 的已知字段（模型配置项）

用法:
    python3 tools/check_params.py [--quiet] [--lesson <dir|id>] [--dump]
"""
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from common import R

TRUTH_CACHE = os.path.join(C.DATA, "truth_params.json")

FLAG_RE = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]{2,})(?![\w-])")
# 只检查"看起来像真参数"的 token；纯 CSS/类名等由 allowlist 兜底


def build_truth():
    """实测构建真值集。"""
    truth = {"cli": set(), "gen": set(), "misc": set(), "source": {}}
    py = sys.executable
    venv_bin = os.path.join(os.path.dirname(C.ROOT), ".venv", "bin")
    venv_py = os.path.join(venv_bin, "python")
    if os.path.isfile(venv_py):
        py = venv_py
    cli = os.path.join(venv_bin, "transformers")
    if not os.path.isfile(cli):
        cli = None

    # 1) CLI --help
    def run_help(args):
        try:
            p = subprocess.run(args, capture_output=True, text=True, timeout=180)
            return (p.stdout or "") + (p.stderr or "")
        except Exception:
            return ""

    top = run_help([cli, "--help"]) if cli else ""
    for m in re.finditer(r"^\s*(?:-[a-zA-Z],\s*)?(--[a-z0-9][a-z0-9-]*)", top, re.M):
        truth["cli"].add(m.group(1))
    subs = re.findall(r"^\s{2,}([a-z][a-z0-9-]{2,})\s{2,}", top, re.M)
    for sub in sorted(set(subs)):
        h = run_help([cli, sub, "--help"]) if cli else ""
        for m in re.finditer(r"^\s*(?:-[a-zA-Z],\s*)?(--[a-z0-9][a-z0-9-]*)", h, re.M):
            truth["cli"].add(m.group(1))
        truth["cli"].add(sub)
    truth["source"]["cli"] = (f"`transformers --help` + {len(set(subs))} 个子命令的 --help"
                              f"（{len(truth['cli'])} 项）")

    # 2) 生成参数 / 配置字段（用 venv 里的 transformers 实测内省）
    probe = r'''
import json, warnings, inspect
warnings.filterwarnings("ignore")
out = {}
from transformers.generation.configuration_utils import GenerationConfig
g = GenerationConfig()
out["gen"] = sorted([k for k in g.to_dict().keys()])
from transformers.generation.utils import GenerationMixin
out["generate"] = sorted(list(inspect.signature(GenerationMixin.generate).parameters.keys()))
from transformers import Glm5NextConfig
c = Glm5NextConfig()
d = c.to_dict()
def walk(obj, acc):
    if isinstance(obj, dict):
        for k, v in obj.items():
            acc.add(str(k)); walk(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            walk(v, acc)
acc = set(); walk(d, acc)
out["cfg"] = sorted(acc)
print("@@JSON@@" + json.dumps(out))
'''
    try:
        p = subprocess.run([py, "-c", probe], capture_output=True, text=True, timeout=300,
                           cwd=os.path.dirname(C.ROOT))
        raw = (p.stdout or "")
        i = raw.find("@@JSON@@")
        if i >= 0:
            d = json.loads(raw[i + 8:].strip().splitlines()[0])
            truth["gen"] |= set(d.get("gen", []))
            truth["gen"] |= set(d.get("generate", []))
            truth["misc"] |= set(d.get("cfg", []))
            truth["source"]["gen"] = (f"GenerationConfig {len(d.get('gen',[]))} 字段 + "
                                      f"GenerationMixin.generate() {len(d.get('generate',[]))} 形参")
            truth["source"]["cfg"] = f"Glm5NextConfig {len(d.get('cfg',[]))} 字段"
        else:
            truth["source"]["gen"] = "FAILED: " + (p.stderr or "")[-200:]
    except Exception as e:
        truth["source"]["gen"] = "FAILED: " + str(e)

    # 4) CSS 自定义属性（shared/theme.css 里真实声明过的）—— 防止把 var(--x) 误判为参数
    cssdir = os.path.join(C.ROOT, "shared")
    cssvars = set()
    if os.path.isdir(cssdir):
        for fn in os.listdir(cssdir):
            if fn.endswith(".css"):
                for m in re.finditer(r"(--[a-z][a-z0-9-]*)\s*:", C.read_text(os.path.join(cssdir, fn))):
                    cssvars.add(m.group(1))
    truth["cli"] |= cssvars
    truth["source"]["css"] = f"shared/*.css 的 CSS 自定义属性（{len(cssvars)} 个）"

    # 5) 本仓库工具自身的命令行选项（tools/*.py 里 argparse 用到的）
    toolflags = set()
    tdir = os.path.join(C.ROOT, "tools")
    if os.path.isdir(tdir):
        for fn in sorted(os.listdir(tdir)):
            if fn.endswith(".py"):
                for m in re.finditer(r'"(\-\-[a-z][a-z0-9-]+)"', C.read_text(os.path.join(tdir, fn))):
                    toolflags.add(m.group(1))
    truth["cli"] |= toolflags
    truth["source"]["tools"] = f"tools/*.py 自身的选项（{len(toolflags)} 个）"

    truth["cli"] = sorted(truth["cli"])
    truth["gen"] = sorted(truth["gen"])
    truth["misc"] = sorted(truth["misc"])
    return truth


def load_truth(rebuild=False):
    if rebuild or not os.path.isfile(TRUTH_CACHE):
        t = build_truth()
        os.makedirs(C.DATA, exist_ok=True)
        json.dump(t, open(TRUTH_CACHE, "w"), ensure_ascii=False, indent=1)
        return t
    t = json.load(open(TRUTH_CACHE, encoding="utf-8"))
    for k in ("cli", "gen", "misc"):
        t[k] = set(t.get(k, []))
    return t


def main(argv):
    quiet = "--quiet" in argv
    only = argv[argv.index("--lesson") + 1] if "--lesson" in argv else None
    if "--dump" in argv:
        t = load_truth(rebuild=True)
        print(json.dumps({k: (v if k != "source" else v) for k, v in t.items()},
                         ensure_ascii=False, indent=1)[:3000])
        return 0

    t = load_truth()
    known = set(t["cli"]) | set(t["gen"]) | set(t["misc"])
    if not quiet:
        for k, v in t.get("source", {}).items():
            print(f"  真值来源[{k}]: {v}")
        print(f"  真值集规模: CLI {len(t['cli'])} / 生成 {len(t['gen'])} / 配置 {len(t['misc'])}")

    lessons = C.existing_lessons()
    if only:
        lessons = [(i, d) for i, d in lessons
                   if os.path.abspath(d) == os.path.abspath(only) or i == only]

    bad = []
    nref = 0
    for lid, d in lessons:
        for fn in ("source.md", "lesson.js", "README.md"):
            p = os.path.join(d, fn)
            if not os.path.isfile(p):
                continue
            for ln_no, ln in enumerate(C.read_text(p).split("\n"), 1):
                for m in FLAG_RE.finditer(ln):
                    tok = m.group(1)
                    nref += 1
                    if tok not in known:
                        bad.append((lid, fn, ln_no, tok))

    if not quiet:
        print(f"扫描到 --flag 引用: {nref} 处")
    if bad:
        R.bad(f"{len(bad)} 处引用了不存在的参数：")
        seen = set()
        for lid, fn, ln, tok in bad:
            if tok in seen:
                continue
            seen.add(tok)
            print(f"    {lid} {fn}:{ln}  {tok}")
        return 1
    R.ok(f"A2 通过：{nref} 处参数引用全部真实存在")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
