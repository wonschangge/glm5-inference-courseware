#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""课件的公共路径与解析工具（全部门禁共用，保证口径一致）。"""
import json
import os
import re
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)                                  # courseware/
UPSTREAM = os.environ.get(
    "GLM5_UPSTREAM",
    os.path.abspath(os.path.join(ROOT, "..", "upstream-transformers", "src", "transformers")),
)
DATA = os.path.join(ROOT, "_data")
UNIVERSE_JSON = os.path.join(DATA, "universe.json")

# 必修四文件
LESSON_FILES = ("index.html", "lesson.js", "source.md", "README.md")

# 代码块的语言标记 -> 是否计入保真校验（text 视为示意代码，不校验）
FIDELITY_LANGS = {"python", "py", "mlir", "bash", "sh", "json"}


def load_universe():
    with open(UNIVERSE_JSON, encoding="utf-8") as f:
        d = json.load(f)
    return d["universe"], d.get("excluded", [])


def upstream_path(rel):
    return os.path.join(UPSTREAM, rel)


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def discover_lessons():
    """扫描所有课件目录，返回 [(lesson_id, abs_dir)]，按课号排序。"""
    sys.path.insert(0, TOOLS)
    import plan
    out = []
    for L in plan.LESSONS:
        d = os.path.join(ROOT, plan.lesson_dir(L))
        out.append((L["id"], d))
    return out


def existing_lessons():
    return [(i, d) for i, d in discover_lessons() if os.path.isdir(d)]


# --------------------------------------------------------------------------
# coverage 声明块：<!-- glm5-coverage ... -->
# --------------------------------------------------------------------------
COV_RE = re.compile(r"<!--\s*glm5-coverage\s*\n(.*?)-->", re.S)


def parse_coverage(md_text):
    m = COV_RE.search(md_text)
    if not m:
        return None
    out = []
    for ln in m.group(1).splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            out.append(ln)
    return out


# --------------------------------------------------------------------------
# 引用块：<!-- src: path --> 之后紧跟的围栏代码块
# --------------------------------------------------------------------------
SRC_RE = re.compile(r"<!--\s*src:\s*(\S+?)\s*-->")
FENCE_RE = re.compile(r"^```(\w*)\s*$")


def parse_blocks(md_text):
    """返回 [ {path, lang, lines:[str], start_line:int} ]。

    规则：`<!-- src: path -->` 标注其**之后最近的一个**围栏代码块属于 path。
    若代码块前没有 src 标注，则 path 为 None（校验时按"未声明"处理）。
    """
    lines = md_text.split("\n")
    blocks = []
    pending = None       # 最近的 src 标注（尚未被消费）
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = SRC_RE.search(ln)
        if m:
            pending = m.group(1)
            i += 1
            continue
        mf = FENCE_RE.match(ln)
        if mf:
            lang = mf.group(1)
            j = i + 1
            body = []
            while j < len(lines) and not lines[j].startswith("```"):
                body.append(lines[j])
                j += 1
            blocks.append(dict(path=pending, lang=lang, lines=body, start_line=i + 2))
            pending = None
            i = j + 1
            continue
        i += 1
    return blocks


def normalize(line):
    """比较前的归一化：去行尾空白、展开 tab。"""
    return line.replace("\t", "    ").rstrip()


def src_index(src_text):
    """规范化行 -> 位置列表。"""
    pos = {}
    for i, ln in enumerate(src_text.split("\n")):
        pos.setdefault(normalize(ln), []).append(i)
    return pos


def contiguous_run(block_lines, pos):
    """贪心判断 block_lines 是否在源文件中占据连续位置。

    返回 (ok, fail_idx, matched_positions)
    """
    last = -2
    got = []
    for k, ln in enumerate(block_lines):
        n = normalize(ln)
        cands = [i for i in pos.get(n, []) if i > last]
        if cands:
            got.append(cands[0])
            last = cands[0]
        else:
            return False, k, got
    return True, -1, got


def runs_of(block_lines, pos):
    """把块切成若干「位置连续」的子块：[(start_k, end_k_exclusive)]"""
    out = []
    cur_start = None
    last = -2
    for k, ln in enumerate(block_lines):
        n = normalize(ln)
        cands = [i for i in pos.get(n, []) if i > last]
        if cands:
            if cur_start is None:
                cur_start = k
            last = cands[0]
        else:
            if cur_start is not None:
                out.append((cur_start, k))
                cur_start = None
            last = -2
            if n.strip():
                out.append((k, k + 1))     # 无法匹配的行单独成块（由保真门禁报错）
    if cur_start is not None:
        out.append((cur_start, len(block_lines)))
    return out


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------
class R:
    G = "\033[32m"; Y = "\033[33m"; Rd = "\033[31m"; D = "\033[2m"; B = "\033[1m"; X = "\033[0m"

    @staticmethod
    def ok(msg): print(f"{R.G}✓{R.X} {msg}")

    @staticmethod
    def bad(msg): print(f"{R.Rd}✗{R.X} {msg}")

    @staticmethod
    def warn(msg): print(f"{R.Y}!{R.X} {msg}")

    @staticmethod
    def info(msg): print(f"{R.D}{msg}{R.X}")
