#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L3-08 构建器：把模板里的行号标记展开成**逐字**引用块。

为什么要有这个脚本：保真门禁最严的一条是「一个字都不能改」。
手抄 143 行代码一定会出错，所以引用块一律由本脚本从源文件**切片**生成。

用法：
    python3 _data/build_l308.py          # 展开两个模板 -> L3-backbone/L3-08/{source.md,lesson.js,README.md}
    python3 _data/build_l308.py --dump   # 只打印各块的首尾行，核对行号

标记写法：@@B:glm5:1280:1299@@   ->  源文件第 1280..1299 行（含两端，1 基）
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # courseware/
UP = os.environ.get("GLM5_UPSTREAM", os.path.abspath(
    os.path.join(ROOT, "..", "upstream-transformers", "src", "transformers")))

FILES = {
    "glm5": "models/glm5_next/modeling_glm5_next.py",
    "layers": "modeling_layers.py",
    "outputs": "modeling_outputs.py",
}

LESSON_DIR = os.path.join(ROOT, "L3-backbone", "L3-08")

# 模板 -> 产物。source.md / lesson.js / README.md 三个都从模板展开（README 里的结论也带逐字引用）。
JOBS = [
    ("_data/tpl_l308_source.md", "source.md"),
    ("_data/tpl_l308_lesson.js", "lesson.js"),
    ("_data/tpl_l308_readme.md", "README.md"),
]

MARK = re.compile(r"@@B:(glm5|layers|outputs):(\d+):(\d+)@@")


def slice_lines(rel, a, b):
    with open(os.path.join(UP, rel), encoding="utf-8") as f:
        all_lines = f.read().split("\n")
    assert 1 <= a <= b <= len(all_lines), (rel, a, b, len(all_lines))
    return all_lines[a - 1:b]


def expand(text, mode):
    """mode: 'md' 展开成 <!-- src: --> + ```python 围栏；'raw' 只展开原始行。"""
    n = [0]

    def sub(m):
        key, a, b = m.group(1), int(m.group(2)), int(m.group(3))
        rel = FILES[key]
        lines = slice_lines(rel, a, b)
        body = "\n".join(lines)
        # lesson.js 里代码要放进反引号模板字符串：源文件里的反引号 / ${ 必须转义，
        # 否则会把 JS 字面量截断。转义只改 JS 源码的写法，**渲染出来的代码仍逐字**。
        # source.md 用 ``` 围栏，不受此限。
        if mode == "raw":
            hits = [x for x, c in (("反引号", "`" in body), ("${", "${" in body)) if c]
            if hits:
                print("  ! 转义 %s L%d-%d 里的 %s" % (rel, a, b, " / ".join(hits)))
            body = body.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
        n[0] += 1
        if mode == "md":
            return "<!-- src: %s -->\n```python\n%s\n```" % (rel, body)
        return body

    out = MARK.sub(sub, text)
    return out, n[0]


def main(argv):
    dump = "--dump" in argv
    total = 0
    for tpl, out_name in JOBS:
        tpl_path = os.path.join(ROOT, tpl)
        if not os.path.isfile(tpl_path):
            print("! 缺少模板", tpl_path)
            continue
        text = open(tpl_path, encoding="utf-8").read()
        mode = "raw" if out_name.endswith(".js") else "md"
        if out_name == "source.md":
            body, n = expand(text, "md")
        else:
            # lesson.js / README.md：标记处只替换为原始行（围栏由模板自己写）
            body, n = expand(text, "raw")
        total += n
        if dump:
            print("%-14s <- %-28s 展开 %d 个块" % (out_name, tpl, n))
            for m in MARK.finditer(text):
                key, a, b = m.group(1), int(m.group(2)), int(m.group(3))
                ls = slice_lines(FILES[key], a, b)
                print("    %-8s L%-5d..%-5d %-3d 行 | 首: %s" % (key, a, b, len(ls), ls[0][:64]))
                print("    %-8s %-16s    | 尾: %s" % ("", "", ls[-1][:64]))
            continue
        os.makedirs(LESSON_DIR, exist_ok=True)
        dst = os.path.join(LESSON_DIR, out_name)
        with open(dst, "w", encoding="utf-8") as f:
            f.write(body)
        print("写入 %s（%d 行）" % (dst, body.count("\n") + 1))
    if dump:
        print("合计标记 %d 个" % total)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
