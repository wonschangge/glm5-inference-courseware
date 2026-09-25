#!/usr/bin/env bash
# 全量门禁：lint -> split_blocks -> fidelity -> params -> coverage -> render
# 用法:
#   tools/run_gates.sh                 # 静态门禁 + 渲染（--all）
#   tools/run_gates.sh --static        # 只跑静态门禁（秒级）
#   tools/run_gates.sh --lesson L0-01  # 只跑一课
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

MODE="${1:---all}"
LESSON="${2:-}"
PY=python3
FAIL=0

hdr() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
step() {
  hdr "$1"; shift
  "$@" ; rc=$?
  if [ $rc -ne 0 ]; then FAIL=1; fi
  return 0
}

if [ "$MODE" = "--lesson" ]; then
  L=("--lesson" "$LESSON")
elif [ "$MODE" = "--static" ]; then
  L=()
else
  L=()
fi

step "A4 语法/结构 (lint)"      $PY tools/lint_lessons.py "${L[@]:-}" --quiet
step "引用块连续性 (split)"     $PY tools/split_blocks.py --check "${L[@]:-}"
step "A1/A3 保真 (fidelity)"    $PY tools/check_fidelity.py "${L[@]:-}" --quiet
step "A2 参数 (params)"         $PY tools/check_params.py "${L[@]:-}" --quiet
step "B 覆盖度 (coverage)"      $PY tools/check_coverage.py "${L[@]:-}" --quiet

if [ "$MODE" = "--static" ]; then
  [ $FAIL -eq 0 ] && printf '\n\033[32m静态门禁全绿\033[0m\n' || printf '\n\033[31m静态门禁有失败项\033[0m\n'
  exit $FAIL
fi

if [ "$MODE" = "--lesson" ]; then
  step "C 渲染 (render)" $PY tools/check_render.py --lesson "$LESSON"
else
  step "C 渲染 (render, 全量)" $PY tools/check_render.py --all --quiet
fi

if [ $FAIL -eq 0 ]; then printf '\n\033[32m★ 全部门禁通过\033[0m\n'; else printf '\n\033[31m★ 存在失败门禁\033[0m\n'; fi
exit $FAIL
