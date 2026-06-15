#!/bin/bash
# macOS 一键启动标注（Finder 双击本文件）。无需 Python、无需服务器。
# 本脚本在 resources/ 内；数据目录 data/ 在上一层（顶层）。它做三件事：
# ① 把 ../data/*.jsonl 刷成 data.js；② 把 ne_graph.json 刷成 ne_graph.js（若有）；
# ③ 用默认浏览器打开总览页（file://）。../data/ 里的 jsonl 有增改后，重新双击即可。
cd "$(dirname "$0")" || exit 1

# ① ../data/*.jsonl -> data.js（与页面同在 resources/，自动加载，省去手动选文件）
if [ -d ../data ]; then
    shopt -s nullglob
    files=(../data/*.jsonl)
    if [ ${#files[@]} -gt 0 ]; then
        {
            printf 'window.FAULT_GROUPS_DATA=['
            first=1
            for f in "${files[@]}"; do
                while IFS= read -r line || [ -n "$line" ]; do
                    [ -z "${line//[[:space:]]/}" ] && continue
                    if [ $first -eq 1 ]; then first=0; else printf ','; fi
                    printf '%s' "$line"
                done < "$f"
            done
            printf '];\n'
        } > data.js
        echo "已从 ${#files[@]} 个 jsonl 生成 data.js"
    else
        rm -f data.js   # 清掉旧的，避免自动加载过期数据
        echo "提示：../data/ 下没有 .jsonl，将不自动加载故障组（可在页面手动选择）。"
    fi
else
    rm -f data.js       # 清掉旧的，避免自动加载过期数据
    echo "提示：未发现 ../data/ 目录，将不自动加载故障组（可在页面手动选择）。"
fi

# ② ne_graph.json -> ne_graph.js（可选）
if [ -f ne_graph.json ]; then
    { printf 'window.NE_GRAPH_DATA='; cat ne_graph.json; printf ';'; } > ne_graph.js
    echo "已从 ne_graph.json 生成 ne_graph.js"
elif [ -f ne_graph.js ]; then
    rm -f ne_graph.js   # 源文件已删，移除旧的生成物
    echo "提示：未发现 ne_graph.json，已移除旧的 ne_graph.js。"
fi

# ③ 打开总览页
open ne_propagation_labeling_browser.html
