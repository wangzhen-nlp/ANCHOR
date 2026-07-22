#!/bin/bash
# macOS 一键启动标注（Finder 双击本文件）。无需 Python、无需服务器。
# 本文件在顶层；页面与生成物在 resources/，故障组数据在顶层 data/。它做三件事：
# ① 把 data/*.jsonl 刷成 resources/data.js；② 把 resources/ne_graph.json 刷成 resources/ne_graph.js（若有）；
# ③ 用默认浏览器打开 resources/ 下的总览页（file://）。data/ 里的 jsonl 有增改后，重新双击即可。
cd "$(dirname "$0")" || exit 1

RES="resources"
if [ ! -d "$RES" ]; then
    echo "未找到 $RES/ 目录，无法启动。"
    read -r -p "按回车关闭…" _
    exit 1
fi

# ① data/*.jsonl（顶层）-> resources/data.js（与页面同目录，自动加载，省去手动选文件）
if [ -d data ]; then
    shopt -s nullglob
    files=(data/*.jsonl)
    if [ ${#files[@]} -gt 0 ]; then
        {
            printf 'window.FAULT_GROUPS_DATA=['
            first=1
            for f in "${files[@]}"; do
                while IFS= read -r line || [ -n "$line" ]; do
                    # 跳过纯空白行。注意：不能用 ${line//[[:space:]]/}，macOS 自带 bash 3.2
                    # 对长字符串的全局替换有 O(n²) 性能塌陷，单行大 JSON 会卡死。
                    case $line in *[![:space:]]*) : ;; *) continue ;; esac
                    if [ $first -eq 1 ]; then first=0; else printf ','; fi
                    printf '%s' "$line"
                done < "$f"
            done
            printf '];\n'
        } > "$RES/data.js"
        echo "已从 ${#files[@]} 个 jsonl 生成 $RES/data.js"
    else
        rm -f "$RES/data.js"   # 清掉旧的，避免自动加载过期数据
        echo "提示：data/ 下没有 .jsonl，将不自动加载故障组（可在页面手动选择）。"
    fi
else
    rm -f "$RES/data.js"       # 清掉旧的，避免自动加载过期数据
    echo "提示：未发现 data/ 目录，将不自动加载故障组（可在页面手动选择）。"
fi

# ② resources/ne_graph.json -> resources/ne_graph.js（可选）
if [ -f "$RES/ne_graph.json" ]; then
    { printf 'window.NE_GRAPH_DATA='; cat "$RES/ne_graph.json"; printf ';'; } > "$RES/ne_graph.js"
    echo "已从 $RES/ne_graph.json 生成 $RES/ne_graph.js"
elif [ -f "$RES/ne_graph.js" ]; then
    rm -f "$RES/ne_graph.js"   # 源文件已删，移除旧的生成物
    echo "提示：未发现 $RES/ne_graph.json，已移除旧的 ne_graph.js。"
fi

# ③ 打开总览页（file://，默认浏览器）
open "$RES/ne_propagation_labeling_browser.html"
