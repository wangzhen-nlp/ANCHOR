#!/usr/bin/env python3
"""从字符串中抽取 site_id，返回 "site_id1,site_id2" 形式的字符串。

输入格式示例:
    07BNS0184: 150122_AMFPLB01,150122_vSGLPG01,GLTE_MUARA_PADANG_BNS_EP

每行冒号前的编号（如 07BNS0184）是 site_id；冒号后是该站点下的
网元名（数字开头，如 150122_AMFPLB01）和站点名称（如 GLTE_xxx）。
"""

import argparse
import re

# 网元名特征: 以数字开头 + 下划线 + 其余部分
NE_PATTERN = re.compile(r"^\d+_\w+$")


def extract_site_id(line: str):
    """从单行中抽取 site_id（冒号前的编号），无法解析时返回 None。"""
    line = line.strip()
    if not line or ":" not in line:
        return None
    site_id, _, _ = line.partition(":")
    return site_id.strip() or None


def extract_site_name(line: str):
    """从单行中抽取站点名称（冒号后非网元的那一项），无法解析时返回 None。"""
    line = line.strip()
    if not line or ":" not in line:
        return None
    _, _, rest = line.partition(":")
    names = [item.strip() for item in rest.split(",")
             if item.strip() and not NE_PATTERN.match(item.strip())]
    return names[0] if names else None


def extract_site_ids(text: str) -> str:
    """从整段字符串中抽取所有 site_id，返回逗号拼接的字符串。"""
    ids = (extract_site_id(line) for line in text.splitlines())
    return ",".join(site_id for site_id in ids if site_id)


def main():
    parser = argparse.ArgumentParser(
        description="从字符串中抽取 site_id，输出 site_id1,site_id2 形式")
    parser.add_argument("text", nargs="?", default=DEMO,
                        help="待解析的多行字符串；缺省时使用内置示例数据")
    args = parser.parse_args()
    print(extract_site_ids(args.text))


DEMO = """\
07BNS0184: 150122_AMFPLB01,150122_vSGLPG01,GLTE_MUARA_PADANG_BNS_EP
07BNS0182: GLT_MUARA_PDG_SLT_TB,150141_vSGLPG01,150141_AMFPLB01
07BNS0183: 150733_AMFPLB01,GLTE_AINUL_YAQIN_MT,150733_vSGLPG01
07BNS0187: 150600_AMFPLB01,GLTE_TIRTORAHARJO_MT,150600_vSGLPG01
07BNS0185: GLTE_JALUR_TB,150139_AMFPLB01,150139_vSGLPG01
07BNS0189: GLTE_MEKAR_JAYA_BNS_ST,150584_AMFPLB01
07BNS0188: GLTE_ARGO_MULYO_TG,150586_AMFPLB01
07OKI0127: 150769_AMFPLB01,GLTE_RANTAU_KARYA_TB,150769_vSGLPG01
07OKI0133: 150582_vSGLPG01,150582_AMFPLB01,GLT_MUKTI_JAYA_OKI_MT
"""

if __name__ == "__main__":
    main()
