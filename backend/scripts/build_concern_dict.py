#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
离线任务：抽取"监管关注点 ↔ 财务异常指标"映射词典
================================================
从历史问询函反解"什么样的情况会被问"：解析问询事项 → 关联函中引用的财务指标
→ 生成映射词典（关注点 → 指标候选池），供案例检索匹配规则 / 预测特征 / 归因素材使用。

运行（从项目根目录）：
    python -m backend.scripts.build_concern_dict

TODO: 由后续开发填充实现。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def main():
    # TODO: 解析问询函 → 抽取关注点+引用指标 → 输出词典 JSON
    raise NotImplementedError


if __name__ == "__main__":
    main()
