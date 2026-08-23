#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单入口导航壳：一个端口聚合 7 个页面（主控 + 6 个 Agent 审计页）。
运行：streamlit run 导航入口.py --server.port 8501
侧边栏自动生成导航，默认打开「主控 Agent」全流程演示页；
各页面脚本仍可单独 streamlit run（独立审计/调试）。
"""
import streamlit as st

pages = [
    st.Page(
        "主控agent.py",
        title="主控 Agent",
        icon=":material/dashboard:",
        url_path="main",
        default=True,
    ),
    st.Page(
        "公告研读agent.py",
        title="公告研读 Agent",
        icon=":material/article:",
        url_path="announcement",
    ),
    st.Page(
        "财务异常agent.py",
        title="财务异常 Agent",
        icon=":material/account_balance:",
        url_path="financial",
    ),
    st.Page(
        "预测建模agent.py",
        title="预测建模 Agent",
        icon=":material/model_training:",
        url_path="predictor",
    ),
    st.Page(
        "案例匹配agent.py",
        title="案例匹配 Agent",
        icon=":material/compare_arrows:",
        url_path="case",
    ),
    st.Page(
        "归因分析agent.py",
        title="归因分析 Agent",
        icon=":material/psychology:",
        url_path="attribution",
    ),
    st.Page(
        "报告生成agent.py",
        title="报告生成 Agent",
        icon=":material/description:",
        url_path="report",
    ),
]

pg = st.navigation(pages, position="sidebar")
pg.run()
