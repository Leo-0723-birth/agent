#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
F4 舆情情绪 · 在线爬取 skill（可复用）
====================================
数据源：
  - B1 新闻关注度 + B2 新闻情绪：东方财富资讯搜索接口（search-api-web.eastmoney.com，
    已实测可用），返回近 N 日新闻标题/正文。
  - B3 股吧热度 + B4 股吧情绪：东方财富股吧接口（gbapi.eastmoney.com）。
    该接口当前对爬虫返回 403，因此 B3/B4 会优雅回退为 NaN（见下方说明），
    一旦接口恢复即可自动填值。

情绪打分（本 skill 的核心）：
  默认用内置金融情感词典打分（离线、免费、可复现）；
  传 use_llm=True + 真实 LLM 配置（api_key/base_url）时，改为让大模型对新闻标题
  批量打情绪分（-1 负 / 0 中 / 1 正），更贴合语义。LLM 走 OpenAI 兼容的
  chat/completions 协议（DeepSeek / 通义 / vLLM 等皆可），只用 requests，不依赖
  openai/anthropic 包。

LLM 配置：环境变量 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL（或经 kwargs 直接传
  api_key / base_url / model）。未配 key 或 backend='mock' 时自动退化为词典打分。

输出 19 维（与离线 sentiment_features.csv 的列顺序一致）：
  B1: sent_news_count_5d/10d/30d, sent_news_title_count_30d, sent_news_daily_peak_30d
  B2: sent_sentiment_mean_30d, sent_sentiment_volatility_30d,
      sent_negative_news_count_30d, sent_negative_ratio_30d, sent_negative_peak_30d
  B3: sent_post_count_5d/30d, sent_comment_count_30d, sent_read_count_30d, sent_post_daily_peak_30d
  B4: sent_guba_sentiment_mean_30d, sent_guba_sentiment_volatility_30d,
      sent_guba_positive_ratio_30d, sent_guba_negative_ratio_30d
"""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://so.eastmoney.com/",
}

# 内置金融情感词典（正/负面关键词，可扩充）
_POS_WORDS = ["增长", "盈利", "扭亏", "回购", "增持", "分红", "中标", "签约", "合作",
              "突破", "新高", "向好", "预增", "利好", "涨停", "扩张", "投产", "获批",
              "减亏", "改善", "复苏", "超预期", "景气"]
_NEG_WORDS = ["亏损", "风险", "警示", "违规", "处罚", "退市", "摘牌", "减持", "诉讼",
              "立案", "调查", "质押", "爆雷", "暴跌", "违约", "失信", "下滑", "下降",
              "造假", "欺诈", "问询", "监管", "停牌", "冻结", "债务", "危机", "预亏",
              "巨亏", "跌停", "利空", "终止", "重组失败"]

# 输出顺序（离线口径）
F4_FEATURE_ORDER = [
    # B1 新闻关注度
    "sent_news_count_5d", "sent_news_count_10d", "sent_news_count_30d",
    "sent_news_title_count_30d", "sent_news_daily_peak_30d",
    # B2 新闻情绪
    "sent_sentiment_mean_30d", "sent_sentiment_volatility_30d",
    "sent_negative_news_count_30d", "sent_negative_ratio_30d", "sent_negative_peak_30d",
    # B3 股吧热度
    "sent_post_count_5d", "sent_post_count_30d", "sent_comment_count_30d",
    "sent_read_count_30d", "sent_post_daily_peak_30d",
    # B4 股吧情绪
    "sent_guba_sentiment_mean_30d", "sent_guba_sentiment_volatility_30d",
    "sent_guba_positive_ratio_30d", "sent_guba_negative_ratio_30d",
]


def _extract_code(company_code: str) -> str:
    """'000004.SZ' / 'SZ000004' / '000004' -> '000004'"""
    from .stock_code import normalize_stock_code

    return normalize_stock_code(company_code).split(".", 1)[0]


# ============================================================
# 1. 新闻抓取（东方财富资讯搜索）
# ============================================================

def _search_news(company_code: str, page_size: int = 50) -> list:
    """东财资讯搜索，返回 [{date, title, content}] 列表（按时间倒序）。"""
    code6 = _extract_code(company_code)
    param = json.dumps({
        "uid": "", "keyword": code6,
        "type": ["cmsArticleWebOld"],
        "client": "web", "clientType": "web", "clientVersion": "curr",
        "param": {"cmsArticleWebOld": {"searchScope": "default", "sort": "time",
                                       "pageIndex": 1, "pageSize": page_size,
                                       "preTag": "", "postTag": ""}},
    }, ensure_ascii=False)
    r = requests.get("https://search-api-web.eastmoney.com/search/jsonp",
                     params={"cb": "cb", "param": param}, headers=_HEADERS, timeout=15)
    txt = r.text
    data = json.loads(txt[txt.find("(") + 1: txt.rfind(")")])
    items = (data.get("result") or {}).get("cmsArticleWebOld") or []
    out = []
    for it in items:
        out.append({
            "date": pd.to_datetime(it.get("date"), errors="coerce"),
            "title": (it.get("title") or "").strip(),
            "content": (it.get("content") or "").strip(),
        })
    return [x for x in out if pd.notna(x["date"])]


# ============================================================
# 2. 股吧抓取（东方财富股吧，当前 403，优雅回退）
# ============================================================

def _search_guba_posts(company_code: str, page_size: int = 50) -> list:
    """东财股吧帖子列表，返回 [{date, title, content, read_count, comment_count}]。

    gbapi.eastmoney.com 当前对爬虫返回 403，本函数捕获后返回空列表（B3/B4 退化为 NaN）。
    """
    code6 = _extract_code(company_code)
    headers = dict(_HEADERS, Referer=f"https://guba.eastmoney.com/list,{code6}.html")
    try:
        r = requests.get(
            "https://gbapi.eastmoney.com/web/api/stock/getStockArticleList",
            params={"code": code6, "pageindex": 1, "pagesize": page_size,
                    "sorttype": "1", "sortfield": "postdate"},
            headers=headers, timeout=12,
        )
        if r.status_code != 200:
            return []
        rows = (r.json().get("re") or {}).get("list") or []
        out = []
        for it in rows:
            out.append({
                "date": pd.to_datetime(it.get("post_publish_time"), errors="coerce"),
                "title": (it.get("post_title") or "").strip(),
                "content": (it.get("post_content") or "").strip(),
                "read_count": float(it.get("post_click_count") or 0),
                "comment_count": float(it.get("post_comment_count") or 0),
            })
        return [x for x in out if pd.notna(x["date"])]
    except Exception:
        return []


# ============================================================
# 3. 情绪打分（词典 + LLM 两种）
# ============================================================

def dict_sentiment(text: str):
    """金融情感词典打分，返回 [-1, 1]；无情感词返回 None。"""
    pos = sum(text.count(w) for w in _POS_WORDS)
    neg = sum(text.count(w) for w in _NEG_WORDS)
    total = pos + neg
    if total == 0:
        return None
    return (pos - neg) / total


_SENTIMENT_SYSTEM = (
    "你是专业的财经新闻情绪分析师。请判断每条新闻标题对目标公司的情绪倾向。"
    "只输出 JSON 数组，每个元素为 -1（负面）、0（中性）、1（正面）三者之一，"
    "长度与输入条数严格一致，不要输出其他文字。"
)


def _llm_sentiment(texts, backend="auto", model="", api_key="", base_url="", timeout=30):
    """用大模型对一组文本批量打情绪分，返回 list[int]（-1/0/1）；失败返回 None。

    走 OpenAI 兼容 chat/completions 协议，只用 requests。
    """
    import os
    model = model or os.environ.get("LLM_MODEL", "deepseek-chat")
    api_key = api_key or os.environ.get("LLM_API_KEY", "")
    base_url = base_url or os.environ.get("LLM_BASE_URL", "")
    if backend == "mock" or not api_key or not base_url:
        return None

    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    prompt = (
        f"以下是与某上市公司相关的 {len(texts)} 条新闻标题，请逐一判断情绪倾向：\n"
        f"{numbered}\n\n请输出 JSON 数组，长度 {len(texts)}，如 [-1, 0, 1, ...]。"
    )
    try:
        url = base_url.rstrip("/") + "/chat/completions"
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "temperature": 0,
                  "messages": [{"role": "system", "content": _SENTIMENT_SYSTEM},
                               {"role": "user", "content": prompt}]},
            timeout=timeout,
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]
        cleaned = raw.strip().strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        arr = json.loads(cleaned)
        if isinstance(arr, list) and len(arr) == len(texts):
            # 非法项置 0，保证返回长度与输入严格一致
            return [int(x) if str(x).lstrip("-").isdigit() else 0 for x in arr]
    except Exception:
        pass
    return None


def _score_texts(texts, use_llm=False, **llm_kwargs):
    """对一组标题打分，返回等长 list[Optional[float]]（每项在 [-1,1]）。"""
    if use_llm:
        llm_scores = _llm_sentiment(texts, **llm_kwargs)
        if llm_scores:
            return [float(s) for s in llm_scores]
    return [dict_sentiment(t) for t in texts]


# ============================================================
# 4. 特征计算
# ============================================================

def _news_features(news, use_llm=False, **llm_kwargs) -> dict:
    """B1 新闻关注度 + B2 新闻情绪（10 维）。"""
    feat: dict = {}
    if not news:
        return feat

    df = pd.DataFrame(news).sort_values("date").reset_index(drop=True)
    latest = df["date"].max()

    def window(days: int):
        return df[df["date"] >= (latest - timedelta(days=days))]

    n5 = window(5)
    n10 = window(10)
    n30 = window(30)

    # B1 关注度
    feat["sent_news_count_5d"] = float(len(n5))
    feat["sent_news_count_10d"] = float(len(n10))
    feat["sent_news_count_30d"] = float(len(n30))
    feat["sent_news_title_count_30d"] = float((n30["title"].str.len() > 0).sum()) if len(n30) else 0.0
    daily = n30.groupby(n30["date"].dt.date).size() if len(n30) else pd.Series(dtype=int)
    feat["sent_news_daily_peak_30d"] = float(daily.max()) if len(daily) else 0.0

    # B2 情绪
    if len(n30):
        titles = n30["title"].tolist()
        scores = _score_texts(titles, use_llm=use_llm, **llm_kwargs)
        s = pd.Series([x for x in scores if x is not None], dtype=float)
        if len(s):
            feat["sent_sentiment_mean_30d"] = float(s.mean())
            feat["sent_sentiment_volatility_30d"] = float(s.std(ddof=0)) if len(s) >= 2 else 0.0
            neg = (s < 0)
            feat["sent_negative_news_count_30d"] = float(int(neg.sum()))
            feat["sent_negative_ratio_30d"] = float(neg.mean())
        # 负面峰值：按日聚合后单日负面占比最大值（复用上面算好的 scores，避免重复打分）
        n30c = n30.copy()
        n30c["neg"] = [x is not None and x < 0 for x in scores]
        if any(n30c["neg"]):
            feat["sent_negative_peak_30d"] = float(
                n30c.groupby(n30c["date"].dt.date)["neg"].mean().max())
    return feat


def _guba_features(posts, use_llm=False, **llm_kwargs) -> dict:
    """B3 股吧热度 + B4 股吧情绪（9 维）。"""
    feat: dict = {}
    if not posts:
        return feat

    df = pd.DataFrame(posts).sort_values("date").reset_index(drop=True)
    latest = df["date"].max()

    def window(days: int):
        return df[df["date"] >= (latest - timedelta(days=days))]

    n5 = window(5)
    n30 = window(30)

    # B3 热度
    feat["sent_post_count_5d"] = float(len(n5))
    feat["sent_post_count_30d"] = float(len(n30))
    feat["sent_comment_count_30d"] = float(n30["comment_count"].sum()) if len(n30) else 0.0
    feat["sent_read_count_30d"] = float(n30["read_count"].sum()) if len(n30) else 0.0
    daily = n30.groupby(n30["date"].dt.date).size() if len(n30) else pd.Series(dtype=int)
    feat["sent_post_daily_peak_30d"] = float(daily.max()) if len(daily) else 0.0

    # B4 情绪
    if len(n30):
        scores = _score_texts(n30["title"].tolist(), use_llm=use_llm, **llm_kwargs)
        s = pd.Series([x for x in scores if x is not None], dtype=float)
        if len(s):
            feat["sent_guba_sentiment_mean_30d"] = float(s.mean())
            feat["sent_guba_sentiment_volatility_30d"] = float(s.std(ddof=0)) if len(s) >= 2 else 0.0
            feat["sent_guba_positive_ratio_30d"] = float((s > 0).mean())
            feat["sent_guba_negative_ratio_30d"] = float((s < 0).mean())
    return feat


def compute_sentiment_features(company_code, use_llm=False, llm_backend="auto", **llm_kwargs) -> dict:
    """计算单公司 F4 全量 19 维舆情特征。"""
    feat: dict = {}
    try:
        news = _search_news(company_code)
        feat.update(_news_features(news, use_llm=use_llm, backend=llm_backend, **llm_kwargs))
    except Exception as e:
        print(f"[F4] 新闻抓取失败: {e}")
    try:
        posts = _search_guba_posts(company_code)
        feat.update(_guba_features(posts, use_llm=use_llm, backend=llm_backend, **llm_kwargs))
    except Exception as e:
        print(f"[F4] 股吧抓取失败: {e}")
    return feat


def crawl_sentiment_features(company_code, use_llm=False, llm_backend="auto", **llm_kwargs) -> dict:
    """F4 统一入口：返回 19 维特征 dict（按标准顺序，缺失补 NaN）。"""
    feat = compute_sentiment_features(
        company_code, use_llm=use_llm, llm_backend=llm_backend, **llm_kwargs)
    return {k: feat.get(k, np.nan) for k in F4_FEATURE_ORDER}


if __name__ == "__main__":
    import io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    f = crawl_sentiment_features("000004.SZ")
    non_nan = [k for k, v in f.items() if v is not None and not (isinstance(v, float) and v != v)]
    print(f"F4 在线特征: {len(f)} 维, 非 NaN {len(non_nan)} 维")
    for k, v in f.items():
        if v is not None and not (isinstance(v, float) and v != v):
            print(f"  {k} = {v}")
    print("（B3/B4 股吧特征若全部缺失，说明 gbapi.eastmoney.com 当前 403，属预期）")
