#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
F3 市场特征在线爬取 skill（可复用）
==================================
从公开接口实时计算 F3 的 35 维市场特征（mkt_ 前缀）。

数据来源（免费公开接口，已实测）：
  - 腾讯 K 线（web.ifzq.gtimg.cn）-> 日线 OHLC + 成交量 -> A2 收益风险 + A3 流动性
  - 腾讯行情（qt.gtimg.cn）        -> 市值/PE/PB        -> A1 估值
  - 上证指数（腾讯 K 线）           -> 市场基准收益       -> A2 超额收益

公式与离线版（F3-F5/F3 目录下的 build 脚本）保持一致。

当前在线可算：A1(部分) + A2 + A3 ≈ 20 维；
TODO（实时难获取，返回 NaN）：
  - A1 的行业偏离(2)、PE/PB 环比(2)：需同行业公司 + 历史估值
  - A4 融资融券(4)：需交易所/东财两融接口
  - A4 机构持仓(3)：需季报机构明细
  - A5 风险警示(4)：需公告 PDF 文本解析
"""

import numpy as np
import pandas as pd
import requests

try:
    from .financial_data_fetch import _to_6digit, _market_of, _get_quote_snapshot
except ImportError:
    from financial_data_fetch import _to_6digit, _market_of, _get_quote_snapshot


# ============================================================
# 1. 工具：日线 K 线抓取
# ============================================================

def _get_daily_kline(company_code, days=120):
    """腾讯前复权日线 -> DataFrame[trade_date, close, volume, amount]

    amount(成交额, 元) = 成交量(手) * 100 * 收盘价，为近似值。
    """
    market = _market_of(company_code).lower()   # 'sz' / 'sh'
    code6 = _to_6digit(company_code)
    url = 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'
    r = requests.get(url, params={'param': f'{market}{code6},day,,,{days},qfq'},
                     headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
    r.encoding = 'utf-8'
    data = r.json().get('data', {})
    node = data.get(f'{market}{code6}', {})

    # 前复权日线 key 可能是 qfqday 或 day
    rows = node.get('qfqday') or node.get('day') or []
    if not rows:
        return pd.DataFrame(columns=['trade_date', 'close', 'volume', 'amount'])

    # 注意：有分红的股票，除权除息日会多一个第7字段(分红信息dict)，只取前6个
    df = pd.DataFrame([row[:6] for row in rows],
                      columns=['trade_date', 'open', 'close', 'high', 'low', 'volume'])
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df['amount'] = df['volume'] * 100 * df['close']   # 手 -> 股 -> 元（近似）
    return df[['trade_date', 'close', 'volume', 'amount']].reset_index(drop=True)


def _get_index_kline(days=120):
    """上证指数(sh000001)日线 -> DataFrame[trade_date, close]（作市场基准）"""
    url = 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'
    r = requests.get(url, params={'param': f'sh000001,day,,,{days},qfq'},
                     headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
    r.encoding = 'utf-8'
    data = r.json().get('data', {})
    node = data.get('sh000001', {})
    rows = node.get('qfqday') or node.get('day') or []
    if not rows:
        return pd.DataFrame(columns=['trade_date', 'close'])
    df = pd.DataFrame([row[:6] for row in rows],
                      columns=['trade_date', 'open', 'close', 'high', 'low', 'volume'])
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    return df[['trade_date', 'close']].reset_index(drop=True)


# ============================================================
# 2. 公式（与离线 build 脚本一致）
# ============================================================

def _compound_return(arr):
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0 or np.isnan(arr).any():
        return np.nan
    return float(np.prod(1.0 + arr) - 1.0)


def _max_drawdown(arr):
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0 or np.isnan(arr).any():
        return np.nan
    wealth = np.cumprod(1.0 + arr)
    running_max = np.maximum.accumulate(wealth)
    return float(np.min(wealth / running_max - 1.0))


def _daily_returns(df):
    """从收盘价算日收益率（前复权价 pct_change ≈ 全收益）"""
    return df['close'].pct_change().to_numpy(dtype=float)


# ============================================================
# 3. 各子族计算
# ============================================================

def compute_a1_valuation(company_code, daily):
    """A1 估值特征：市值/PE/PB(腾讯快照) + 市值环比(近似)。返回 dict"""
    quote = _get_quote_snapshot(company_code)
    feat = {}

    feat['mkt_market_cap'] = quote.get('market_cap')
    mc = feat['mkt_market_cap']
    feat['mkt_log_market_cap'] = np.log1p(mc) if (mc is not None and mc >= 0) else np.nan
    feat['mkt_pe_ratio'] = quote.get('pe_ratio')
    feat['mkt_pb_ratio'] = quote.get('pb_ratio')

    # 市值环比：约等于 收盘价今/约63个交易日前 - 1（假设股本不变，近似）
    closes = daily['close'].dropna().to_numpy(dtype=float)
    if len(closes) >= 64:
        feat['mkt_market_cap_qoq'] = float(closes[-1] / closes[-64] - 1.0)
    else:
        feat['mkt_market_cap_qoq'] = np.nan

    # 以下需要历史估值/同行业，实时暂不可算
    feat['mkt_pe_change_qoq'] = np.nan
    feat['mkt_pb_change_qoq'] = np.nan
    feat['mkt_cap_industry_zscore'] = np.nan
    feat['mkt_pb_industry_zscore'] = np.nan
    return feat


def compute_a2_return_risk(daily, index):
    """A2 收益与风险特征（8 维）。返回 dict"""
    feat = {}
    ret = _daily_returns(daily)[1:]            # 去掉首个 NaN
    idx_ret = _daily_returns(index)[1:]

    def window(arr, n):
        return arr[-n:] if len(arr) >= n else np.array([])

    r5, r20, r60 = window(ret, 5), window(ret, 20), window(ret, 60)
    feat['mkt_return_5d'] = _compound_return(r5) if len(r5) == 5 else np.nan
    feat['mkt_return_20d'] = _compound_return(r20) if len(r20) == 20 else np.nan
    feat['mkt_return_60d'] = _compound_return(r60) if len(r60) == 60 else np.nan

    # 超额收益 = 个股20日 - 市场20日
    im20 = window(idx_ret, 20)
    if len(r20) == 20 and len(im20) == 20:
        feat['mkt_excess_return_20d'] = float(feat['mkt_return_20d'] - _compound_return(im20))
    else:
        feat['mkt_excess_return_20d'] = np.nan

    feat['mkt_volatility_20d'] = float(np.std(r20, ddof=1)) if len(r20) == 20 else np.nan
    feat['mkt_volatility_60d'] = float(np.std(r60, ddof=1)) if len(r60) == 60 else np.nan
    feat['mkt_max_drawdown_60d'] = _max_drawdown(r60) if len(r60) == 60 else np.nan
    feat['mkt_extreme_down_days_20d'] = int(np.sum(r20 < -0.05)) if len(r20) == 20 else np.nan
    return feat


def compute_a3_liquidity(daily):
    """A3 成交与流动性特征（7 维）。返回 dict"""
    feat = {}
    feat['mkt_volume_ratio_20d'] = np.nan
    feat['mkt_amount_ratio_20d'] = np.nan
    feat['mkt_volume_cv_20d'] = np.nan
    feat['mkt_amount_cv_20d'] = np.nan
    feat['mkt_abnormal_volume_days_20d'] = np.nan
    feat['mkt_zero_volume_days_20d'] = np.nan
    feat['mkt_amihud_illiquidity_20d'] = np.nan

    vol = daily['volume'].to_numpy(dtype=float)
    amt = daily['amount'].to_numpy(dtype=float)
    ret = _daily_returns(daily)

    if len(vol) < 20:
        return feat

    vol20 = vol[-20:]
    amt20 = amt[-20:]
    ret20 = ret[-20:]

    # 当日/前19日均
    mean_vol = np.nanmean(vol20[:-1])
    if mean_vol > 0:
        feat['mkt_volume_ratio_20d'] = float(vol20[-1] / mean_vol)
    mean_amt = np.nanmean(amt20[:-1])
    if mean_amt > 0:
        feat['mkt_amount_ratio_20d'] = float(amt20[-1] / mean_amt)

    # 变异系数
    m20v = np.nanmean(vol20)
    if m20v > 0:
        feat['mkt_volume_cv_20d'] = float(np.nanstd(vol20, ddof=1) / m20v)
    m20a = np.nanmean(amt20)
    if m20a > 0:
        feat['mkt_amount_cv_20d'] = float(np.nanstd(amt20, ddof=1) / m20a)

    # 异常放量天数（> 均值 + 2*标准差）
    vol_mean, vol_std = np.nanmean(vol20), np.nanstd(vol20, ddof=1)
    if np.isfinite(vol_std):
        feat['mkt_abnormal_volume_days_20d'] = int(np.nansum(vol20 > vol_mean + 2 * vol_std))

    # 零成交量天数
    feat['mkt_zero_volume_days_20d'] = int(np.nansum(vol20 == 0))

    # Amihud 非流动性 = mean(|ret| / (成交额/百万元))
    valid = np.isfinite(ret20) & np.isfinite(amt20) & (amt20 > 0)
    if valid.sum() >= 15:
        amihud = np.abs(ret20[valid]) / (amt20[valid] / 1_000_000)
        feat['mkt_amihud_illiquidity_20d'] = float(np.nanmean(amihud))

    return feat


# ============================================================
# 4. TODO 子族（实时暂不可算，返回 NaN）
# ============================================================

def _nan_features(names):
    return {n: np.nan for n in names}


def compute_a4_margin():
    """A4 融资融券（4 维）—— TODO 接交易所/东财两融接口"""
    return _nan_features([
        'mkt_financing_balance', 'mkt_securities_balance',
        'mkt_margin_total_balance', 'mkt_financing_balance_change',
    ])


def compute_a4_institution():
    """A4 机构持仓（3 维）—— TODO 接季报机构明细"""
    return _nan_features([
        'mkt_institutional_holding_ratio',
        'mkt_institutional_holding_change',
        'mkt_institutional_holder_count',
    ])


def compute_a5_risk_warning():
    """A5 风险警示（4 维）—— TODO 解析风险提示公告"""
    return _nan_features([
        'mkt_risk_warning_count_30d', 'mkt_risk_warning_count_90d',
        'mkt_risk_warning_flag_30d', 'mkt_days_since_last_risk_warning',
    ])


# ============================================================
# 5. 统一入口
# ============================================================

F3_FEATURE_ORDER = [
    # A1
    'mkt_market_cap', 'mkt_log_market_cap', 'mkt_pe_ratio', 'mkt_pb_ratio',
    'mkt_market_cap_qoq', 'mkt_pe_change_qoq', 'mkt_pb_change_qoq',
    'mkt_cap_industry_zscore', 'mkt_pb_industry_zscore',
    # A2
    'mkt_return_5d', 'mkt_return_20d', 'mkt_return_60d', 'mkt_excess_return_20d',
    'mkt_volatility_20d', 'mkt_volatility_60d', 'mkt_max_drawdown_60d', 'mkt_extreme_down_days_20d',
    # A3
    'mkt_volume_ratio_20d', 'mkt_amount_ratio_20d', 'mkt_volume_cv_20d', 'mkt_amount_cv_20d',
    'mkt_abnormal_volume_days_20d', 'mkt_zero_volume_days_20d', 'mkt_amihud_illiquidity_20d',
    # A4
    'mkt_financing_balance', 'mkt_securities_balance', 'mkt_margin_total_balance', 'mkt_financing_balance_change',
    'mkt_institutional_holding_ratio', 'mkt_institutional_holding_change', 'mkt_institutional_holder_count',
    # A5
    'mkt_risk_warning_count_30d', 'mkt_risk_warning_count_90d',
    'mkt_risk_warning_flag_30d', 'mkt_days_since_last_risk_warning',
]


def crawl_market_features(company_code, days=120):
    """在线计算 F3 的 35 维市场特征。返回 dict（按 F3_FEATURE_ORDER 排序）"""
    daily = _get_daily_kline(company_code, days=days)
    index = _get_index_kline(days=days)

    feats = {}
    feats.update(compute_a1_valuation(company_code, daily))
    feats.update(compute_a2_return_risk(daily, index))
    feats.update(compute_a3_liquidity(daily))
    feats.update(compute_a4_margin())
    feats.update(compute_a4_institution())
    feats.update(compute_a5_risk_warning())

    # 按标准顺序输出，补齐可能缺失的 key
    return {k: feats.get(k, np.nan) for k in F3_FEATURE_ORDER}


if __name__ == '__main__':
    import sys, io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    f = crawl_market_features("000004.SZ")
    non_nan = sum(1 for v in f.values() if v is not None and not (isinstance(v, float) and v != v))
    print(f"F3 在线特征: {len(f)} 维, 非 NaN {non_nan} 维")
    for k, v in f.items():
        if v is not None and not (isinstance(v, float) and v != v):
            print(f"  {k} = {v}")
