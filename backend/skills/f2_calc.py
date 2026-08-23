#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
F2 财务异常特征计算 skill（可复用）
====================================
从 f2_f6_pipeline.py 的 Step 3 抽出，封装成独立、可被 agent 反复调用的 skill。

一句话说明：
    输入一个「财务数据 DataFrame」，输出 67 维 F2 财务异常特征。

与原始 pipeline 的区别（关键改造）：
    原始代码是针对「全量批量数据」写的（3.7 万行一次性算完）；
    本 skill 让它变成「给多少算多少」——你传单公司的历史序列也能算，
    传整个行业的数据也能算，供 agent 按需调用。

67 维特征 = 7 个特征族：
    A1  盈利偿债能力  15 维
    A2  盈利质量      10 维
    A3  Beneish M-Score 盈余管理  9 维
    A3b Piotroski F-Score 基本面  10 维
    A4  Benford 定律  5 维
    A5  趋势波动      10 维
    A6  行业偏离      8 维
"""

import pandas as pd
import numpy as np
from scipy import stats
from collections import OrderedDict


# ============================================================
# 1. 工具函数
# ============================================================

def safe_div(a, b):
    """安全除法：分母绝对值 < 1e-12 时返回 NaN（避免除零报错）"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(b) > 1e-12, a / b, np.nan)
    return result


def calc_total_assets(df):
    """估算总资产（原始数据没有直接的总资产字段，用两个口径回退估算）

    v1 = 净利润 / (ROA/100)   —— 优先（ROA 有效时）
    v2 = 市值 / 市净率 / (1 - 资产负债率/100)  —— 回退
    """
    ta_v1 = safe_div(df['net_profit'], df['roa'] / 100)
    equity_v2 = safe_div(df['market_cap'], df['pb_ratio'])
    ta_v2 = safe_div(equity_v2, 1 - df['debt_to_assets_ratio'] / 100)
    total_assets = np.where(
        pd.notna(ta_v1) & (np.abs(df['roa'].values) > 0.01),
        ta_v1, ta_v2
    )
    return pd.Series(total_assets, index=df.index)


# ============================================================
# 2. 七个特征族计算函数（从 pipeline 原样抽出）
# ============================================================

def _market_cap_quintile(df):
    """市值五分位（1~5）：按 (report_period, industry) 组内 rank 分位。

    与离线训练表的 f2_market_cap_quintile 对齐（离线为全局分位，组内样本不足时略有差异）。
    """
    pct = df.groupby(['report_period', 'industry'])['market_cap'].rank(pct=True)
    return np.ceil(pct * 5).clip(lower=1, upper=5)


def calc_a1_profitability_solvency(df):
    """A1: 盈利与偿债能力 15 维"""
    f = pd.DataFrame(index=df.index)

    # 直接取字段
    f['f2_roe'] = df['roe']                                          # 净资产收益率
    f['f2_roa'] = df['roa']                                          # 总资产收益率
    f['f2_net_margin'] = safe_div(df['net_profit'], df['total_revenue'])  # 净利率
    f['f2_debt_ratio'] = df['debt_to_assets_ratio']                  # 资产负债率
    f['f2_pe'] = df['pe_ratio']                                      # 市盈率
    f['f2_pb'] = df['pb_ratio']                                      # 市净率
    f['f2_neg_pe_flag'] = (df['pe_ratio'] < 0).astype(int)           # 负市盈率标志（与训练表对齐）
    f['f2_neg_pb_flag'] = (df['pb_ratio'] < 0).astype(int)           # 负市净率标志（与训练表对齐）
    f['f2_log_market_cap'] = np.log(df['market_cap'].clip(lower=1))  # 对数市值
    f['f2_market_cap_quintile'] = _market_cap_quintile(df)           # 市值五分位 1~5（与训练表对齐）
    f['f2_ocf_to_revenue'] = safe_div(df['operating_cash_flow'], df['total_revenue'])  # 经营现金流/营收
    f['f2_ocf_to_profit'] = safe_div(df['operating_cash_flow'], df['net_profit'])       # 经营现金流/净利润
    f['f2_roe_industry_rank'] = df.groupby(['report_period', 'industry'])['roe'].rank(pct=True)  # ROE行业百分位
    f['f2_loss_flag'] = (df['net_profit'] < 0).astype(int)           # 亏损标志
    f['f2_high_debt_flag'] = (df['debt_to_assets_ratio'] > 70).astype(int)  # 高负债标志(>70%)

    return f


def calc_a2_earnings_quality(df):
    """A2: 盈利质量 10 维"""
    f = pd.DataFrame(index=df.index)
    total_assets = calc_total_assets(df)

    f['f2_accruals'] = df['net_profit'] - df['operating_cash_flow']          # 应计利润
    f['f2_accruals_to_assets'] = safe_div(f['f2_accruals'], total_assets)    # 应计/总资产
    f['f2_accruals_to_revenue'] = safe_div(f['f2_accruals'], df['total_revenue'])  # 应计/营收
    f['f2_ocf_to_assets'] = safe_div(df['operating_cash_flow'], total_assets)     # 经营现金流/总资产

    # 利润-现金流背离标志（盈利但经营现金流失血，或亏损但经营现金流入）
    f['f2_profit_ocf_diverge'] = (
        ((df['net_profit'] > 0) & (df['operating_cash_flow'] < 0)) |
        ((df['net_profit'] < 0) & (df['operating_cash_flow'] > 0))
    ).astype(int)

    # Sloan 应计质量 Z-score（同报告期同行业标准化）
    # 用 transform 计算分组均值/标准差，比 groupby.apply 更稳（兼容 pandas 2.x）
    accrual_to_assets = f['f2_accruals_to_assets'].copy()
    keys = [df['report_period'], df['industry']]
    acc_mean = accrual_to_assets.groupby(keys).transform('mean')
    acc_std = accrual_to_assets.groupby(keys).transform('std')
    f['f2_accrual_quality_zscore'] = np.where(
        acc_std > 0, (accrual_to_assets - acc_mean) / acc_std, 0.0
    )

    f['f2_neg_accruals_flag'] = (f['f2_accruals'] < 0).astype(int)

    # 经营现金流波动率（滚动 4 期标准差）
    f['f2_ocf_volatility_4q'] = df.groupby('company_code')['operating_cash_flow'].transform(
        lambda x: x.rolling(4, min_periods=2).std()
    )

    # 现金流/利润极端值（>5 或 <-1 视为异常）
    ocf_to_profit = safe_div(df['operating_cash_flow'], df['net_profit'])
    f['f2_ocf_to_profit_extreme'] = ((ocf_to_profit > 5) | (ocf_to_profit < -1)).astype(int)

    # 应计趋势（滚动 4 期线性回归斜率）
    def calc_slope(series):
        arr = series.values
        mask = ~np.isnan(arr)
        if mask.sum() < 3:
            return np.nan
        x = np.arange(len(arr))
        slope, _, _, _, _ = stats.linregress(x[mask], arr[mask])
        return slope

    f['f2_accruals_trend'] = accrual_to_assets.groupby(df['company_code']).transform(
        lambda s: s.rolling(4, min_periods=3).apply(calc_slope, raw=False)
    )

    return f


def calc_a3_beneish(df):
    """A3: Beneish M-Score 盈余管理检测 9 维"""
    f = pd.DataFrame(index=df.index)
    df_s = df.sort_values(['company_code', 'report_date'])

    prev_roa = df_s.groupby('company_code')['roa'].shift(1)
    f['f2_gmi'] = safe_div(prev_roa, df_s['roa'])                      # 毛利率指数(GMI近似)

    prev_rev = df_s.groupby('company_code')['total_revenue'].shift(1)
    f['f2_sgi'] = safe_div(df_s['total_revenue'], prev_rev)            # 销售收入指数(SGI)

    prev_lev = df_s.groupby('company_code')['debt_to_assets_ratio'].shift(1)
    f['f2_lvgi'] = safe_div(df_s['debt_to_assets_ratio'], prev_lev)    # 杠杆指数(LVGI)

    accruals = df_s['net_profit'] - df_s['operating_cash_flow']
    total_assets = calc_total_assets(df_s)
    f['f2_tata'] = safe_div(accruals, total_assets)                    # 总应计/总资产(TATA)

    # Beneish M-Score 简化版（先缩尾再套公式，M > -2.22 视为可疑）
    def winsorize(s, lo=0.01, hi=0.99):
        if s.notna().sum() < 3:
            return s
        lo_v, hi_v = s.quantile(lo), s.quantile(hi)
        return s.clip(lower=lo_v, upper=hi_v)

    gmi_w = winsorize(f['f2_gmi'])
    sgi_w = winsorize(f['f2_sgi'])
    tata_w = winsorize(f['f2_tata'])
    lvgi_w = winsorize(f['f2_lvgi'])

    f['f2_beneish_m'] = (
        -4.84
        + 0.528 * gmi_w.fillna(1.0)
        + 0.892 * sgi_w.fillna(1.0)
        + 4.679 * tata_w.fillna(0.0)
        - 0.327 * lvgi_w.fillna(1.0)
        + 0.92 * 1.0    # DSRI(默认中性)
        + 0.404 * 1.0   # AQI(默认中性)
        + 0.115 * 1.0   # DEPI(默认中性)
        - 0.172 * 1.0   # SGAI(默认中性)
    )

    # 缺少明细科目数据的指数，标记 NaN
    f['f2_dsri'] = np.nan   # 需要应收账款
    f['f2_aqi'] = np.nan    # 需要非流动资产
    f['f2_depi'] = np.nan   # 需要折旧
    f['f2_sgai'] = np.nan   # 需要销管费用

    # 对齐回原始顺序
    return f.loc[df.index]


def calc_a3b_piotroski(df):
    """A3b: Piotroski F-Score 基本面评分 10 维"""
    f = pd.DataFrame(index=df.index)
    df_s = df.sort_values(['company_code', 'report_date'])

    f['f2_p_roa'] = (df_s['roa'] > 0).astype(int)
    f['f2_p_cfo'] = (df_s['operating_cash_flow'] > 0).astype(int)

    prev_roa = df_s.groupby('company_code')['roa'].shift(1)
    f['f2_p_droa'] = (df_s['roa'] > prev_roa).astype(int)

    f['f2_p_accrual'] = (df_s['operating_cash_flow'] > df_s['net_profit']).astype(int)

    prev_lev = df_s.groupby('company_code')['debt_to_assets_ratio'].shift(1)
    f['f2_p_dlever'] = (df_s['debt_to_assets_ratio'] < prev_lev).astype(int)

    # 净利率变化（用净利率替代毛利率）
    net_margin = pd.Series(safe_div(df_s['net_profit'], df_s['total_revenue']), index=df_s.index)
    prev_nm = net_margin.groupby(df_s['company_code']).shift(1)
    f['f2_p_dmargin'] = (net_margin > prev_nm).astype(int).fillna(0).astype(int)

    # 资产周转率变化
    tot_assets = calc_total_assets(df_s)
    at = pd.Series(safe_div(df_s['total_revenue'], tot_assets), index=df_s.index)
    prev_at = at.groupby(df_s['company_code']).shift(1)
    f['f2_p_dturnover'] = (at > prev_at).astype(int).fillna(0).astype(int)

    score_cols = ['f2_p_roa', 'f2_p_cfo', 'f2_p_droa', 'f2_p_accrual',
                  'f2_p_dlever', 'f2_p_dmargin', 'f2_p_dturnover']
    f['f2_p_score'] = sum(f[c] for c in score_cols)

    # 不可计算项
    f['f2_p_dliquid'] = 0.5  # 需要流动资产/负债
    f['f2_p_equity'] = 0.5    # 需要股本数据

    return f.loc[df.index]


def calc_a4_benford(df):
    """A4: Benford 定律数字分布异常检测 5 维"""
    f = pd.DataFrame(index=df.index)
    benford_expected = np.array([np.log10(1 + 1 / d) for d in range(1, 10)])

    def first_digit(arr):
        arr = np.abs(arr[arr > 0])
        if len(arr) == 0:
            return np.array([])
        digits = np.floor(arr / (10 ** np.floor(np.log10(arr))))
        return digits[(digits >= 1) & (digits <= 9)]

    def benford_chi2(series):
        digits = first_digit(series.values)
        if len(digits) < 3:
            return np.nan
        observed = np.array([np.sum(digits == d) for d in range(1, 10)])
        expected = benford_expected * len(digits)
        chi2 = np.sum(safe_div((observed - expected) ** 2, expected))
        return chi2

    def calc_company_benford(grp):
        n = len(grp)
        if n < 3:
            return pd.DataFrame({
                'f2_benford_chi2_revenue': [np.nan] * n,
                'f2_benford_chi2_profit': [np.nan] * n,
                'f2_benford_chi2_ocf': [np.nan] * n,
            }, index=grp.index)
        chi2_r = benford_chi2(grp['total_revenue'])
        chi2_p = benford_chi2(grp['net_profit'])
        chi2_c = benford_chi2(grp['operating_cash_flow'])
        return pd.DataFrame({
            'f2_benford_chi2_revenue': [chi2_r] * n,
            'f2_benford_chi2_profit': [chi2_p] * n,
            'f2_benford_chi2_ocf': [chi2_c] * n,
        }, index=grp.index)

    benford_df = df.groupby('company_code').apply(calc_company_benford).reset_index(level=0, drop=True)
    f['f2_benford_chi2_revenue'] = benford_df['f2_benford_chi2_revenue']
    f['f2_benford_chi2_profit'] = benford_df['f2_benford_chi2_profit']
    f['f2_benford_chi2_ocf'] = benford_df['f2_benford_chi2_ocf']

    chi2_cols = ['f2_benford_chi2_revenue', 'f2_benford_chi2_profit', 'f2_benford_chi2_ocf']
    f['f2_benford_max_dev'] = f[chi2_cols].max(axis=1)
    # 卡方检验临界值: chi2(8, alpha=0.05) = 15.507
    f['f2_benford_flag'] = (f['f2_benford_max_dev'] > 15.507).astype(int)

    return f


def calc_a5_trend_volatility(df):
    """A5: 趋势波动 10 维"""
    f = pd.DataFrame(index=df.index)
    grouped = df.groupby('company_code')

    f['f2_roe_qoq'] = grouped['roe'].diff()                              # ROE 环比变化
    f['f2_revenue_qoq'] = grouped['total_revenue'].diff()                # 营收环比变化
    f['f2_profit_qoq'] = grouped['net_profit'].diff()                    # 净利润环比变化
    f['f2_roe_yoy'] = grouped['roe'].pct_change(4)                       # ROE 同比(4期前)

    def calc_decline_streak(series):
        """连续环比下降期数"""
        s = series.values
        streak = np.zeros(len(s))
        for i in range(1, len(s)):
            if pd.notna(s[i]) and pd.notna(s[i - 1]) and s[i] < s[i - 1]:
                streak[i] = streak[i - 1] + 1
        return pd.Series(streak, index=series.index)

    f['f2_roe_decline_streak'] = grouped['roe'].transform(calc_decline_streak)
    f['f2_profit_decline_streak'] = grouped['net_profit'].transform(calc_decline_streak)

    f['f2_roe_volatility'] = grouped['roe'].transform(lambda x: x.rolling(4, min_periods=2).std())
    f['f2_profit_volatility'] = grouped['net_profit'].transform(lambda x: x.rolling(4, min_periods=2).std())
    f['f2_revenue_volatility'] = grouped['total_revenue'].transform(lambda x: x.rolling(4, min_periods=2).std())

    # 趋势恶化综合指标（5 个恶化信号的计数，0~5）
    f['f2_trend_deterioration'] = (
        (f['f2_roe_qoq'] < 0).astype(int) +
        (f['f2_revenue_qoq'] < 0).astype(int) +
        (f['f2_profit_qoq'] < 0).astype(int) +
        (f['f2_roe_decline_streak'] >= 2).astype(int) +
        (f['f2_profit_decline_streak'] >= 2).astype(int)
    )

    return f


def calc_a6_industry_deviation(df):
    """A6: 行业偏离 8 维（同报告期同行业 Z-score）"""
    f = pd.DataFrame(index=df.index)

    net_margin = pd.Series(safe_div(df['net_profit'], df['total_revenue']), index=df.index)

    metrics = OrderedDict([
        ('f2_z_roe', df['roe']),
        ('f2_z_net_margin', net_margin),
        ('f2_z_debt_ratio', df['debt_to_assets_ratio']),
        ('f2_z_revenue_yoy', df['revenue_yoy_growth']),
        ('f2_z_profit_yoy', df['net_profit_yoy_growth']),
        ('f2_z_pe', df['pe_ratio']),
        ('f2_z_pb', df['pb_ratio']),
    ])

    for col_name, col_data in metrics.items():
        mean_v = col_data.groupby([df['report_period'], df['industry']]).transform('mean')
        std_v = col_data.groupby([df['report_period'], df['industry']]).transform('std')
        f[col_name] = np.where(std_v > 0, (col_data - mean_v) / std_v, np.nan)

    z_cols = list(metrics.keys())
    # 行业离群指标计数：|Z| > 1.96 的指标个数(0~7)
    f['f2_industry_outlier_count'] = (f[z_cols].abs() > 1.96).sum(axis=1)

    return f


# ============================================================
# 3. 特征族定义 + 对外接口
# ============================================================

# 计算 F2 需要输入的原始财务字段（爬取 skill 必须保证提供这些列）
REQUIRED_COLUMNS = [
    'company_code', 'industry', 'report_period',
    'market_cap', 'pe_ratio', 'pb_ratio',
    'total_revenue', 'net_profit', 'operating_cash_flow',
    'roe', 'roa', 'debt_to_assets_ratio',
    'revenue_yoy_growth', 'net_profit_yoy_growth',
]

# 7 个特征族 -> 每个族包含的特征列名（顺序即输出顺序）
FEATURE_FAMILIES = OrderedDict([
    ("A1_盈利偿债能力", [
        'f2_roe', 'f2_roa', 'f2_net_margin', 'f2_debt_ratio',
        'f2_pe', 'f2_pb', 'f2_neg_pe_flag', 'f2_neg_pb_flag', 'f2_log_market_cap',
        'f2_market_cap_quintile', 'f2_ocf_to_revenue', 'f2_ocf_to_profit',
        'f2_roe_industry_rank', 'f2_loss_flag', 'f2_high_debt_flag',
    ]),
    ("A2_盈利质量", [
        'f2_accruals', 'f2_accruals_to_assets', 'f2_accruals_to_revenue', 'f2_ocf_to_assets',
        'f2_profit_ocf_diverge', 'f2_accrual_quality_zscore', 'f2_neg_accruals_flag',
        'f2_ocf_volatility_4q', 'f2_ocf_to_profit_extreme', 'f2_accruals_trend',
    ]),
    ("A3_Beneish", [
        'f2_gmi', 'f2_sgi', 'f2_lvgi', 'f2_tata', 'f2_beneish_m',
        'f2_dsri', 'f2_aqi', 'f2_depi', 'f2_sgai',
    ]),
    ("A3b_Piotroski", [
        'f2_p_roa', 'f2_p_cfo', 'f2_p_droa', 'f2_p_accrual', 'f2_p_dlever',
        'f2_p_dmargin', 'f2_p_dturnover', 'f2_p_score', 'f2_p_dliquid', 'f2_p_equity',
    ]),
    ("A4_Benford", [
        'f2_benford_chi2_revenue', 'f2_benford_chi2_profit', 'f2_benford_chi2_ocf',
        'f2_benford_max_dev', 'f2_benford_flag',
    ]),
    ("A5_趋势波动", [
        'f2_roe_qoq', 'f2_revenue_qoq', 'f2_profit_qoq', 'f2_roe_yoy',
        'f2_roe_decline_streak', 'f2_profit_decline_streak', 'f2_roe_volatility',
        'f2_profit_volatility', 'f2_revenue_volatility', 'f2_trend_deterioration',
    ]),
    ("A6_行业偏离", [
        'f2_z_roe', 'f2_z_net_margin', 'f2_z_debt_ratio', 'f2_z_revenue_yoy',
        'f2_z_profit_yoy', 'f2_z_pe', 'f2_z_pb', 'f2_industry_outlier_count',
    ]),
])

# 展平成有序的 67 维特征名列表
F2_FEATURE_NAMES = [c for cols in FEATURE_FAMILIES.values() for c in cols]


def _prepare(df_fin):
    """校验输入列 + 派生 report_date，供各特征族使用"""
    df = df_fin.copy()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必需字段: {missing}")

    # report_period 可能是 int(如 20200331) 或 str，统一转 str 再转日期
    df['report_period'] = df['report_period'].astype(str)
    df['report_date'] = pd.to_datetime(df['report_period'], format='%Y%m%d', errors='coerce')
    df['year'] = df['report_date'].dt.year
    df['quarter'] = df['report_date'].dt.quarter

    df = df.sort_values(['company_code', 'report_date']).reset_index(drop=True)
    return df


def compute_f2_features(df_fin):
    """计算 F2 特征（核心入口）。

    参数
    ----
    df_fin : DataFrame
        财务数据，必须包含 REQUIRED_COLUMNS 里列出的 14 个字段。
        可以是「全量批量数据」，也可以是「单公司/单行业切片」——传多少算多少。

    返回
    ----
    DataFrame
        元数据列(company_code/industry/report_period/report_date/year/quarter)
        + 67 维 f2_ 特征列。
    """
    df = _prepare(df_fin)

    a1 = calc_a1_profitability_solvency(df)
    a2 = calc_a2_earnings_quality(df)
    a3 = calc_a3_beneish(df)
    a3b = calc_a3b_piotroski(df)
    a4 = calc_a4_benford(df)
    a5 = calc_a5_trend_volatility(df)
    a6 = calc_a6_industry_deviation(df)

    f2 = pd.concat([
        df[['company_code', 'industry', 'report_period', 'report_date', 'year', 'quarter']],
        a1, a2, a3, a3b, a4, a5, a6,
    ], axis=1)

    return f2


def compute_f2_latest_for_company(df_fin, company_code):
    """计算单公司最新一期的 67 维 F2 特征（PIT：取最新报告期）。

    参数
    ----
    df_fin : DataFrame
        财务数据（至少包含该公司的历史序列；行业特征若要准确，需包含同行业公司）。
    company_code : str
        目标公司代码，如 '000004.SZ'。

    返回
    ----
    Series
        以 67 个 f2_ 特征名为 index 的特征向量（取该公司最新一期）。
        若该公司无数据，返回全 NaN 的 67 维向量。
    """
    f2_full = compute_f2_features(df_fin)
    company_f2 = f2_full[f2_full['company_code'] == company_code].sort_values('report_date')

    if len(company_f2) == 0:
        return pd.Series(np.nan, index=F2_FEATURE_NAMES)

    latest = company_f2.iloc[-1]
    return latest[F2_FEATURE_NAMES]


if __name__ == "__main__":
    # 自测：用东财在线数据跑一遍（F2 67 维）
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from backend.skills.financial_data_fetch import DataFetcher
    df = DataFetcher(rate_limit=0.5).fetch_financials("000004.SZ")
    f2 = compute_f2_features(df)
    print(f"F2 特征维度: {len(F2_FEATURE_NAMES)} 维")
    print(f"输出 shape: {f2.shape}")
    print(f2[['company_code', 'report_period'] + F2_FEATURE_NAMES[:5]].tail(3).to_string())
