# -*- coding: utf-8 -*-
"""
build_f2_financial_anomaly.py
=============================
根据「财务市场特征数据集」(wind_features_extracted.csv) 构建 F2 财务异常特征。

关键设计：Point-in-Time (PIT) 对齐
  每个 report_period (报告期) 的特征，使用「截至该报告期已披露的最新一期财报」的数值：
      report_period 0331 -> 上一自然年 0930 (Q3)
      report_period 0630 -> 同年 0331 (Q1)
      report_period 0930 -> 同年 0630 (Q2)
      report_period 1231 -> 同年 0930 (Q3)
  即避免「数据穿越 / look-ahead bias」。

特征分三类计算：
  1) 直接特征  : 原始字段 PIT 对齐后取值 / 比值
  2) 时序特征  : 在原始时间序列上计算 (qoq/波动率/连续下滑等)，再 PIT 对齐
  3) 截面特征  : 在 PIT 对齐后的截面上做行业内 rank / z-score

输出列结构与参考 F2_financial_anomaly.csv 完全一致 (70 列)。
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent          # 脚本所在目录 (参考/输出文件在此)
DATA_ROOT = BASE.parent.parent                  # d:\BaiduNetdiskDownload (原始数据集目录)
WIND_CSV = DATA_ROOT / "财务市场特征数据集" / "财务市场特征数据集" / "wind_features_extracted.csv"
LABELS_CSV = DATA_ROOT / "标签与评测数据集" / "标签与评测数据集" / "dataset_split_labels.csv"
OUT_CSV = BASE / "output" / "F2_financial_anomaly.csv"
OUT_XLSX = BASE / "output" / "F2_financial_anomaly.xlsx"
OUT_PARQUET = BASE / "output" / "F2_financial_anomaly.parquet"

# 70 列，严格对齐参考文件列顺序
COLUMNS = [
    "company_code", "report_period", "split",
    "f2_roe", "f2_roa", "f2_debt_ratio", "f2_pe", "f2_pb", "f2_net_margin",
    "f2_log_market_cap", "f2_ocf_to_revenue", "f2_ocf_to_profit", "f2_roe_industry_rank",
    "f2_loss_flag", "f2_neg_pe_flag", "f2_neg_pb_flag", "f2_high_debt_flag",
    "f2_market_cap_quintile", "f2_accruals", "f2_accruals_to_assets",
    "f2_accruals_to_revenue", "f2_ocf_to_assets", "f2_profit_ocf_diverge",
    "f2_accrual_quality_zscore", "f2_neg_accruals_flag", "f2_ocf_volatility_4q",
    "f2_ocf_to_profit_extreme", "f2_accruals_trend", "f2_gmi", "f2_sgi", "f2_lvgi",
    "f2_tata", "f2_beneish_m", "f2_dsri", "f2_aqi", "f2_depi", "f2_sgai",
    "f2_p_roa", "f2_p_cfo", "f2_p_droa", "f2_p_accrual", "f2_p_dlever",
    "f2_p_dmargin", "f2_p_dturnover", "f2_p_score", "f2_p_dliquid", "f2_p_equity",
    "f2_benford_chi2_revenue", "f2_benford_chi2_profit", "f2_benford_chi2_ocf",
    "f2_benford_max_dev", "f2_benford_flag", "f2_roe_qoq", "f2_revenue_qoq",
    "f2_profit_qoq", "f2_roe_yoy", "f2_roe_decline_streak", "f2_profit_decline_streak",
    "f2_roe_volatility", "f2_profit_volatility", "f2_revenue_volatility",
    "f2_trend_deterioration", "f2_z_roe", "f2_z_net_margin", "f2_z_debt_ratio",
    "f2_z_revenue_yoy", "f2_z_profit_yoy", "f2_z_pe", "f2_z_pb",
    "f2_industry_outlier_count",
]

HIGH_DEBT_THRESHOLD = 70.0
OCF_PROFIT_EXTREME = 5.0


def pit_source(report_period: str) -> str:
    """返回该报告期对应的「已披露最新财报期」(PIT 源期)。"""
    y = int(report_period[:4])
    md = report_period[4:]
    return {
        "0331": f"{y - 1}0930",
        "0630": f"{y}0331",
        "0930": f"{y}0630",
        "1231": f"{y}0930",
    }.get(md, report_period)


def pit_align(df: pd.DataFrame, value_col: str) -> pd.Series:
    """把「原始期」上的 value_col 对齐到 report_period (PIT)。"""
    src = df[["company_code", "report_period", value_col]].rename(
        columns={value_col: "__v"}
    )
    target = df[["company_code", "report_period"]].copy()
    target["__src"] = target["report_period"].map(pit_source)
    out = target.merge(
        src, left_on=["company_code", "__src"], right_on=["company_code", "report_period"],
        how="left",
    )
    return out["__v"].astype(float).to_numpy()


def industry_zscore(s: pd.Series) -> pd.Series:
    mean = s.mean()
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - mean) / std


def benford_chi2(vals: np.ndarray) -> float:
    """对一组数值的首位数字做 Benford 卡方检验，返回卡方统计量。"""
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals) & (np.abs(vals) > 0)]
    if len(vals) < 8:
        return np.nan
    first = pd.to_numeric(pd.Series(np.abs(vals)).astype(str).str[0], errors="coerce")
    first = first[(first >= 1) & (first <= 9)].dropna()
    if len(first) < 8:
        return np.nan
    n = len(first)
    obs = first.value_counts().reindex(range(1, 10), fill_value=0).to_numpy()
    exp = np.array([np.log10(1 + 1 / d) for d in range(1, 10)]) * n
    return float(((obs - exp) ** 2 / exp).sum())


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    wind = pd.read_csv(WIND_CSV, encoding="utf-8-sig")
    wind.columns = wind.columns.str.strip()
    labels = pd.read_csv(LABELS_CSV).rename(columns={"secucode": "company_code"})

    # 排序 + 类型清洗
    wind = wind.sort_values(["company_code", "report_period"]).reset_index(drop=True)
    wind["report_period"] = wind["report_period"].astype(str)
    for c in ["market_cap", "pe_ratio", "pb_ratio", "total_revenue", "net_profit",
              "operating_cash_flow", "roe", "roa", "debt_to_assets_ratio",
              "revenue_yoy_growth", "net_profit_yoy_growth"]:
        wind[c] = pd.to_numeric(wind[c], errors="coerce")

    cc = wind["company_code"]
    mc = wind["market_cap"]; pe = wind["pe_ratio"]; pb = wind["pb_ratio"]
    rev = wind["total_revenue"]; nprof = wind["net_profit"]; ocf = wind["operating_cash_flow"]
    roe = wind["roe"]; roa = wind["roa"]; dar = wind["debt_to_assets_ratio"]
    ryg = wind["revenue_yoy_growth"]; npyg = wind["net_profit_yoy_growth"]

    # ---- 派生基础量 ----
    total_assets = nprof / roa * 100.0       # 总资产近似 (roa 为百分比)
    accruals = nprof - ocf                    # 应计利润
    net_margin = nprof / rev

    wind["__ta"] = total_assets
    wind["__accr"] = accruals
    wind["__accr_ta"] = accruals / total_assets
    wind["__nm"] = net_margin

    # 按公司分组
    g = wind.groupby("company_code", sort=False)

    # ---- 直接特征 (PIT 对齐) ----
    feats = pd.DataFrame()
    feats["company_code"] = cc
    feats["report_period"] = wind["report_period"].astype(int)

    feats["f2_roe"] = pit_align(wind, "roe")
    feats["f2_roa"] = pit_align(wind, "roa")
    feats["f2_debt_ratio"] = pit_align(wind, "debt_to_assets_ratio")
    feats["f2_pe"] = pit_align(wind, "pe_ratio")
    feats["f2_pb"] = pit_align(wind, "pb_ratio")
    feats["f2_net_margin"] = pit_align(wind, "__nm")
    feats["f2_log_market_cap"] = np.log(pit_align(wind, "market_cap"))
    feats["f2_ocf_to_revenue"] = pit_align(wind, "operating_cash_flow") / pit_align(wind, "total_revenue")
    feats["f2_ocf_to_profit"] = pit_align(wind, "operating_cash_flow") / pit_align(wind, "net_profit")
    feats["f2_accruals"] = pit_align(wind, "__accr")
    feats["f2_accruals_to_assets"] = pit_align(wind, "__accr_ta")
    feats["f2_accruals_to_revenue"] = pit_align(wind, "__accr") / pit_align(wind, "total_revenue")
    feats["f2_ocf_to_assets"] = pit_align(wind, "operating_cash_flow") / pit_align(wind, "__ta")
    feats["f2_tata"] = pit_align(wind, "__accr_ta")

    # ---- 标志位 (PIT 对齐) ----
    feats["f2_loss_flag"] = (pit_align(wind, "net_profit") < 0).astype(float)
    feats["f2_neg_pe_flag"] = (pit_align(wind, "pe_ratio") < 0).astype(float)
    feats["f2_neg_pb_flag"] = (pit_align(wind, "pb_ratio") < 0).astype(float)
    feats["f2_high_debt_flag"] = (pit_align(wind, "debt_to_assets_ratio") > HIGH_DEBT_THRESHOLD).astype(float)
    feats["f2_neg_accruals_flag"] = (pit_align(wind, "__accr") < 0).astype(float)
    _np = pit_align(wind, "net_profit"); _ocf = pit_align(wind, "operating_cash_flow")
    feats["f2_profit_ocf_diverge"] = (((_np > 0) & (_ocf < 0)) | ((_np < 0) & (_ocf > 0))).astype(float)
    feats["f2_ocf_to_profit_extreme"] = (np.abs(_ocf / _np) > OCF_PROFIT_EXTREME).astype(float)

    # ---- 截面特征 (PIT 对齐后, 按 report_period+industry) ----
    feats["f2_roe_industry_rank"] = feats["f2_roe"].groupby(
        [wind["report_period"], wind["industry"]]
    ).rank(pct=True, method="average")
    feats["f2_market_cap_quintile"] = np.ceil(
        pd.Series(pit_align(wind, "market_cap")).groupby(wind["report_period"]).rank(pct=True) * 5
    )

    # ---- 时序特征 (原始序列计算, 再 PIT 对齐) ----
    wind["__roe_qoq"] = g["roe"].diff()
    wind["__rev_qoq"] = g["total_revenue"].diff()
    wind["__np_qoq"] = g["net_profit"].diff()
    wind["__roe_yoy"] = g["roe"].pct_change(4)
    wind["__roe_vol"] = g["roe"].transform(lambda s: s.rolling(4, min_periods=1).std())
    wind["__np_vol"] = g["net_profit"].transform(lambda s: s.rolling(4, min_periods=1).std())
    wind["__rev_vol"] = g["total_revenue"].transform(lambda s: s.rolling(4, min_periods=1).std())
    wind["__ocf_vol4"] = g["operating_cash_flow"].transform(lambda s: s.rolling(4, min_periods=1).std())
    wind["__sgi"] = rev / g["total_revenue"].shift(1)
    wind["__lvgi"] = dar / g["debt_to_assets_ratio"].shift(1)
    wind["__gmi"] = g["__nm"].shift(1) / wind["__nm"]
    wind["__accr_trend"] = g["__accr_ta"].diff(4)
    # 连续下滑
    def _streak(down: pd.Series) -> np.ndarray:
        return (
            down.groupby(wind["company_code"])
            .apply(lambda x: x.groupby((~x).cumsum()).cumsum().clip(upper=3))
            .astype(float).to_numpy()
        )
    wind["__roe_streak"] = _streak(g["roe"].diff() < 0)
    wind["__np_streak"] = _streak(g["net_profit"].diff() < 0)

    for src, dst in [
        ("__roe_qoq", "f2_roe_qoq"), ("__rev_qoq", "f2_revenue_qoq"),
        ("__np_qoq", "f2_profit_qoq"), ("__roe_yoy", "f2_roe_yoy"),
        ("__roe_vol", "f2_roe_volatility"), ("__np_vol", "f2_profit_volatility"),
        ("__rev_vol", "f2_revenue_volatility"), ("__ocf_vol4", "f2_ocf_volatility_4q"),
        ("__sgi", "f2_sgi"), ("__lvgi", "f2_lvgi"), ("__gmi", "f2_gmi"),
        ("__accr_trend", "f2_accruals_trend"), ("__roe_streak", "f2_roe_decline_streak"),
        ("__np_streak", "f2_profit_decline_streak"),
    ]:
        feats[dst] = pit_align(wind, src)

    # ---- Beneish M-score (DSRI/AQI/DEPI/SGAI 因缺字段置 0) ----
    feats["f2_dsri"] = 0
    feats["f2_aqi"] = 0
    feats["f2_depi"] = 0
    feats["f2_sgai"] = 0
    feats["f2_beneish_m"] = (
        -4.84 + 0.528 * feats["f2_gmi"] + 0.892 * feats["f2_sgi"]
        + 4.679 * feats["f2_tata"] - 0.327 * feats["f2_lvgi"]
    )

    # ---- Piotroski F-score ----
    _roa = pit_align(wind, "roa"); _ocf2 = pit_align(wind, "operating_cash_flow")
    _np2 = pit_align(wind, "net_profit")
    feats["f2_p_roa"] = (_roa > 0).astype(float)
    feats["f2_p_cfo"] = (_ocf2 > 0).astype(float)
    # p_droa: roa 较上期是否上升 (原始序列)
    wind["__roa_diff_flag"] = (g["roa"].diff() > 0).astype(float)
    wind["__dar_diff_flag"] = (g["debt_to_assets_ratio"].diff() < 0).astype(float)
    wind["__nm_diff_flag"] = (g["__nm"].diff() > 0).astype(float)
    turnover = rev / wind["__ta"]
    wind["__turnover_diff_flag"] = (turnover.groupby(cc).diff() > 0).astype(float)
    feats["f2_p_droa"] = pit_align(wind, "__roa_diff_flag")
    feats["f2_p_accrual"] = (_ocf2 > _np2).astype(float)
    feats["f2_p_dlever"] = pit_align(wind, "__dar_diff_flag")
    feats["f2_p_dmargin"] = pit_align(wind, "__nm_diff_flag")
    feats["f2_p_dturnover"] = pit_align(wind, "__turnover_diff_flag")
    feats["f2_p_dliquid"] = 0.5
    feats["f2_p_equity"] = 0.5
    feats["f2_p_score"] = (
        feats["f2_p_roa"] + feats["f2_p_cfo"] + feats["f2_p_droa"]
        + feats["f2_p_accrual"] + feats["f2_p_dlever"] + feats["f2_p_dmargin"]
        + feats["f2_p_dturnover"]
    )

    # ---- Benford (公司全历史首位数字卡方) ----
    for base, dst in [("total_revenue", "f2_benford_chi2_revenue"),
                      ("net_profit", "f2_benford_chi2_profit"),
                      ("operating_cash_flow", "f2_benford_chi2_ocf")]:
        wind[f"__b_{base}"] = wind.groupby("company_code")[base].transform(
            lambda s: benford_chi2(s.to_numpy())
        )
        feats[dst] = pit_align(wind, f"__b_{base}")
    feats["f2_benford_max_dev"] = feats[["f2_benford_chi2_revenue",
                                          "f2_benford_chi2_profit",
                                          "f2_benford_chi2_ocf"]].max(axis=1)
    feats["f2_benford_flag"] = (feats["f2_benford_max_dev"] > 15.5).astype(float)

    # ---- 趋势恶化 (0-5, 下滑信号计数) ----
    feats["f2_trend_deterioration"] = (
        (feats["f2_roe_qoq"] < 0).astype(float) + (feats["f2_revenue_qoq"] < 0).astype(float)
        + (feats["f2_profit_qoq"] < 0).astype(float) + (feats["f2_roe_yoy"] < 0).astype(float)
        + (feats["f2_net_margin"] < 0).astype(float)
    )

    # ---- 截面 z-score (按 report_period + industry) ----
    feats["f2_accrual_quality_zscore"] = feats["f2_accruals_to_assets"].groupby(
        [wind["report_period"], wind["industry"]]
    ).transform(industry_zscore)
    for col, dst in [("f2_roe", "f2_z_roe"), ("f2_net_margin", "f2_z_net_margin"),
                     ("f2_debt_ratio", "f2_z_debt_ratio")]:
        feats[dst] = feats[col].groupby([wind["report_period"], wind["industry"]]).transform(industry_zscore)
    for src, dst in [("revenue_yoy_growth", "f2_z_revenue_yoy"),
                     ("net_profit_yoy_growth", "f2_z_profit_yoy"),
                     ("pe_ratio", "f2_z_pe"), ("pb_ratio", "f2_z_pb")]:
        feats[dst] = pit_align(wind, src)
        feats[dst] = feats[dst].groupby([wind["report_period"], wind["industry"]]).transform(industry_zscore)

    # ---- 行业异常点计数 (|z|>2 的个数, 3 个 z 指标) ----
    feats["f2_industry_outlier_count"] = (
        (np.abs(feats["f2_z_roe"]) > 2).astype(float)
        + (np.abs(feats["f2_z_debt_ratio"]) > 2).astype(float)
        + (np.abs(feats["f2_z_net_margin"]) > 2).astype(float)
    )

    # ---- split 标签 ----
    feats = feats.merge(
        labels[["company_code", "split"]], on="company_code", how="left"
    )

    # 统一列序 + 无穷改 NaN
    feats = feats[COLUMNS]
    feats = feats.replace([np.inf, -np.inf], np.nan)

    feats.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    feats.to_excel(OUT_XLSX, index=False)
    feats.to_parquet(OUT_PARQUET, index=False)

    # 报告
    ref = pd.read_csv(BASE / "F2_financial_anomaly.csv")
    ref = ref.sort_values(["company_code", "report_period"]).reset_index(drop=True)
    feats = feats.sort_values(["company_code", "report_period"]).reset_index(drop=True)
    print(f"输出: {OUT_CSV}  ({len(feats):,} 行 x {len(COLUMNS)} 列)")
    print("\n与参考文件逐列匹配率 (数值容差 1e-6):")
    for c in COLUMNS[3:]:
        a = feats[c].to_numpy(float); b = ref[c].to_numpy(float)
        mm = ~(np.isnan(a) | np.isnan(b))
        if mm.sum() == 0:
            print(f"  {c:32s} 无重叠")
            continue
        ex = np.isclose(a[mm], b[mm], rtol=0, atol=1e-6, equal_nan=True).mean() * 100
        print(f"  {c:32s} {ex:6.2f}%")


if __name__ == "__main__":
    main()
