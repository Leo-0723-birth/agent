#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
财务异常检测 Agent (FinancialDetectorAgent) —— 任务2 的特征工程
================================================================
职责：输入公司代码 → 爬取财务指标（东方财富免费接口，演示数据源）→ 行业对标 Z-Score
      → 规则异常检测（含双负信号兜底）→ 输出结构化财务异常到共享 context。
输出（写回 ctx.financial）：
    features       F2 特征（后续特征组装使用）
    indicators     原始指标（含 report_period）
    benchmarks     行业对标 Z-Score（可选，依赖 config.FIN_WIND_CSV）
    anomaly_list   异常清单（type/severity/indicator/value/threshold/evidence/label_ref）
    risk_level     低/中/高/跳过
    skip           特殊行业 / 无数据时跳过财务分析

数据源说明（重要）：
    - 本 Agent 面向【评委演示】，数据源 = 东方财富免费接口（实时真实数据）
    - 比赛模型训练/测试使用官方数据集（另走 scripts/train_predictor.py），与本 Agent 解耦

设计：纯规则 + 可选 LLM 解读（backend/llm.py，模型 deepseek-v4-flash）。
参考：底层建模方案 2.3/2.4 + 桌面《财务异常agent》已有实现（已跑通）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, AttributeError, OSError):
        pass

from ..config import (
    FIN_CF_TO_PROFIT,
    FIN_DEBT_RATIO_MAX,
    FIN_ROE_NEGATIVE,
    FIN_ROE_TREND_SLOPE,
    FIN_WIND_CSV,
    FIN_Z_SCORE,
)
from ..llm import chat, chat_json
from ..skills import f2_calc, market_fetch
from ..skills.financial_data_fetch import DataFetcher
from ..skills.stock_code import StockCodeError, normalize_stock_code
from .base import AgentBase

# 特殊行业的财务特点（高杠杆/金融属性；仍参与常规检测，供审计参考）
# 现阶段策略：金融/地产/建筑均参与常规财务异常检测（不跳过）
SPECIAL_INDUSTRY_PROFILES = {
    "金融业": {
        "特点": "高杠杆经营，资产负债率天然 >90%，常规负债率阈值需结合不良率、拨备覆盖率、资本充足率综合解读",
        "跳过分析": False,
    },
    "房地产业": {
        "特点": "高杠杆高负债，负债率 70%-90% 属行业常态；需结合现金流、去化率、融资成本、有息负债结构综合解读",
        "跳过分析": False,
    },
    "建筑业": {
        "特点": "垫资施工模式，应收账款和负债率偏高；核心看工程回款、垫资比例",
        "跳过分析": False,
    },
}


def _map_columns(df):
    """把中文列名映射成英文（wind 特征 CSV 用）。"""
    rename = {"company_code": "company_code", "industry": "industry",
              "report_period": "report_period"}
    for c in df.columns:
        if "market_cap" in c:
            rename[c] = "market_cap"
        elif "pe_ratio" in c:
            rename[c] = "pe_ratio"
        elif "pb_ratio" in c:
            rename[c] = "pb_ratio"
        elif "total_revenue" in c:
            rename[c] = "total_revenue"
        elif c.startswith("net_profit("):
            rename[c] = "net_profit"
        elif "operating_cash" in c:
            rename[c] = "operating_cash_flow"
        elif c.startswith("roe("):
            rename[c] = "roe"
        elif c.startswith("roa("):
            rename[c] = "roa"
        elif "debt_to_assets" in c:
            rename[c] = "debt_to_assets_ratio"
        elif "revenue_yoy" in c:
            rename[c] = "revenue_yoy_growth"
        elif "net_profit_yoy" in c:
            rename[c] = "net_profit_yoy_growth"
    return df.rename(columns=rename)


def _safe_float(v):
    try:
        if pd.isna(v):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _ratio(a, b):
    if a is not None and b not in (None, 0):
        return round(a / b, 4)
    return None


def _risk_level(n):
    if n >= 3:
        return "高"
    elif n >= 1:
        return "中"
    return "低"


class FinancialDetectorAgent(AgentBase):
    name = "FinancialDetector"

    def __init__(self, use_llm=True, rate_limit=0.5, wind_csv_path=None):
        super().__init__()
        self.fetcher = DataFetcher(rate_limit=rate_limit)
        self.use_llm = use_llm
        self._wind_csv = wind_csv_path or (FIN_WIND_CSV or "")
        self._wind_df = None   # 懒加载

    # ================= Skill 1: 财务指标计算 =================
    def financial_indicator_calc(self, df):
        """从爬取的财务 DataFrame 取最新一期，构造 indicators（含派生指标）。"""
        latest = df.iloc[-1]
        indicators = {
            "report_period": int(latest["report_period"]),
            "total_revenue": _safe_float(latest.get("total_revenue")),
            "net_profit": _safe_float(latest.get("net_profit")),
            "operating_cash_flow": _safe_float(latest.get("operating_cash_flow")),
            "roe": _safe_float(latest.get("roe")),
            "roa": _safe_float(latest.get("roa")),
            "debt_to_assets_ratio": _safe_float(latest.get("debt_to_assets_ratio")),
            "revenue_yoy_growth": _safe_float(latest.get("revenue_yoy_growth")),
            "net_profit_yoy_growth": _safe_float(latest.get("net_profit_yoy_growth")),
        }
        indicators["cf_to_profit"] = _ratio(
            indicators["operating_cash_flow"], indicators["net_profit"])
        indicators["roe_trend_4q"] = self._roe_trend(df)
        return indicators

    def _roe_trend(self, df):
        """ROE 最近 4 期趋势斜率（连续下滑检测）。"""
        roe = df["roe"].dropna().tail(4).values
        if len(roe) < 2:
            return 0.0
        slope = np.polyfit(np.arange(len(roe)), roe, 1)[0]
        return round(float(slope), 4)

    # ================= Skill 2: 行业对标 Z-Score =================
    def _load_wind(self):
        """懒加载 wind 特征 CSV（同报告期同行对标样本）。未配置则返回 None。"""
        if self._wind_df is not None:
            return self._wind_df
        if self._wind_csv and Path(self._wind_csv).exists():
            raw = pd.read_csv(self._wind_csv)
            self._wind_df = _map_columns(raw)
            self._wind_df["report_period"] = self._wind_df["report_period"].astype(int)
        else:
            self._wind_df = None
        return self._wind_df

    def industry_benchmark(self, company_code, indicators):
        """
        行业对标 Z-Score（口径修正：优先用【同一报告期】的同行，
        同行不足时退回全部报告期；未配置样本则优雅降级）。
        """
        wind = self._load_wind()
        if wind is None or len(wind) == 0:
            return {"note": "无行业对标样本（config.FIN_WIND_CSV 未配置）"}

        industry = indicators.get("industry")
        if not industry:
            return {"note": "缺少行业信息"}
        rp = indicators.get("report_period")

        peers = wind[wind["industry"] == industry]
        same_period = peers[peers["report_period"] == rp] if rp is not None else peers
        if len(same_period) >= 3:
            peers = same_period
        if len(peers) < 2:
            return {"note": f"行业[{industry}]同行样本不足", "industry_peer_count": 0}

        benchmarks = {}
        for col in ["roe", "roa", "debt_to_assets_ratio", "revenue_yoy_growth",
                    "net_profit_yoy_growth"]:
            vals = peers[col].dropna()
            if len(vals) < 2 or vals.std() == 0 or indicators.get(col) is None:
                benchmarks[f"{col}_zscore"] = None
                continue
            z = (indicators[col] - vals.mean()) / vals.std()
            benchmarks[f"{col}_zscore"] = round(float(z), 3)
        benchmarks["industry_peer_count"] = int(peers["company_code"].nunique())
        benchmarks["peer_period"] = "same_period" if same_period is peers else "all_periods"
        return benchmarks

    # ================= Skill 3: 规则异常检测（含双负兜底） =================
    def anomaly_detect(self, indicators, benchmarks):
        """
        规则异常检测，返回异常列表（新格式：供归因 Agent 直接使用）。
        双负兜底：净利润<0 且 经营现金流<0 时，cf_to_profit 比值失真（负/负=正），
        旧规则"现金流背离"不触发 → 直接给"双负信号"。
        """
        anomalies = []
        cf = indicators.get("cf_to_profit")
        np_ = indicators.get("net_profit")
        ocf = indicators.get("operating_cash_flow")

        # --- 现金流（含双负兜底） ---
        if cf is not None and np_ is not None and ocf is not None:
            if np_ < 0 and ocf < 0:
                anomalies.append({
                    "type": "双负信号",
                    "severity": 4,
                    "indicator": "cf_income_ratio",
                    "value": cf,
                    "threshold": "net_profit<0 且 operating_cash_flow<0（比值失真，直接预警）",
                    "evidence": f"净利润 {np_/1e4:.0f}万 与经营现金流 {ocf/1e4:.0f}万 均为负，盈利质量严重恶化",
                    "label_ref": "盈利质量",
                })
            elif cf < FIN_CF_TO_PROFIT:
                anomalies.append({
                    "type": "现金流背离",
                    "severity": 3,
                    "indicator": "cf_income_ratio",
                    "value": cf,
                    "threshold": f"< {FIN_CF_TO_PROFIT}",
                    "evidence": f"经营现金流/净利润 = {cf:.2f} < {FIN_CF_TO_PROFIT}，利润质量存疑",
                    "label_ref": "盈利质量",
                })

        # --- 高负债 ---
        if indicators.get("debt_to_assets_ratio") is not None and \
                indicators["debt_to_assets_ratio"] > FIN_DEBT_RATIO_MAX:
            anomalies.append({
                "type": "高负债",
                "severity": 3,
                "indicator": "debt_to_assets_ratio",
                "value": indicators["debt_to_assets_ratio"],
                "threshold": f"> {FIN_DEBT_RATIO_MAX}%",
                "evidence": f"资产负债率 = {indicators['debt_to_assets_ratio']:.1f}% > {FIN_DEBT_RATIO_MAX}%",
                "label_ref": "偿债能力",
            })

        # --- 亏损 ---
        if indicators.get("roe") is not None and indicators["roe"] < FIN_ROE_NEGATIVE:
            anomalies.append({
                "type": "亏损",
                "severity": 4,
                "indicator": "roe",
                "value": indicators["roe"],
                "threshold": f"< {FIN_ROE_NEGATIVE}",
                "evidence": f"ROE = {indicators['roe']:.2f}% < {FIN_ROE_NEGATIVE}，净资产亏损",
                "label_ref": "盈利能力",
            })

        # --- 盈利持续恶化 ---
        if indicators.get("roe_trend_4q") is not None and \
                indicators["roe_trend_4q"] < FIN_ROE_TREND_SLOPE:
            anomalies.append({
                "type": "盈利持续恶化",
                "severity": 4,
                "indicator": "roe_trend_4q",
                "value": indicators["roe_trend_4q"],
                "threshold": f"< {FIN_ROE_TREND_SLOPE}",
                "evidence": f"ROE 近4期趋势斜率 = {indicators['roe_trend_4q']:.1f} 个百分点/季，连续下滑",
                "label_ref": "盈利能力",
            })

        # --- 行业偏离（|Z| > 阈值） ---
        for k, v in benchmarks.items():
            if k.endswith("_zscore") and v is not None and abs(v) > FIN_Z_SCORE:
                anomalies.append({
                    "type": "行业偏离",
                    "severity": 2,
                    "indicator": k.replace("_zscore", ""),
                    "value": v,
                    "threshold": f"|Z| > {FIN_Z_SCORE}",
                    "evidence": f"{k.replace('_zscore','')} Z-Score = {v}，偏离行业均值超过{FIN_Z_SCORE}σ",
                    "label_ref": "行业对标",
                })
        return anomalies

    # ================= Skill 3b: F2 特征异常规则（对齐官方标签体系） =================
    def f2_anomaly_rules(self, f2):
        """基于 F2 数值特征的异常规则（label_ref 对齐任务1官方标签 A-H 体系）。"""
        anomalies = []
        m = f2.get("f2_beneish_m")
        if m is not None and m > -2.22:
            anomalies.append({
                "type": "盈余操纵嫌疑",
                "severity": 4,
                "indicator": "f2_beneish_m",
                "value": m,
                "threshold": "> -2.22（Beneish M-Score 可疑线）",
                "evidence": f"Beneish M-Score = {m:.2f} > -2.22，存在盈余操纵嫌疑",
                "label_ref": "盈利能力",
            })
        if f2.get("f2_benford_flag") == 1:
            anomalies.append({
                "type": "数字分布异常",
                "severity": 3,
                "indicator": "f2_benford_flag",
                "value": 1,
                "threshold": "Benford 卡方 > 15.507",
                "evidence": f"Benford 最大偏离卡方 = {f2.get('f2_benford_max_dev')}，超过临界值 15.507",
                "label_ref": "财务异常",
            })
        if (f2.get("f2_trend_deterioration") or 0) >= 4:
            anomalies.append({
                "type": "趋势恶化",
                "severity": 3,
                "indicator": "f2_trend_deterioration",
                "value": f2.get("f2_trend_deterioration"),
                "threshold": ">= 4（5 个恶化信号中至少 4 个）",
                "evidence": f"趋势恶化综合指标 = {f2.get('f2_trend_deterioration')}"
                            "（ROE/营收/利润环比下滑等 5 信号）",
                "label_ref": "盈利能力",
            })
        return anomalies

    # ================= Skill 4: 输入解析（LLM 可选） =================
    def _resolve_company(self, user_input):
        """把名称/简称/代码解析为标准 secucode（用共享 LLM）。失败返回空 dict。"""
        prompt = f"""你是A股上市公司识别助手。请识别用户输入所指的公司，输出 JSON：
{{"secucode": "6位代码.交易所后缀", "company_name": "公司简称", "matched": true/false}}
规则：6/9开头（92除外）→.SH，0/2/3开头→.SZ，4/8/92开头→.BJ；无法识别 matched=false。
用户输入：{user_input}"""
        try:
            return chat_json("", prompt, max_tokens=200)
        except Exception:
            return {}

    # ================= Skill 5: LLM 财务解读（可选） =================
    def _llm_analyze(self, company_name, indicators, anomalies):
        """用共享 LLM 对异常做自然语言解读。失败返回空串（不打断流程）。"""
        anomalies_text = "\n".join(
            [f"- {a['type']}（严重度{a['severity']}）: {a['evidence']}" for a in anomalies])
        prompt = (
            f"你是资深财务风控分析师。以下是上市公司 {company_name} 的财务异常检测结果，"
            f"请用 3-5 句话概括其财务风险，并给出监管关注可能性判断。\n\n"
            f"财务指标：ROE {indicators.get('roe')}%，ROA {indicators.get('roa')}%，"
            f"资产负债率 {indicators.get('debt_to_assets_ratio')}%，"
            f"营收同比 {indicators.get('revenue_yoy_growth')}%，"
            f"现金流/净利润 {indicators.get('cf_to_profit')}\n\n"
            f"检测到的异常：\n{anomalies_text}\n\n请直接输出分析结论："
        )
        try:
            return chat("你是资深财务风控分析师。", prompt, temperature=0.3, max_tokens=500, json_mode=False)
        except Exception as e:
            return f"[LLM 分析失败] {e}"

    # ================= 主入口（统一签名） =================
    def execute(self, company, ctx):
        """统一签名：读 ctx（company/name/window），写回 ctx.financial。"""
        raw_company = str(company or "").strip()
        # 1. 输入解析：股票代码统一规范；纯名称仅在启用 LLM 时解析。
        try:
            code = normalize_stock_code(raw_company)
        except StockCodeError:
            if not self.use_llm or any(character.isdigit() for character in raw_company):
                raise
            resolved = self._resolve_company(raw_company)
            # 兼容只返回 secucode 的旧模型响应；只有明确 matched=false 才拒绝。
            if resolved.get("matched") is False or not resolved.get("secucode"):
                raise StockCodeError(f"无法把公司名称“{raw_company}”解析为上市公司股票代码。")
            code = normalize_stock_code(resolved["secucode"])
            ctx.name = resolved.get("company_name") or ctx.name
        ctx.company = code

        # 2. 公司资料（行业判断）
        profile = self.fetcher.fetch_company_profile(code)
        industry = profile.get("industry") if profile else None
        ctx.financial.industry = industry or ""

        characteristics, skip = self._industry_characteristics(industry)
        if skip:
            ctx.financial.skip = True
            ctx.financial.skip_reason = "特殊行业，不适用常规财务异常检测"
            ctx.financial.risk_level = "跳过"
            return ctx

        # 3. 爬取财务指标（最新一期）
        df = self.fetcher.fetch_financials(code)
        if df is None or len(df) == 0:
            ctx.financial.skip = True
            ctx.financial.skip_reason = "财务数据获取失败（网络受限或无数据）"
            ctx.financial.risk_level = "跳过"
            return ctx

        indicators = self.financial_indicator_calc(df)
        indicators["industry"] = ctx.financial.industry

        # 3.5 F2 完整特征（67 维，队友 calc_f2_features 迁移；失败降级空）
        f2_features = {}
        try:
            f2_df = f2_calc.compute_f2_features(df)
            f2_latest = f2_df[f2_df["company_code"] == code].sort_values("report_date")
            if len(f2_latest):
                f2_features = {k: _safe_float(v) for k, v in
                               f2_latest.iloc[-1][f2_calc.F2_FEATURE_NAMES].items()}
        except Exception as e:
            print(f"  [F2 计算失败] {code}: {e}")

        # 3.6 F3 市场特征（35 维在线，队友 crawl_market 迁移；失败降级空）
        mkt_features = {}
        try:
            mkt_features = {k: _safe_float(v) for k, v in
                            market_fetch.crawl_market_features(code).items()}
        except Exception as e:
            print(f"  [F3 抓取失败] {code}: {e}")

        # 3.7 F4/F5 特征（在线爬取优先 → 离线预处理表兜底，在线带超时保护）
        #     在线：股吧舆情(F4)/股东治理(F5)，保证拿到近日最新数据；
        #     超时/失败/公司不在线时回退离线预处理表（训练同源，取最新一期）。
        #     F6 监管问询函特征：由【公告研读 Agent】从巨潮公告计算（见
        #     backend/skills/inquiry_features.py），财务侧不输出 F6。
        f456_features = {}
        try:
            from ..skills.crawl_sentiment import crawl_sentiment_features
            from ..skills.crawl_governance import crawl_governance_features
            online_crawlers = {
                "F4": crawl_sentiment_features,
                "F5": crawl_governance_features,
            }
            from ..skills.feature_loader import load_latest_features
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout

            def _run_with_timeout(fn, *args, timeout=30):
                with ThreadPoolExecutor(max_workers=1) as ex:
                    return ex.submit(fn, *args).result(timeout=timeout)

            for fam, crawler in online_crawlers.items():
                feats = None
                try:
                    feats = _run_with_timeout(crawler, code, timeout=30)   # 在线：近日最新数据
                except FutTimeout:
                    print(f"  [{fam} 在线抓取超时(30s)，回退离线表] {code}")
                except Exception as e:
                    print(f"  [{fam} 在线抓取失败，回退离线表] {code}: {e}")
                if feats is None:
                    try:
                        feats = load_latest_features(code, fam)             # 离线表兜底
                    except Exception as e:
                        print(f"  [{fam} 离线表加载失败] {code}: {e}")
                        feats = {}
                f456_features.update({k: _safe_float(v) for k, v in (feats or {}).items()})
        except Exception as e:
            print(f"  [F4/F5 加载失败] {code}: {e}")

        # 4. 行业对标 Z-Score（可选）
        benchmarks = self.industry_benchmark(code, indicators)

        # 5. 异常检测（规则 + 双负兜底 + F2 特征规则）
        anomalies = self.anomaly_detect(indicators, benchmarks) + self.f2_anomaly_rules(f2_features)

        # 6. LLM 财务解读（可选，失败不打断）
        llm_analysis = ""
        if self.use_llm and anomalies:
            llm_analysis = self._llm_analyze(ctx.name or code, indicators, anomalies)

        # 7. 写回 ctx.financial
        ctx.financial.indicators = indicators
        ctx.financial.benchmarks = benchmarks
        ctx.financial.anomaly_list = anomalies
        ctx.financial.risk_level = _risk_level(len(anomalies))
        ctx.financial.llm_analysis = llm_analysis
        # F2-F6 特征（供预测模型）：F2 67 + F3 35 + F4/F5/F6 离线 + 异常统计 + 关键指标
        ctx.financial.features = {
            **f2_features,
            **mkt_features,
            **f456_features,
            "anomaly_count": len(anomalies),
            "max_severity": max((a["severity"] for a in anomalies), default=0),
            "cf_income_ratio": indicators.get("cf_to_profit"),
            "roe": indicators.get("roe"),
            "roe_trend_4q": indicators.get("roe_trend_4q"),
            "debt_to_assets_ratio": indicators.get("debt_to_assets_ratio"),
            "revenue_yoy_growth": indicators.get("revenue_yoy_growth"),
            "net_profit_yoy_growth": indicators.get("net_profit_yoy_growth"),
        }

        # 7.5 规则引擎风险因素（队员 risk_factors 迁移：F2-F6 特征 → 风险因素 JSON 输出②）
        ctx.financial.risk_factors = {}
        try:
            from ..skills.risk_factors import generate_risk_factors
            ctx.financial.risk_factors = generate_risk_factors(
                ctx.financial.features, code, str(indicators.get("report_period", "")))
        except Exception as e:
            print(f"  [风险因素生成失败] {code}: {e}")
        return ctx

    def _industry_characteristics(self, industry):
        """根据行业返回特点说明 + 是否跳过常规财务分析。"""
        if not industry:
            return None, False
        for prefix, profile in SPECIAL_INDUSTRY_PROFILES.items():
            if industry.startswith(prefix):
                return profile["特点"], profile["跳过分析"]
        return "常规行业，适用标准财务异常检测", False

    # ================= 批量扫雷（演示用） =================
    def sweep_batch(self, companies, window=60):
        """批量扫雷：逐家执行 execute()，返回摘要列表（供 Streamlit 排序展示）。"""
        from ..context import Context
        reports = []
        for c in companies:
            ctx = Context(company=c, window=window)
            self.execute(c, ctx)
            reports.append({
                "company": ctx.company,
                "name": ctx.name,
                "industry": ctx.financial.industry,
                "risk_level": ctx.financial.risk_level,
                "anomaly_count": len(ctx.financial.anomaly_list),
                "skip": ctx.financial.skip,
                "skip_reason": ctx.financial.skip_reason,
            })
        return reports


# ============================================================
# 自测入口（python -m backend.agents.financial_detector）
# ============================================================
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from backend.context import Context
    from backend.agents.financial_detector import FinancialDetectorAgent

    agent = FinancialDetectorAgent()
    ctx = Context(company="000004.SZ", window=60)
    agent.execute("000004.SZ", ctx)

    print(f"公司: {ctx.company} | 行业: {ctx.financial.industry} | "
          f"风险等级: {ctx.financial.risk_level} | 跳过: {ctx.financial.skip}")
    if ctx.financial.skip:
        print(f"跳过原因: {ctx.financial.skip_reason}")
    else:
        print(f"报告期: {ctx.financial.indicators.get('report_period')} | "
              f"异常数: {len(ctx.financial.anomaly_list)}")
        for a in ctx.financial.anomaly_list:
            print(f"  [{a['type']}/{a['severity']}] {a['evidence']}  (label_ref={a['label_ref']})")
