#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
财务数据爬虫模块 (DataFetcher) —— 演示数据源
=============================================
用东方财富免费接口（无需登录/token）爬取上市公司财务指标。
面向【评委演示】使用实时真实数据；比赛模型训练用官方数据集（另走 train_predictor.py）。

输入: 公司名称 + 股票代码（如 国华网安 + 000004.SZ）
输出: 财务特征 DataFrame（report_period, total_revenue, net_profit, roe, ...）

字段映射（东方财富 → Wind 可比）:
  TOTALOPERATEREVE    → total_revenue   营业总收入
  PARENTNETPROFIT     → net_profit      归母净利润
  NETCASH_OPERATE_PK  → operating_cash_flow 经营现金流
  ROEJQ               → roe             净资产收益率(加权)
  ZZCJLL              → roa             总资产净利率
  ZCFZL               → debt_to_assets_ratio 资产负债率
  TOTALOPERATEREVETZ  → revenue_yoy_growth 营收同比
  PARENTNETPROFITTZ   → net_profit_yoy_growth 净利润同比
"""
import sys
import time
import requests
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, AttributeError, OSError):
        pass

# 东方财富财务主要指标接口
EASTMONEY_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"

# 字段映射: 东方财富字段 → Wind 可比英文名
FIELD_MAP = {
    "SECURITY_CODE": "stock_code",
    "SECURITY_NAME_ABBR": "company_name",
    "REPORT_DATE": "report_date",
    "TOTALOPERATEREVE": "total_revenue",
    "PARENTNETPROFIT": "net_profit",
    "NETCASH_OPERATE_PK": "operating_cash_flow",
    "ROEJQ": "roe",
    "ZZCJLL": "roa",
    "ZCFZL": "debt_to_assets_ratio",
    "TOTALOPERATEREVETZ": "revenue_yoy_growth",
    "PARENTNETPROFITTZ": "net_profit_yoy_growth",
    "XSMLL": "gross_margin",
    "XSJLL": "net_margin",
    "LD": "current_ratio",
    "SD": "quick_ratio",
}


class DataFetcher:
    """财务数据爬虫，封装东方财富免费接口。"""

    def __init__(self, rate_limit=1.0):
        self.rate_limit = rate_limit  # 每次请求间隔秒数
        self.headers = {"User-Agent": "Mozilla/5.0"}

    def _secucode(self, stock_code):
        """把 000004 / 000004.SZ 补全成东方财富要求的 SECUCODE 格式。"""
        code = stock_code.strip().upper()
        if "." not in code:
            if code.startswith(("6", "9")):
                code += ".SH"
            elif code.startswith(("0", "3")):
                code += ".SZ"
            elif code.startswith(("4", "8")):
                code += ".BJ"
        return code

    def fetch_financials(self, stock_code, page_size=20):
        """爬取单家公司财务指标（默认最近20期），返回标准化 DataFrame。"""
        secucode = self._secucode(stock_code)
        params = {
            "reportName": "RPT_F10_FINANCE_MAINFINADATA",
            "columns": "ALL",
            "filter": f'(SECUCODE="{secucode}")',
            "pageNumber": 1,
            "pageSize": page_size,
            "sortTypes": -1,
            "sortColumns": "REPORT_DATE",
        }
        try:
            r = requests.get(EASTMONEY_URL, params=params, timeout=15, headers=self.headers)
            data = r.json()
            if data.get("result") is None or not data["result"].get("data"):
                return None
            rows = data["result"]["data"]
        except Exception as e:
            print(f"  [爬取失败] {secucode}: {e}")
            return None

        df = pd.DataFrame(rows)
        keep = {k: v for k, v in FIELD_MAP.items() if k in df.columns}
        df_out = df[list(keep.keys())].rename(columns=keep)
        df_out["secucode"] = secucode
        df_out["report_date"] = pd.to_datetime(df_out["report_date"])
        df_out["report_period"] = df_out["report_date"].dt.strftime("%Y%m%d").astype(int)
        df_out = df_out.sort_values("report_date").reset_index(drop=True)
        return df_out

    def fetch_company_profile(self, stock_code):
        """爬取公司基本资料（名称、行业、板块、上市日期）。失败返回 None。"""
        secucode = self._secucode(stock_code)
        params = {
            "reportName": "RPT_F10_BASIC_ORGINFO",
            "columns": "ALL",
            "filter": f'(SECUCODE="{secucode}")',
            "pageNumber": 1,
            "pageSize": 1,
        }
        try:
            r = requests.get(EASTMONEY_URL, params=params, timeout=15, headers=self.headers)
            data = r.json()
            rows = data.get("result", {}).get("data", [])
            if not rows:
                return None
            row = rows[0]
            return {
                "secucode": secucode,
                "company_name": row.get("SECURITY_NAME_ABBR"),
                "org_name": row.get("ORG_NAME"),
                "industry": row.get("INDUSTRYCSRC1"),
                "board": row.get("BOARD_NAME_LEVEL"),
                "listing_date": str(row.get("LISTING_DATE", ""))[:10],
            }
        except Exception as e:
            print(f"  [资料爬取失败] {secucode}: {e}")
            return None

    def fetch_company_list(self, company_list, output_csv=None):
        """批量爬取公司列表的财务特征并保存。"""
        all_frames = []
        for i, item in enumerate(company_list):
            if isinstance(item, (tuple, list)):
                name, code = item[0], item[1]
            else:
                name, code = None, item
            print(f"  [{i+1}/{len(company_list)}] 爬取 {name or ''} ({code}) ...")
            df = self.fetch_financials(code)
            if df is not None and len(df) > 0:
                if name:
                    df["input_name"] = name
                all_frames.append(df)
            else:
                print(f"    -> 无数据，跳过 {code}")
            if i < len(company_list) - 1:
                time.sleep(self.rate_limit)

        if not all_frames:
            print("  所有公司均无数据")
            return None
        merged = pd.concat(all_frames, ignore_index=True)
        if output_csv:
            import os
            os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
            merged.to_csv(output_csv, index=False, encoding="utf-8-sig")
            print(f"  已保存 {len(merged)} 行到 {output_csv}")
        return merged


# ============================================================
# 自测入口
# ============================================================
if __name__ == "__main__":
    fetcher = DataFetcher(rate_limit=0.5)
    df = fetcher.fetch_financials("000004.SZ")
    if df is not None:
        print(f"爬取到 {len(df)} 期数据")
        cols = ["report_period", "total_revenue", "net_profit", "roe", "debt_to_assets_ratio"]
        print(df[cols].tail(3).to_string(index=False))
    else:
        print("无数据（可能网络受限）")
