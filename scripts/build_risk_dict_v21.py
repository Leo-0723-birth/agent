# -*- coding: utf-8 -*-
"""
生成 risk_dictionary_v2.1-taxonomy-v1.1.yaml：在冻结版 v2.0.0 基础上，
补充"批次1"高影响主题的规则（关键词+正则，风格与 v2.0.0 一致），并输出覆盖报告。
批次1 = 案例库中样本量最大的未覆盖主题：C06 E01 E04 A01 G04 E02 A02 A04 E06 E07 B01
"""
import io
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SRC = Path(r"D:\competition_agent\backend\data\labels\risk_dictionary.yaml")
OUT = Path(r"D:\competition_agent\backend\data\labels\risk_dictionary_v2.1-taxonomy-v1.1.yaml")

base = yaml.safe_load(SRC.read_text(encoding="utf-8"))

# ============================================================
# 批次 1 新增规则（11 个主题，风格对齐 v2.0.0）
# ============================================================
NEW_RULES = {
    "A": [
        {
            "rule_id": "A01_REVENUE_DECLINE",
            "label": "A01",
            "severity": "medium",
            "keywords": [
                "营业收入大幅下降", "营业收入减少", "收入大幅下滑", "主营业务收入下降",
                "收入规模缩小", "销售收入下降", "收入大幅减少", "营业收入下滑",
            ],
            "regexes": [
                r"营业(?:总)?收入.{0,16}(?:同比下降|较上年同期减少|下滑|减少|下降)\s*(?:约)?\d+(?:\.\d+)?%",
                r"(?:收入|营业收入).{0,12}(?:下滑|下降|减少)(?:至|为)\d+(?:\.\d+)?(?:亿|万)?元",
            ],
            "note": "扩展标签：收入规模与收入变动（营收下滑/减少/规模缩小）",
        },
        {
            "rule_id": "A02_MARGIN_COST",
            "label": "A02",
            "severity": "medium",
            "keywords": [
                "毛利率下降", "毛利率下滑", "毛利率偏低", "毛利率大幅下降", "成本费用增加",
                "期间费用上升", "毛利率明显低于", "综合毛利率下降", "成本上升",
            ],
            "regexes": [
                r"毛利率.{0,12}(?:下降|下滑|降低|减少)\s*(?:约)?\d+(?:\.\d+)?%",
                r"(?:营业成本|销售费用|管理费用|财务费用|研发费用).{0,10}(?:同比)?(?:大幅)?(?:增长|增加|上升)\s*(?:约)?\d+(?:\.\d+)?%",
            ],
            "note": "扩展标签：成本费用与毛利率（毛利下滑/费用率上升）",
        },
        {
            "rule_id": "A04_BUSINESS_SUBSTANCE",
            "label": "A04",
            "severity": "medium",
            "keywords": [
                "商业实质", "业务模式", "经营模式", "主营业务是否", "业务真实性",
                "盈利模式", "业务实质", "是否具备商业实质", "主营业务变更",
            ],
            "regexes": [
                r"(?:是否|是否具备).{0,10}(?:商业实质|真实商业背景|业务真实性)",
                r"(?:业务模式|经营模式|商业模式).{0,20}(?:是否合理|是否真实|是否符合|是否持续)",
            ],
            "note": "扩展标签：业务模式与商业实质（模式合理性/真实性）",
        },
    ],
    "B": [
        {
            "rule_id": "B01_REVENUE_RECOGNITION",
            "label": "B01",
            "severity": "high",
            "keywords": [
                "收入确认", "确认收入", "提前确认收入", "收入确认政策", "收入确认时点",
                "截止性", "跨期确认收入", "收入确认方法", "是否满足收入确认条件",
            ],
            "regexes": [
                r"(?:是否)?(?:提前|跨期|延迟)?确认收入.{0,20}(?:是否合理|是否符合准则|是否合规|是否恰当)",
                r"收入确认.{0,16}(?:时点|政策|方法|条件|原则)",
            ],
            "note": "扩展标签：收入确认与截止性（B类会计主题，重点补充）",
        },
    ],
    "C": [
        {
            "rule_id": "C06_OTHER_ASSETS_INVEST",
            "label": "C06",
            "severity": "medium",
            "keywords": [
                "其他应收款", "长期股权投资", "投资性房地产", "交易性金融资产",
                "可供出售金融资产", "公允价值变动", "投资损失", "投资收益确认",
                "其他流动资产", "其他非流动金融资产",
            ],
            "regexes": [
                r"(?:其他应收款|长期股权投资|投资性房地产|交易性金融资产).{0,20}(?:减值|跌价|公允价值变动|处置|核销|真实性)",
                r"(?:投资损失|投资收益).{0,16}(?:确认|合理性|会计处理|是否恰当)",
            ],
            "note": "扩展标签：其他资产与投资资产（C06，样本量最大）",
        },
    ],
    "E": [
        {
            "rule_id": "E01_TRANSACTION_FAIRNESS",
            "label": "E01",
            "severity": "high",
            "keywords": [
                "定价公允", "定价公允性", "交易对价", "公允价值", "商业实质",
                "交易合理性", "评估价值", "成交价格", "价格是否公允", "交易价格公允",
            ],
            "regexes": [
                r"(?:交易|收购|出售|转让)?(?:价格|定价|对价).{0,16}(?:是否公允|公允性|是否合理|合理性)",
                r"(?:评估|估值).{0,12}(?:价值|增值率).{0,20}(?:公允|合理|是否恰当)",
            ],
            "note": "扩展标签：交易商业实质、合理性与定价公允性",
        },
        {
            "rule_id": "E02_RELATED_PARTY",
            "label": "E02",
            "severity": "high",
            "keywords": [
                "关联方", "关联交易", "关联关系", "关联资金", "关联担保",
                "关联采购", "关联销售", "关联方资金往来", "是否构成关联交易",
            ],
            "regexes": [
                r"关联(?:方|交易).{0,20}(?:是否公允|是否合理|是否履行|披露|定价|必要性|商业实质)",
                r"(?:与|向)关联(?:方|人).{0,16}(?:采购|销售|资金|往来|借款|担保)",
            ],
            "note": "扩展标签：关联方识别与关联交易",
        },
        {
            "rule_id": "E04_ASSET_TRANSFER",
            "label": "E04",
            "severity": "high",
            "keywords": [
                "年末突击交易", "突击交易", "资产出售", "股权转让", "资产收购",
                "处置资产", "转让股权", "年末突击", "突击处置",
            ],
            "regexes": [
                r"(?:年末|期末|第四季度|第四季).{0,10}(?:突击|集中).{0,8}(?:交易|出售|处置|转让)",
                r"(?:出售|转让|处置)(?:股权|资产|子公司|房产).{0,20}(?:原因|合理性|是否",
            ],
            "note": "扩展标签：资产购买、出售与年末突击交易",
        },
        {
            "rule_id": "E06_VALUATION",
            "label": "E06",
            "severity": "high",
            "keywords": [
                "评估方法", "估值方法", "评估报告", "收益法", "市场法",
                "资产基础法", "评估增值", "评估值", "评估合理性", "估值合理性",
            ],
            "regexes": [
                r"(?:评估|估值)(?:方法|报告|机构).{0,16}(?:合理性|恰当性|是否合理|是否恰当)",
                r"(?:收益法|市场法|资产基础法).{0,20}(?:评估|估值|增值)",
            ],
            "note": "扩展标签：估值方法与评估合理性",
        },
        {
            "rule_id": "E07_PERFORMANCE_COMMITMENT",
            "label": "E07",
            "severity": "high",
            "keywords": [
                "业绩承诺", "业绩补偿", "补偿安排", "对赌协议", "业绩对赌",
                "补偿义务", "业绩承诺未达标", "业绩承诺完成", "业绩承诺期",
            ],
            "regexes": [
                r"业绩承诺.{0,20}(?:未达标|未完成|实现|完成率|补偿|履行|变更)",
                r"(?:补偿|对赌).{0,16}(?:安排|义务|协议|条款|股份回购|现金补偿)",
            ],
            "note": "扩展标签：业绩承诺、补偿安排与对赌协议",
        },
    ],
    "G": [
        {
            "rule_id": "G04_DISCLOSURE_COMPLETENESS",
            "label": "G04",
            "severity": "medium",
            "keywords": [
                "信息披露不完整", "披露不完整", "信息披露遗漏", "披露遗漏",
                "重大遗漏", "未披露", "披露是否完整", "信息遗漏", "披露义务",
            ],
            "regexes": [
                r"(?:是否)?(?:充分|完整|及时|准确)?披露.{0,16}(?:不完整|遗漏|缺失|不充分)",
                r"(?:重大)?(?:信息)?遗漏|未(?:能|按)?(?:及时|完整)?披露",
            ],
            "note": "扩展标签：信息披露完整性",
        },
    ],
}

# ============================================================
# 合并生成 v2.1（保留 v2.0.0 全部规则 + 新增；冻结版缺失的一级类别补建）
# ============================================================
CAT_NAMES = {
    "A": "经营业绩与商业模式", "B": "财务报告与会计处理", "C": "资产质量与减值",
    "D": "资金、现金流与偿债", "E": "交易、关联方与资本运作",
    "F": "股东、控制权与公司治理", "G": "审计、披露与监管合规", "H": "市场交易与证券行为",
}
have = {c.get("category_id") for c in base["categories"]}
for cid, name in CAT_NAMES.items():
    if cid not in have:
        base["categories"].append({"category_id": cid, "category_name": name, "rules": []})

new_count = 0
for cat in base["categories"]:
    cid = cat.get("category_id")
    if cid in NEW_RULES:
        existing = {r.get("rule_id") for r in cat.get("rules", [])}
        for rule in NEW_RULES[cid]:
            if rule["rule_id"] not in existing:
                cat.setdefault("rules", []).append(rule)
                new_count += 1

base["dictionary_version"] = "2.1.0-taxonomy-v1.1"
base["note"] = (
    "v2.1.0 = 冻结版 v2.0.0（任务1交付，16主题规则）基础上，扩展批次1共 %d 条规则"
    "（A01/A02/A04/B01/C06/E01/E02/E04/E06/E07/G04），服务公告全文 45 主题门控；"
    "规则层仍仅作候选召回，最终判定由门控/LLM 结合上下文。" % new_count
)
# 与冻结版一致：词典实际以 JSON 格式存储（加载器 json.load 读取）
OUT.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"已生成 {OUT.name}：新增 {new_count} 条规则")

# ============================================================
# 验证：用案例库 letter_excerpt（自带 taxonomy_labels）测命中率/误报率
# ============================================================
import json
from backend.skills.rule_risk_extract import RuleRiskExtractor

extractor = RuleRiskExtractor(dict_path=str(OUT))
case_db = json.load(open(r"D:\competition_agent\backend\data\vector_db\case_db.json", encoding="utf-8"))

# 统计 v2.1 实际新增的规则数（诊断用）
new_ids = {r["rule_id"] for rules in NEW_RULES.values() for r in rules}
added = [r["rule_id"] for cat in base["categories"] for r in cat.get("rules", []) if r["rule_id"] in new_ids]
print("实际入库新规则:", len(added), added)

def text_of(c):
    t = (c.get("letter_excerpt") or "").strip()
    if not t:
        t = "；".join(c.get("focus_points") or [])
    return t

targets = ["A01", "A02", "A04", "B01", "C06", "E01", "E02", "E04", "E06", "E07", "G04"]
print(f"\n=== 批次1 验证（案例库 {len(case_db)} 条）===")
print(f"{'主题':<5}{'正样本':<7}{'命中':<6}{'命中率':<8}{'误报数':<7}{'误报率'}")
for lab in targets:
    pos = [c for c in case_db if lab in (c.get("taxonomy_labels") or [])]
    neg = [c for c in case_db if lab not in (c.get("taxonomy_labels") or [])]
    hit = 0
    for c in pos:
        hits = extractor.extract(text_of(c))
        if any(h["label"] == lab for h in hits):
            hit += 1
    fp = 0
    for c in neg:
        hits = extractor.extract(text_of(c))
        if any(h["label"] == lab for h in hits):
            fp += 1
    hr = hit / len(pos) if pos else 0
    fr = fp / len(neg) if neg else 0
    print(f"{lab:<5}{len(pos):<7}{hit:<6}{hr:<8.1%}{fp:<7}{fr:.2%}")
