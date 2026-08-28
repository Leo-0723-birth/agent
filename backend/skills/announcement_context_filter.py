#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""公告标题与证据语境过滤：降低制度条款、法规引用和会计模板误报。"""
from __future__ import annotations

import re


FILTER_VERSION = "announcement_context_filter_v1"


# 风险事实类标题优先级高于制度标题，防止误跳过真实处罚、辞职或诉讼公告。
RISK_TITLE_OVERRIDES = (
    r"立案(?:告知书|调查|通知)",
    r"行政处罚(?:决定书|事先告知书)?",
    r"监管措施|纪律处分|警示函|公开谴责|通报批评",
    r"问询函|关注函",
    r"辞职|解聘|免职|无法履职|失联|留置|拘留|逮捕|强制措施",
    r"股份冻结|轮候冻结|强制平仓",
    r"(?:违规|非经营性)资金占用|违规担保|债务逾期|债券违约",
    r"重大诉讼|重大仲裁",
    r"(?:拟|预计|关于)?计提.{0,16}(?:减值|坏账)准备",
    r"业绩预告|退市风险警示|终止上市风险",
)


TITLE_EXCLUSION_GROUPS = (
    (
        "governance_rules",
        (
            r"公司章程(?:修订案|（.*?修订.*?）|\(.*?修订.*?\))?$",
            r"(?:修订|修正).{0,8}[《]?公司章程",
            r"(?:股东大[会]|董事会|监事会)议事规则",
            r"独立董事(?:工作|专门会议|年报工作)制度",
            r"董事会.{0,20}委员会.{0,12}(?:工作细则|议事规则)",
            r"(?:董事|监事|高级管理人员|董监高).{0,16}(?:行为规范|管理制度|履职评价办法)",
            r"累积投票(?:制)?实施细则",
        ),
    ),
    (
        "candidate_declaration",
        (
            r"(?:独立董事)?候选人(?:声明|履历|承诺)",
            r"独立董事提名人声明",
            r"关于独立董事.{0,20}任职资格.{0,12}(?:声明|意见|审核)",
        ),
    ),
    (
        "general_management_policy",
        (
            r"(?:对外担保|关联交易|募集资金|信息披露|内幕信息知情人|投资者关系|内部控制|合规)管理制度",
            r"防范.{0,20}(?:控股股东|关联方).{0,20}资金占用.{0,12}(?:制度|办法)",
            r"资产减值准备管理制度",
            r"会计政策(?:和会计估计)?(?:管理办法|制度)$",
        ),
    ),
)


NORMATIVE_CONTEXT_PATTERNS = (
    (
        "governance_eligibility_clause",
        r"因涉嫌证券期货违法犯罪.{0,120}被中国证监会立案调查.{0,80}(?:或者被司法机关立案侦查|尚未有明确结论)",
    ),
    (
        "governance_eligibility_clause",
        r"(?:候选人|被提名人|董事|监事|高级管理人员|董监高).{0,180}(?:不得|不应|不具备|任职资格)",
    ),
    (
        "governance_eligibility_clause",
        r"(?:不得|不应|不具备).{0,180}(?:候选人|被提名人|董事|监事|高级管理人员|董监高)",
    ),
    (
        "conditional_or_hypothetical_clause",
        r"(?:存在|发生|出现)下列情形之一|(?:如|若|假如|一旦).{0,100}(?:发生|存在|出现)",
    ),
    (
        "prohibition_or_duty_clause",
        r"(?:应当|不得|禁止|严禁|有权).{0,100}(?:立案调查|行政处罚|监管措施|资金占用|违规担保|减值准备)",
    ),
    (
        "reporting_duty_clause",
        r"(?:立案调查|行政处罚|监管措施|资金占用|违规担保|重大缺陷|重大风险).{0,120}(?:应当|须|需要).{0,100}(?:报告|报送|披露|通知|提交|告知)",
    ),
    (
        "responsibility_clause",
        r"(?:职责|履职|工作要求|权限).{0,160}(?:应当|须|负责|有权|不得)",
    ),
    (
        "definition_or_template_clause",
        r"(?:本制度所称|是指|释义|附件模板|填写说明).{0,140}",
    ),
)


ACCOUNTING_TEMPLATE_PATTERNS = (
    r"(?:会计政策|会计估计|计提方法|计提比例|减值测试方法)",
    r"(?:存在|出现)减值迹象时.{0,80}(?:应当|需|应)计提",
    r"(?:减值|坏账)准备.{0,60}按(?:单项|组合|账龄|比例)计提",
    r"(?:信用|资产)减值损失.{0,40}损失以.{0,8}[—－-].{0,8}号填列",
    r"项目.{0,30}(?:本期发生额|本期金额).{0,30}(?:上期发生额|上期金额)",
)


def classify_announcement_title(title: str) -> dict:
    """返回标题处置结果；只对明确模板硬过滤，风险标题始终优先保留。"""
    normalized = re.sub(r"\s+", "", str(title or ""))
    for pattern in RISK_TITLE_OVERRIDES:
        if re.search(pattern, normalized, re.IGNORECASE):
            return {
                "decision": "analyze",
                "reason": "risk_title_override",
                "matched_pattern": pattern,
                "filter_version": FILTER_VERSION,
            }
    for reason, patterns in TITLE_EXCLUSION_GROUPS:
        for pattern in patterns:
            if re.search(pattern, normalized, re.IGNORECASE):
                return {
                    "decision": "exclude",
                    "reason": reason,
                    "matched_pattern": pattern,
                    "filter_version": FILTER_VERSION,
                }
    return {
        "decision": "analyze",
        "reason": "",
        "matched_pattern": "",
        "filter_version": FILTER_VERSION,
    }


def apply_title_policy(item: dict, mark_unfetched: bool = False) -> dict:
    """把标题策略写入公告元数据；在线源可同时标记为无需下载。"""
    decision = classify_announcement_title(item.get("title", ""))
    item["analysis_filter_version"] = decision["filter_version"]
    item["analysis_status"] = (
        "excluded_by_title" if decision["decision"] == "exclude" else "eligible"
    )
    item["analysis_skip_reason"] = decision["reason"]
    item["analysis_title_pattern"] = decision["matched_pattern"]
    if mark_unfetched and decision["decision"] == "exclude":
        item["text_status"] = "skipped_title_policy"
        item["ocr_status"] = "not_applicable"
    return item


def is_analysis_eligible(item: dict) -> bool:
    return item.get("analysis_status") != "excluded_by_title"


def _nearby_context(text: str, start: int, end: int, radius: int = 260) -> str:
    text = text or ""
    floor = max(0, start - radius)
    ceiling = min(len(text), end + radius)
    left_marks = [text.rfind(mark, floor, start) for mark in ("\n", "。", "！", "？", "；")]
    left = max(left_marks) + 1 if max(left_marks) >= 0 else floor
    right_marks = [text.find(mark, end, ceiling) for mark in ("\n", "。", "！", "？", "；")]
    positive = [position for position in right_marks if position >= 0]
    right = min(positive) + 1 if positive else ceiling
    return re.sub(
        r"\s+",
        " ",
        text[left:right],
    ).strip()


def contextual_suppression_reason(
    *, rule_id: str = "", label: str = "", text: str, start: int, end: int
) -> str:
    """判断命中是否只是制度/模板文字，或缺少现实事件锚点。"""
    context = _nearby_context(text, start, end)
    for reason, pattern in NORMATIVE_CONTEXT_PATTERNS:
        if re.search(pattern, context, re.IGNORECASE):
            return reason

    if str(label).startswith("C"):
        for pattern in ACCOUNTING_TEMPLATE_PATTERNS:
            if re.search(pattern, context, re.IGNORECASE):
                return "accounting_policy_or_table"

    if label == "G07" or rule_id == "G07_INVESTIGATION_PENALTY":
        subject = re.search(
            r"(?:公司|本公司|上市公司|控股股东|实际控制人|董事|监事|高级管理人员|董监高)",
            context,
        )
        factual_action = re.search(
            r"(?:收到|已被|被.{0,18}(?:立案|调查|处罚)|受到|决定对.{0,20}(?:立案|处罚)|正在接受|正接受)",
            context,
        )
        if not (subject and factual_action):
            return "missing_event_anchors"

    if label == "E03" or rule_id == "E03_FUND_OCCUPATION":
        subject = re.search(r"(?:控股股东|实际控制人|关联方|公司|本公司)", context)
        factual_action = re.search(
            r"(?:已|实际|发生|存在|形成|占用|挪用|余额|金额|尚未归还|已归还)",
            context,
        )
        if not (subject and factual_action):
            return "missing_event_anchors"

    return ""
