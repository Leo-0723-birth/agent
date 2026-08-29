#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
共享 LLM 客户端（全项目唯一模型访问点）
======================================
一行 import 即用：
    from backend.llm import chat, chat_json
    text = chat("你是金融分析专家", "请分析这段公告的风险。")
    data = chat_json("你是提取器", "请抽取风险要素。", schema_hint="risk_factors")

设计要点：
- 模型：deepseek（OpenAI 兼容协议），可用环境变量 DEEPSEEK_MODEL 覆盖
- 双通道：
    ① LangChain 通道（首选，langchain_deepseek.ChatDeepSeek，Pydantic 结构化输出
      chat_structured 走 with_structured_output）
    ② requests 直连通道（兜底：LangChain 未安装 / 调用异常时自动回落，
      行为与历史版本完全一致）
- 低温度（默认 0.1）+ JSON 输出约束（防幻觉 / 稳定，方案 5.4）
- 对外签名 chat/chat_json 保持不变 → 所有 Agent 调用方零改动

使用前：在项目根 .env 填入 DEEPSEEK_API_KEY=sk-xxx
"""
import json
import logging
import os
from pathlib import Path

import requests

from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
_logger = logging.getLogger(__name__)

# 默认值（会被 .env / 环境变量覆盖）
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 2000

_client = None  # 单例（requests 兜底通道）
_lc_base = None  # 单例（LangChain 基础模型）
LANGCHAIN_AVAILABLE = False
try:
    from langchain_deepseek import ChatDeepSeek

    LANGCHAIN_AVAILABLE = True
except Exception as _e:  # 未安装 langchain-deepseek 时降级（记录日志便于排查）
    _logger.warning("langchain-deepseek 不可用，将使用 requests 直连: %s: %s", type(_e).__name__, _e)
    ChatDeepSeek = None


def _require_key():
    """返回 DEEPSEEK_API_KEY；未配置时抛出明确异常。"""
    if not LLM_API_KEY:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，请在项目根 .env 中填写")
    return LLM_API_KEY


def get_client():
    """获取共享客户端（requests 兜底通道，单例）。"""
    global _client
    if _client is None:
        _client = _DeepSeekClient(
            api_key=_require_key(),
            base_url=LLM_BASE_URL,
            model=LLM_MODEL,
        )
    return _client


def _get_lc_base():
    """获取 LangChain 基础模型（单例，参数用 bind 按调用覆盖）。"""
    global _lc_base
    if _lc_base is None:
        _lc_base = ChatDeepSeek(
            model=LLM_MODEL,
            api_key=_require_key(),
            base_url=LLM_BASE_URL,
            timeout=120,
        )
    return _lc_base


def _langchain_chat(system: str, prompt: str, temperature: float,
                    max_tokens: int) -> str:
    """LangChain 通道：ChatDeepSeek.invoke（非 JSON 调用）。

    说明：json_mode 已改走 requests 直连（thinking 禁用），本函数不再处理 JSON。
    """
    model = _get_lc_base().bind(temperature=temperature, max_tokens=max_tokens)
    messages = [
        ("system", system or "你是严谨的金融风控分析助手。"),
        ("human", prompt),
    ]
    resp = model.invoke(messages)
    return str(resp.content)


def chat(system: str = "", prompt: str = "", temperature: float = DEFAULT_TEMPERATURE,
         max_tokens: int = DEFAULT_MAX_TOKENS, json_mode: bool = False) -> str:
    """统一 LLM 调用：返回文本。json_mode=True 时要求模型输出 JSON 对象。

    通道策略：
    - json_mode=True：直接走 requests 直连（thinking 禁用）——DeepSeek v4-flash 的
      thinking 模式会吃光 max_tokens 导致 LengthFinishReasonError，且输出 JSON 结构不稳；
    - 其余：LangChain 优先，异常回落 requests 直连。
    """
    if json_mode:
        return get_client().chat(system, prompt, temperature, max_tokens, json_mode=True)
    if LANGCHAIN_AVAILABLE:
        try:
            return _langchain_chat(system, prompt, temperature, max_tokens)
        except Exception as e:
            _logger.warning("LangChain 通道异常，回落 requests 直连: %s: %s", type(e).__name__, e)
    return get_client().chat(system, prompt, temperature, max_tokens, json_mode=False)


def chat_json(system: str = "", prompt: str = "", temperature: float = DEFAULT_TEMPERATURE,
              max_tokens: int = DEFAULT_MAX_TOKENS, schema_hint: str = "") -> dict:
    """统一 LLM 调用：要求 JSON 输出并稳健解析，失败返回 {}。"""
    text = chat(system, prompt, temperature, max_tokens, json_mode=True)
    return _extract_json(text)


def chat_structured(model_class, system: str = "", prompt: str = "",
                    temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = DEFAULT_MAX_TOKENS):
    """Pydantic 结构化输出：JSON Schema 约束 + json_object 模式 + 强校验。

    返回 model_class 实例；调用失败/解析失败返回 None（调用方自行降级）。

    实现说明：DeepSeek v4 的 thinking 模式不支持 tool_choice（with_structured_output
    默认走 tool calling 会被 API 拒绝），故采用「Schema 提示 + response_format
    json_object + model_validate_json」的兼容路径。

    用法：
        from pydantic import BaseModel
        class RiskFactor(BaseModel):
            category: str
            severity: int
        f = chat_structured(RiskFactor, "抽取风险", "……")
    """
    if not LANGCHAIN_AVAILABLE:
        _logger.warning("LangChain 不可用，chat_structured 返回 None")
        return None
    try:
        schema = model_class.model_json_schema()
        json_prompt = (
            f"{prompt}\n\n请输出一个合法的 JSON 对象（json 格式），"
            f"严格匹配以下 JSON Schema：\n{schema}"
        )
        text = chat(system, json_prompt, temperature, max_tokens, json_mode=True)
        if not text or not text.strip():
            return None
        return model_class.model_validate_json(text)
    except Exception as e:
        _logger.warning("chat_structured 失败: %s: %s", type(e).__name__, e)
        return None


def _extract_json(text: str) -> dict:
    """从 LLM 回复里稳健提取 JSON 对象（容忍 markdown 代码块、前后杂文）。"""
    if not text:
        return {}
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


class _DeepSeekClient:
    """DeepSeek 客户端（OpenAI 兼容协议，requests 兜底通道）。"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def chat(self, system: str, prompt: str, temperature: float,
             max_tokens: int, json_mode: bool) -> str:
        if not self.api_key:
            raise RuntimeError("未配置 DEEPSEEK_API_KEY，请在项目根 .env 中填写")
        url = self.base_url + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system or "你是严谨的金融风控分析助手。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        # 禁用 thinking：v4-flash 推理会耗尽 max_tokens 且 JSON 结构不稳；
        # 抽取/结构化任务走确定性输出（可用环境变量 DEEPSEEK_THINKING=1 重新开启）
        if os.getenv("DEEPSEEK_THINKING", "0") != "1":
            payload["thinking"] = {"type": "disabled"}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"LLM 调用失败: {resp.status_code} {resp.text[:300]}")
        return resp.json()["choices"][0]["message"]["content"]


# ============================================================
# 自测入口（python -m backend.llm）
# ============================================================
if __name__ == "__main__":
    print(f"模型: {LLM_MODEL}")
    print(f"Base URL: {LLM_BASE_URL}")
    print(f"API Key: {'已配置' if LLM_API_KEY else '未配置（请在 .env 填写）'}")
    print(f"LangChain 通道: {'可用' if LANGCHAIN_AVAILABLE else '不可用（将使用 requests 直连）'}")
