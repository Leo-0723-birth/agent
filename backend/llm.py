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
- 模型：deepseek-v4-flash（OpenAI 兼容协议），可用环境变量 DEEPSEEK_MODEL 覆盖
- 只依赖 requests（无 langchain），配置从项目根 .env 读取
- 低温度（默认 0.1）+ JSON 输出约束（防幻觉 / 稳定，方案 5.4）
- 全项目共享同一个客户端实例（get_client 单例），省连接、风格统一

使用前：在项目根 .env 填入 DEEPSEEK_API_KEY=sk-xxx
"""
import json
import os
from pathlib import Path

import requests

# 默认值（会被 .env / 环境变量覆盖）
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 2000

_client = None  # 单例


def _load_env():
    """从项目根 .env 读取 DEEPSEEK_*（系统环境变量优先，不覆盖已存在的）。"""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if k.strip() in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL"):
                os.environ.setdefault(k.strip(), v.strip())


_load_env()


def get_client():
    """获取共享客户端（单例）。"""
    global _client
    if _client is None:
        _client = _DeepSeekClient(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            model=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL),
        )
    return _client


def chat(system: str = "", prompt: str = "", temperature: float = DEFAULT_TEMPERATURE,
         max_tokens: int = DEFAULT_MAX_TOKENS, json_mode: bool = False) -> str:
    """统一 LLM 调用：返回文本。json_mode=True 时要求模型输出 JSON 对象。"""
    return get_client().chat(system, prompt, temperature, max_tokens, json_mode)


def chat_json(system: str = "", prompt: str = "", temperature: float = DEFAULT_TEMPERATURE,
              max_tokens: int = DEFAULT_MAX_TOKENS, schema_hint: str = "") -> dict:
    """统一 LLM 调用：要求 JSON 输出并稳健解析，失败返回 {}。"""

    text = get_client().chat(
        system,
        prompt,
        temperature,
        max_tokens,
        json_mode=True
    )

    print("\n========== LLM RAW ==========")
    print(text[:500] if text else "EMPTY")
    print("============================")

    result = _extract_json(text)

    print("[LLM RESULT]", result.keys())

    return result


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
    """DeepSeek 客户端（OpenAI 兼容协议）。"""

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
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"LLM 调用失败: {resp.status_code} {resp.text[:300]}")
        return resp.json()["choices"][0]["message"]["content"]


# ============================================================
# 自测入口（python -m backend.llm）
# ============================================================
if __name__ == "__main__":
    print(f"模型: {os.getenv('DEEPSEEK_MODEL', DEFAULT_MODEL)}")
    print(f"Base URL: {os.getenv('DEEPSEEK_BASE_URL', DEFAULT_BASE_URL)}")
    print(f"API Key: {'已配置' if os.getenv('DEEPSEEK_API_KEY') else '未配置（请在 .env 填写）'}")
