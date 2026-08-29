#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单公司在线公告的全量流水线同口径语义特征生成器。

口径固定为：600/100 字符切块 → BGE-large-zh-v1.5 CLS → 公告内主题
Top-50 召回（阈值 0.42、最多 8 主题）→ bge-reranker-v2-m3 精排
（每主题 Top-20、每公告 Top-100）→ finbert-tone-chinese 情绪与确定性语境
门控 → 公告×主题行。输出可直接交给 ``FullRunF1Transformer``。

模型首次使用会由 HuggingFace 下载。可用 ``F1_BGE_MODEL_PATH``、
``F1_RERANK_MODEL_PATH``、``F1_FINBERT_MODEL_PATH`` 指向离线模型目录。
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "f1"
QUERY_PATH = MODEL_DIR / "risk_theme_queries.json"
DICTIONARY_PATH = MODEL_DIR / "risk_dictionary_v2.1.yaml"
MODEL_SPECS = {
    "bge": ("BAAI/bge-large-zh-v1.5", "79e7739b6ab944e86d6171e44d24c997fc1e0116"),
    "rerank": ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"),
    "finbert": ("yiyanghkust/finbert-tone-chinese", "e91b1a3af10e1e8c9c03429d3cd7d5e9a1c8000d"),
}
RE_SENT = re.compile(r"(?<=[。！？；\n])")
RE_WS = re.compile(r"[ \t\u3000]+")
RE_MULTINL = re.compile(r"\n{2,}")
RE_NEGATION = re.compile(
    r"不存在|未发现|未发生|没有发生|不涉及|不构成|未受到|未被|并无|无重大|"
    r"不适用|不要求|无需|未出现|未达到|尚未|未有|无此情形|不会导致|"
    r"未[买卖持占违逾]|无[违逾占]|均已(?:解决|消除|整改)|已(?:全部)?(?:归还|清偿|收回)"
)
RE_HEDGE = re.compile(
    r"没有虚假记载|误导性陈述或重大遗漏|真实、准确、完整|"
    r"敬请投资者注意投资风险|本(?:公告|报告)所述|不构成(?:任何)?(?:投资)?建议|"
    r"前瞻性(?:陈述|描述)|存在不确定性，敬请|风险提示：|详见(?:公司)?于"
)
RE_STRONG = re.compile(
    r"逾期未(?:偿还|归还|支付)|无法(?:偿还|清偿|表示意见)|否定意见|保留意见|"
    r"立案(?:调查|告知书)|行政处罚|违规(?:担保|占用)|资金占用余额|"
    r"被实施(?:退市)?风险警示|终止上市|持续经营(?:能力)?(?:存在)?重大不确定"
)


def split_into_chunks(text, size=600, overlap=100, minimum=60):
    text = RE_MULTINL.sub("\n", RE_WS.sub(" ", (text or "").replace("\r", ""))).strip()
    sentences = [value for value in RE_SENT.split(text) if value.strip()]
    chunks, buffer, length, position, start = [], [], 0, 0, 0
    for sentence in sentences:
        sentence_len = len(sentence)
        if sentence_len > size * 2:
            if buffer:
                chunks.append((start, "".join(buffer)))
                buffer, length = [], 0
            for index in range(0, sentence_len, size - overlap):
                piece = sentence[index:index + size]
                if len(piece) >= minimum:
                    chunks.append((position + index, piece))
            position += sentence_len
            start = position
            continue
        if length + sentence_len > size and buffer:
            chunks.append((start, "".join(buffer)))
            tail, tail_len = [], 0
            for previous in reversed(buffer):
                tail.insert(0, previous)
                tail_len += len(previous)
                if tail_len >= overlap:
                    break
            buffer, length, start = tail, tail_len, position - tail_len
        buffer.append(sentence)
        length += sentence_len
        position += sentence_len
    if buffer and length >= minimum:
        chunks.append((start, "".join(buffer)))
    return [(start, value.strip()) for start, value in chunks if len(value.strip()) >= minimum]


class FullRunOnlineSemanticPipeline:
    """懒加载三模型；一个实例可连续处理多家公司。"""

    def __init__(self, device=None, batch_size=None):
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = int(batch_size or os.getenv("F1_ONLINE_BATCH_SIZE", "32"))
        query_bundle = json.loads(QUERY_PATH.read_text(encoding="utf-8"))
        self.themes = [query_bundle["themes"][key] for key in sorted(query_bundle["themes"])]
        self._models = None
        self._query_vectors = None
        self._rule_extractor = None

    @staticmethod
    def _load_hf(kind, model_class):
        from transformers import AutoTokenizer

        repo, revision = MODEL_SPECS[kind]
        path = os.getenv(f"F1_{kind.upper()}_MODEL_PATH") or repo
        kwargs = {} if Path(path).exists() else {"revision": revision}
        tokenizer = AutoTokenizer.from_pretrained(path, **kwargs)
        model = model_class.from_pretrained(path, **kwargs)
        return tokenizer, model, path, revision

    def _load_models(self):
        if self._models is not None:
            return
        from transformers import AutoModel, AutoModelForSequenceClassification

        bge_tok, bge, _, _ = self._load_hf("bge", AutoModel)
        rr_tok, rerank, _, _ = self._load_hf("rerank", AutoModelForSequenceClassification)
        fin_tok, finbert, _, _ = self._load_hf("finbert", AutoModelForSequenceClassification)
        dtype_half = self.device.startswith("cuda")
        if dtype_half:
            bge, rerank, finbert = bge.half(), rerank.half(), finbert.half()
        self._models = (
            bge_tok, bge.to(self.device).eval(),
            rr_tok, rerank.to(self.device).eval(),
            fin_tok, finbert.to(self.device).eval(),
        )
        query_texts = [item["query_for_embedding"] for item in self.themes]
        self._query_vectors = self._embed(query_texts, query=True)

    def _batches(self, values):
        for start in range(0, len(values), self.batch_size):
            yield start, values[start:start + self.batch_size]

    def _embed(self, texts, query=False):
        bge_tok, bge = self._models[0], self._models[1]
        output = []
        for _, batch in self._batches(texts):
            encoded = bge_tok(
                batch, padding=True, truncation=True,
                max_length=128 if query else 512, return_tensors="pt",
            ).to(self.device)
            with self.torch.no_grad():
                vectors = bge(**encoded).last_hidden_state[:, 0]
                vectors = self.torch.nn.functional.normalize(vectors, p=2, dim=1)
            output.append(vectors.float().cpu().numpy())
        return np.concatenate(output) if output else np.empty((0, 1024), dtype=np.float32)

    def _rerank(self, queries, texts):
        tok, model = self._models[2], self._models[3]
        output = np.empty(len(texts), dtype=np.float32)
        for start, indices in self._batches(list(range(len(texts)))):
            encoded = tok(
                [queries[index] for index in indices],
                [texts[index] for index in indices],
                padding=True, truncation=True, max_length=320, return_tensors="pt",
            ).to(self.device)
            with self.torch.no_grad():
                values = model(**encoded).logits.view(-1).float().cpu().numpy()
            output[start:start + len(indices)] = values
        return output

    def _sentiment(self, texts):
        tok, model = self._models[4], self._models[5]
        id2label = getattr(model.config, "id2label", {}) or {}
        label_indices = {}
        for index, label in id2label.items():
            lowered = str(label).lower()
            if "neg" in lowered or lowered in ("0", "负面"):
                label_indices["neg"] = int(index)
            elif "pos" in lowered or lowered in ("2", "正面"):
                label_indices["pos"] = int(index)
            else:
                label_indices["neu"] = int(index)
        if len(label_indices) < 3:
            label_indices = {"neg": 0, "neu": 1, "pos": 2}
        result = np.empty((len(texts), 3), dtype=np.float32)
        for start, batch in self._batches(texts):
            encoded = tok(
                batch, padding=True, truncation=True, max_length=256,
                return_tensors="pt",
            ).to(self.device)
            with self.torch.no_grad():
                probability = self.torch.softmax(model(**encoded).logits.float(), -1)
            values = probability.cpu().numpy()
            result[start:start + len(batch)] = values[:, [
                label_indices["neg"], label_indices["neu"], label_indices["pos"]
            ]]
        return result

    def _rule_counts(self, text):
        """用冻结 v2.1 词典计数；复用在线规则引擎的否定和上下文过滤。"""
        from .rule_risk_extract import RuleRiskExtractor

        if self._rule_extractor is None:
            self._rule_extractor = RuleRiskExtractor(dict_path=DICTIONARY_PATH)
        extractor = self._rule_extractor
        counts = defaultdict(lambda: {"effective": 0, "negated": 0, "excluded": 0})
        for hit in extractor.extract(text):
            theme = str(hit.get("label") or "")
            if hit.get("negated"):
                counts[theme]["negated"] += 1
            elif hit.get("excluded"):
                counts[theme]["excluded"] += 1
            else:
                counts[theme]["effective"] += 1
        return counts

    def analyze(self, announcements, company_code=""):
        self._load_models()
        output = []
        for announcement in announcements:
            text = str(announcement.get("text") or "")
            pieces = split_into_chunks(text)
            if not pieces:
                continue
            chunk_texts = [item[1] for item in pieces]
            chunk_vectors = self._embed(chunk_texts)
            similarities = self._query_vectors @ chunk_vectors.T
            maxima = similarities.max(axis=1)
            active = np.flatnonzero(maxima >= 0.42)
            if len(active) > 8:
                active = active[np.argsort(maxima[active])[-8:]]
            candidates = []
            for theme_index in active:
                top = np.argsort(similarities[theme_index])[-min(50, len(pieces)):][::-1]
                for rank, chunk_index in enumerate(top, 1):
                    candidates.append({
                        "theme_index": int(theme_index), "chunk_index": int(chunk_index),
                        "bge_score": float(similarities[theme_index, chunk_index]),
                        "bge_rank": rank,
                    })
            if not candidates:
                continue
            queries = [self.themes[row["theme_index"]]["query_text"] for row in candidates]
            evidence = [chunk_texts[row["chunk_index"]] for row in candidates]
            rerank_scores = self._rerank(queries, evidence)
            for row, score in zip(candidates, rerank_scores):
                row["rerank_score"] = float(score)
            selected = []
            by_theme = defaultdict(list)
            for row in candidates:
                by_theme[row["theme_index"]].append(row)
            for rows in by_theme.values():
                selected.extend(sorted(rows, key=lambda value: -value["rerank_score"])[:20])
            selected = sorted(selected, key=lambda value: -value["rerank_score"])[:100]
            selected_text = [chunk_texts[row["chunk_index"]] for row in selected]
            sentiment = self._sentiment(selected_text)
            for row, probs, evidence_text in zip(selected, sentiment, selected_text):
                row.update({
                    "sent_neg": float(probs[0]), "sent_neu": float(probs[1]),
                    "sent_pos": float(probs[2]), "evidence_text": evidence_text,
                    "negation_flag": bool(RE_NEGATION.search(evidence_text)),
                    "hedge_flag": bool(RE_HEDGE.search(evidence_text)),
                    "strong_flag": bool(RE_STRONG.search(evidence_text)),
                })
                sigmoid = 1.0 / (1.0 + np.exp(-row["rerank_score"]))
                context_weight = (
                    (0.25 if row["negation_flag"] else 1.0)
                    * (0.40 if row["hedge_flag"] else 1.0)
                    * (1.30 if row["strong_flag"] else 1.0)
                )
                row["risk_strength"] = float(
                    sigmoid * (0.5 + 0.5 * (row["sent_neg"] - row["sent_pos"]))
                    * context_weight
                )
                neg_ratio = row["sent_neg"] / (row["sent_neg"] + row["sent_pos"] + 1e-6)
                row["risk_strength_v2"] = float(
                    sigmoid * (0.30 + 0.70 * neg_ratio) * context_weight
                )
            rules = self._rule_counts(text)
            for theme_index, rows in by_theme.items():
                rows = [row for row in selected if row["theme_index"] == theme_index]
                if not rows:
                    continue
                theme = self.themes[theme_index]
                strengths = sorted((row["risk_strength_v2"] for row in rows), reverse=True)
                rule = rules.get(theme["risk_theme"], {})
                top = sorted(rows, key=lambda value: -value["risk_strength"])[:3]
                output.append({
                    "announcement_id": announcement.get("announcement_id") or announcement.get("id"),
                    "company_code": company_code,
                    "publish_date": announcement.get("published_at") or announcement.get("date"),
                    "doc_type": announcement.get("category") or announcement.get("type") or "",
                    "risk_theme": theme["risk_theme"], "l1_code": theme["l1_code"],
                    "evidence_count": len(rows),
                    "rerank_score_max": max(row["rerank_score"] for row in rows),
                    "sent_neg_max": max(row["sent_neg"] for row in rows),
                    "strong_count": sum(row["strong_flag"] for row in rows),
                    "risk_strength_v2_top3": float(np.mean(strengths[:3])),
                    "rule_effective_hits": int(rule.get("effective", 0)),
                    "rule_negated_hits": int(rule.get("negated", 0)),
                    "rule_excluded_hits": int(rule.get("excluded", 0)),
                    "top_evidence": json.dumps([
                        {"chunk_index": row["chunk_index"], "bge_rank": row["bge_rank"],
                         "rerank_score": round(row["rerank_score"], 4),
                         "risk_strength": round(row["risk_strength"], 4),
                         "text": row["evidence_text"][:250]}
                        for row in top
                    ], ensure_ascii=False),
                })
        return output, {
            "status": "generated", "pipeline": "fullrun-online-v1",
            "device": self.device, "announcement_count": len(announcements),
            "announcement_theme_rows": len(output), "model_revisions": {
                key: value[1] for key, value in MODEL_SPECS.items()
            },
        }
