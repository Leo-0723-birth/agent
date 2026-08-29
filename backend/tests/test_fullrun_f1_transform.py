from datetime import date
import threading

import numpy as np
import pytest

from backend.context import Context
from backend.skills.fullrun_f1_transform import FullRunF1Transformer
from backend.skills.fullrun_online_semantics import FullRunOnlineSemanticPipeline, split_into_chunks


def _row(**overrides):
    row = {
        "announcement_id": "ANN-1",
        "publish_date": "2025-04-15",
        "risk_theme": "G07",
        "l1_code": "G",
        "risk_strength_v2_top3": 0.8,
        "rerank_score_max": 3.2,
        "sent_neg_max": 0.7,
        "strong_count": 1,
        "rule_effective_hits": 2,
    }
    row.update(overrides)
    return row


def test_frozen_fullrun_transform_is_stable_50_dimensions():
    transformer = FullRunF1Transformer()
    features, audit = transformer.transform([_row()], "2025-04-30")

    assert list(features) == [f"announcement_semantic_{i:03d}" for i in range(50)]
    assert np.isfinite(list(features.values())).all()
    assert audit["raw_feature_dim"] == 208
    assert audit["pca_components"] == 50
    assert audit["compatible"] is True


def test_fullrun_aggregation_respects_windows_and_unique_announcements():
    transformer = FullRunF1Transformer()
    rows = [
        _row(),
        _row(risk_theme="C03", l1_code="C", risk_strength_v2_top3=0.2),
        _row(
            announcement_id="ANN-OLD",
            publish_date="2025-01-01",
            risk_theme="D04",
            l1_code="D",
            risk_strength_v2_top3=0.9,
        ),
    ]
    raw = transformer.aggregate_raw(rows, date(2025, 4, 30))

    assert raw["w30_n_ann"] == 1
    assert raw["w30_n_rows"] == 2
    assert raw["w30_n_high"] == 1
    assert raw["w30_n_themes"] == 2
    assert raw["w180_n_ann"] == 2


def test_fullrun_transform_rejects_non_contract_rows():
    transformer = FullRunF1Transformer()
    with pytest.raises(ValueError, match="缺少字段"):
        transformer.transform([{"announcement_id": "rule-only"}], "2025-04-30")


def test_predictor_preparation_keeps_failed_rows_out_of_model_input():
    from backend.agents.predictor import PredictorAgent

    ctx = Context(company="000004.SZ", as_of="2025-04-30")
    ctx.semantic.f1_announcement_risk_rows = [{"announcement_id": "rule-only"}]
    PredictorAgent()._prepare_realtime_f1(ctx)

    assert ctx.semantic.f1_model_features == {}
    assert ctx.semantic.f1_model_audit["compatible"] is False
    assert "缺少字段" in ctx.semantic.f1_model_audit["reason"]


def test_fullrun_chunking_is_deterministic_and_overlapping():
    text = ("第一句用于测试公告语义切块。" * 30) + ("第二句继续提供风险证据。" * 30)
    first = split_into_chunks(text)
    second = split_into_chunks(text)

    assert first == second
    assert len(first) >= 2
    assert all(len(chunk) >= 60 for _, chunk in first)
    assert first[1][0] < first[0][0] + len(first[0][1])


def test_online_semantics_loads_and_releases_one_model_stage_at_a_time():
    class FakePipeline(FullRunOnlineSemanticPipeline):
        def __init__(self):
            self.torch = None
            self.device = "cpu"
            self.batch_size = 8
            self.themes = [{
                "query_for_embedding": "立案调查",
                "query_text": "是否发生立案调查",
                "risk_theme": "G07",
                "l1_code": "G",
            }]
            self._active_kind = None
            self._tokenizer = None
            self._model = None
            self._query_vectors = None
            self._rule_extractor = None
            self.events = []

        def _load_stage(self, kind):
            assert self._active_kind is None
            self._active_kind = kind
            self._model = object()
            self.events.append(("load", kind))

        def _release_stage(self):
            if self._active_kind is not None:
                self.events.append(("release", self._active_kind))
            self._active_kind = None
            self._model = None

        def _embed(self, texts, query=False):
            assert self._active_kind == "bge"
            return np.ones((len(texts), 1), dtype=np.float32)

        def _rerank(self, queries, texts):
            assert self._active_kind == "rerank"
            return np.ones(len(texts), dtype=np.float32)

        def _sentiment(self, texts):
            assert self._active_kind == "finbert"
            return np.tile(np.asarray([[0.8, 0.1, 0.1]], dtype=np.float32), (len(texts), 1))

        def _rule_counts(self, text):
            return {"G07": {"effective": 1, "negated": 0, "excluded": 0}}

    pipeline = FakePipeline()
    rows, audit = pipeline.analyze([{
        "announcement_id": "ANN-ONLINE",
        "published_at": "2025-04-15",
        "category": "risk_warning",
        "text": "公司因涉嫌信息披露违法违规被证监会立案调查。" * 8,
    }], "000004.SZ")

    assert pipeline.events == [
        ("load", "bge"), ("release", "bge"),
        ("load", "rerank"), ("release", "rerank"),
        ("load", "finbert"), ("release", "finbert"),
    ]
    assert pipeline._active_kind is None
    assert rows and rows[0]["risk_theme"] == "G07"
    assert audit["pipeline"] == "fullrun-online-v2-staged"
    features, transform_audit = FullRunF1Transformer().transform(rows, "2025-04-30")
    assert len(features) == 50
    assert transform_audit["raw_feature_dim"] == 208


def test_online_semantics_cancel_releases_active_model_immediately():
    class CancelPipeline(FullRunOnlineSemanticPipeline):
        def __init__(self):
            self.torch = None
            self.device = "cpu"
            self.batch_size = 1
            self.themes = [{"query_for_embedding": "风险"}]
            self._active_kind = None
            self._tokenizer = None
            self._model = None
            self._query_vectors = None
            self._rule_extractor = None
            self._cancel_event = None

        def _load_stage(self, kind):
            self._active_kind = kind
            self._model = object()

        def _release_stage(self):
            self._active_kind = None
            self._model = None

    cancel = threading.Event()
    cancel.set()
    pipeline = CancelPipeline()
    with pytest.raises(RuntimeError, match="已取消"):
        pipeline.analyze([], "000004.SZ", cancel_event=cancel)
    assert pipeline._active_kind is None
    assert pipeline._cancel_event is None
