from backend.context import Context
from backend.agents.predictor import PredictorAgent
from backend.skills.feature_composer import realtime_f1_compatibility


def test_scalar_f1_cannot_be_relabelled_as_pca50_model_input():
    ctx = Context(company="000001.SZ")
    ctx.semantic.f1_features = {"scalar_features": {"risk_event_count_30d": 3}}
    ctx.semantic.f1_model_audit = {"source": "online_rule_llm_scalars_only"}
    manifest = {
        "windows": {
            "30": {"features": ["announcement_semantic_000", "f2_roe"]},
            "60": {"features": ["announcement_semantic_001"]},
        }
    }

    compatible, audit = realtime_f1_compatibility(ctx, manifest)

    assert not compatible
    assert audit["required"] == 2
    assert audit["provided"] == 0
    assert audit["source"] == "online_rule_llm_scalars_only"
    assert "BGE-CLS" in audit["reason"]


def test_offline_fallback_for_known_company_records_model_version(monkeypatch):
    agent = PredictorAgent()
    row = {
        "report_period": "20250331",
        "announcement_semantic_000": 0.1,
    }
    monkeypatch.setattr(agent, "_load_survival", lambda: None)
    monkeypatch.setattr(
        agent,
        "_load_manifest",
        lambda: {"windows": {"30": {"features": ["announcement_semantic_000"]}}},
    )
    monkeypatch.setattr(agent, "_validate_manifest", lambda manifest: (True, []))
    monkeypatch.setattr(agent, "_lookup", lambda code, as_of: row)
    monkeypatch.setattr(agent, "_predict_one", lambda horizon, data, manifest: (0.2, 0.8, []))

    ctx = Context(company="000004.SZ", as_of="2025-12-02")
    result = agent.execute("000004.SZ", ctx)

    assert result.prediction["data_source"] == "offline_lookup"
    assert result.prediction["model_version"].startswith("manifest-sha256:")
    assert result.prediction["feature_anchor"] == "20250331"


def test_predictor_skips_malformed_lightgbm_artifact_before_native_load(tmp_path):
    (tmp_path / "models_manifest.json").write_text(
        '{"windows":{"30":{"features":["f0","f1"]}}}', encoding="utf-8"
    )
    (tmp_path / "model_lgb_30d.txt").write_text(
        "tree\nmax_feature_idx=2\nfeature_names=f0 f1\n", encoding="utf-8"
    )

    models = PredictorAgent(horizons=["30d"], model_dir=tmp_path)._load_models("30d")

    assert models["lgb"] is None


def test_manifest_cache_isolated_by_model_directory(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "models_manifest.json").write_text(
        '{"windows":{"30":{"features":["first"]}}}', encoding="utf-8"
    )
    (second / "models_manifest.json").write_text(
        '{"windows":{"30":{"features":["second"]}}}', encoding="utf-8"
    )

    assert PredictorAgent(model_dir=first)._load_manifest()["windows"]["30"]["features"] == ["first"]
    assert PredictorAgent(model_dir=second)._load_manifest()["windows"]["30"]["features"] == ["second"]
