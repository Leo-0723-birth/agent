import json

import numpy as np
import pandas as pd

from backend.scripts.import_full_announcement_f1 import import_full_f1


def test_import_full_f1_maps_only_features_and_keys(tmp_path):
    source_dir = tmp_path / "run"
    raw_dir = tmp_path / "raw"
    source_dir.mkdir()
    raw_dir.mkdir()

    pcs = {f"PC{i:02d}": [float(i), float(i + 1), float(i + 2)] for i in range(1, 51)}
    source = pd.DataFrame({
        "company_code": ["000001.SZ", "000002.SZ", "EXTRA.SH"],
        "report_period": [20240331, 20240331, 20240331],
        "split": ["Train", "Test", "Train"],
        "y_inquiry_next": [1, 0, 1],
        "has_announcement": [True, False, True],
        **pcs,
    })
    source.to_parquet(source_dir / "company_quarter_pca50.parquet", index=False)
    pd.DataFrame({
        "company_code": ["000001.SZ", "000002.SZ"],
        "report_period": [20240331, 20240331],
        "split": ["Train", "Test"],
    }).to_csv(raw_dir / "F2_financial_anomaly.csv", index=False, encoding="utf-8-sig")

    manifest = import_full_f1(source_dir, raw_dir)
    output = pd.read_parquet(raw_dir / "F1_announcement_semantic_features.parquet")

    assert len(output) == 3
    assert list(output.columns[:2]) == ["stock_code", "T_date"]
    assert "announcement_semantic_000" in output
    assert "announcement_semantic_049" in output
    assert "split" not in output
    assert "y_inquiry_next" not in output
    assert np.isfinite(output.filter(like="announcement_semantic_").to_numpy()).all()
    assert manifest["quality"]["f2_rows_covered"] == 2
    assert manifest["quality"]["source_rows_not_in_f2_skeleton"] == 1
    assert manifest["quality"]["f2_rows_with_announcement"] == 1
    saved = json.loads((raw_dir / "f1_full_run_manifest.json").read_text(encoding="utf-8"))
    assert saved["leakage_guard"].startswith("未导入")
