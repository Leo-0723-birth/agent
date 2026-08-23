# raw/ —— 建模源头数据（训练输入）

> 本目录存放**训练用源头数据**（导入区），与运行时实时爬取的数据分离。
> 由队友/队长导入，导入后在 `../README.md` 记录版本。

## 应含的表（与训练管线对应）

| 文件 | 内容 | 来源 | 状态 |
|---|---|---|---|
| `F1_base_financial.csv` / `F1_semantic_features.parquet` | 公告语义 F1 原始表（50 维） | 队友"预处理后数据"目录 | ❌ 待导入 |
| `F2_financial_anomaly.csv` | F2 财务异常 67 维（新列名，37222 行） | 队友交付（2026-08-23 已收） | ✅ 已导入（含 build_f2_financial_anomaly.py） |
| `F3_market_features.csv` | F3 市场特征 | 队友/队长（本地已有副本在 ../preprocessed/） | ✅ 已就位 |
| `F4_sentiment_features.csv` | F4 舆情特征 | 队友 | ✅ 已就位 |
| `F5_ownership_governance.csv` | F5 治理特征 | 队友 | ✅ 已就位 |
| `F6_inquiry_history.csv` | F6 问询历史特征 | 队友 | ✅ 已就位 |
| `监管问询JSONL/` | 官方问询函结构化数据（4785 案例库源头） | `D:\新建文件夹\02_监管问询` | ❌ 按需导入 |
| `公告PDF/` | 巨潮公告 PDF（公告研读索引源头） | `D:\BaiduNetdiskDownload` | ❌ 按需导入 |

## 导入指引（以 F2 为例）

```bash
# 1. 把队友"预处理后数据"里的 F2_financial_anomaly.csv 复制到本目录
# 2. 重建训练数据集（F1_top50 + F2-F6 + 标签）：
python -m backend.scripts.build_modeling_dataset
# 3. 重训三模型 × 三窗口 + SHAP：
python -m backend.scripts.train_models
# 4. 重建预测兜底字典（fill/）：
python -m backend.scripts.build_fill_dict   # 若存在；否则按 processed_dataset Train split 中位数重建
```

## 注意

- 大文件（>10MB）不入 git（.gitignore 已覆盖 backend/data/cache 等；raw 内大表建议用 git-lfs 或仅在本机保留，README 记录来源路径即可）；
- 表格式统一：含 `company_code`（如 000004.SZ）、`report_period`（YYYY-MM-DD）与特征列；
- 新增表前先确认列名与 `backend/skills/feature_loader.py` 的 `FEATURE_TABLE_CONFIG` 一致。
