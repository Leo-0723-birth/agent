# raw/ —— 建模源头数据（训练输入）

> 本目录存放**训练用源头数据**（导入区），与运行时实时爬取的数据分离。
> 由队友/队长导入，导入后在 `../README.md` 记录版本。

## 应含的表（与训练管线对应）

| 文件 | 内容 | 来源 | 状态 |
|---|---|---|---|
| `F1_announcement_semantic_features.parquet` | F1 全量公告公司季度 PCA50（39,140 行；训练骨架覆盖 37,222/37,222） | `run_20260826_001320_full/company_quarter_pca50.parquet` | ✅ 2026-08-28 已导入 |
| `f1_full_run_manifest.json` | F1 来源哈希、模型 revision、PCA 方差与覆盖率审计 | 导入脚本自动生成 | ✅ 已生成 |
| `F2_financial_anomaly.csv` | F2 财务异常 67 维（新列名，37222 行） | 队友交付（2026-08-23 已收） | ✅ 已导入（含 build_f2_financial_anomaly.py） |
| `F3_market_features.csv` | F3 市场特征 | 队友/队长（本地已有副本在 ../preprocessed/） | ✅ 已就位 |
| `F4_sentiment_features.csv` | F4 舆情特征 | 队友 | ✅ 已就位 |
| `F5_ownership_governance.csv` | F5 治理特征 | 队友 | ✅ 已就位 |
| `F6_inquiry_history.csv` | F6 问询历史特征 | 队友 | ✅ 已就位 |
| `inquiry_events.csv` | 10056 份问询回复函解析 | 队友 | ✅ 已导入 |
| `f1_selection/` | F1 Top-50 选取说明 + 描述文档 + 脚本 | 队友 | ✅ 已导入 |
| `build_f2_financial_anomaly.py` / `F2F6_特征选择与数据来源.xlsx` | F2 构建脚本 / 特征来源说明 | 队友 | ✅ 已导入 |
| `监管问询JSONL/` | 官方问询函结构化数据（4785 案例库源头） | `D:\新建文件夹\02_监管问询` | ❌ 按需导入 |
| `公告PDF/` | 巨潮公告 PDF（公告研读索引源头） | `D:\BaiduNetdiskDownload` | ❌ 按需导入 |

## 全量 F1 导入与重训

```bash
# Windows 示例；其他成员只需把路径替换为自己的全量运行目录
python -m backend.scripts.import_full_announcement_f1 --source-dir "D:\BaiduNetdiskDownload\run_20260826_001320_full"

# 重建训练数据集（F1 PCA50 + F2-F6 + 本仓库标签）
python -m backend.scripts.build_modeling_dataset

# 重训三模型 × 三窗口、校准器与 SHAP
python -m backend.scripts.train_models
```

## 注意

- 导入脚本不读取上游的 `y_inquiry_next` 作为标签，也不把上游 `split` 写入特征表；30/60/90 天标签由本仓库 `inquiry_events.csv` 重建，防止标签泄漏；
- F2 是训练样本骨架。全量 F1 多出的 1,918 行不会在缺少 F2-F6 时强行加入；详细计数见 `f1_full_run_manifest.json`；
- 表格式统一：含 `company_code`（如 000004.SZ）、`report_period`（YYYY-MM-DD）与特征列；
- 新增表前先确认列名与 `backend/skills/feature_loader.py` 的 `FEATURE_TABLE_CONFIG` 一致。
