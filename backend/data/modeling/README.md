# 建模数据框架（训练/预测分离）

> 本目录集中存放**与建模相关的全部数据**，遵循一条主线：
> **训练用离线数据（raw → preprocessed → processed_dataset → models）**，
> **预测用实时爬取数据（公告研读 F1 标量 + 财务 F2-F6 实时值）+ 离线字典兜底**。
> 更新：2026-08-23

## 目录结构

```
backend/data/modeling/
├── README.md                  本框架文档
├── raw/                       源头/导入区（训练输入，需人工或脚本导入）
│   └── README.md              各源头表来源与导入指引（队员"预处理后数据"、官方 JSONL）
├── preprocessed/              离线预处理特征表（与训练同源，F3-F6）
│   ├── F3_market_features.csv
│   ├── F4_sentiment_features.csv
│   ├── F5_ownership_governance.csv
│   └── F6_inquiry_history.csv
├── processed_dataset.csv      训练数据集 37222×204（F1_top50 + F2-F6 + 30/60/90d 标签）
├── F1_top50_features.csv      F1 语义特征 Top-50 清单（Spearman 选取）
├── fill/                      预测兜底字典（训练集 Train split 中位数，实时特征缺失列用）
│   ├── fill_median_30d.csv
│   ├── fill_median_60d.csv
│   └── fill_median_90d.csv
└── output/                    训练/评估输出（SHAP、预测、风险排序、模型摘要）
```

## 数据流：训练 vs 预测

```
【训练（离线）】
  源头数据(队员预处理后数据/官方数据)
    → raw/（导入，含 F1-F6 表）
    → preprocessed/（F3-F6 已就位；F2 表待导入）→ 合并 → processed_dataset.csv
    → backend/scripts/build_modeling_dataset.py → train_models.py
    → backend/models/predictor/（9 模型 + models_manifest.json + feature_importance）
    → output/（SHAP/预测/风险排序，评估用）

【预测（实时）】
  运行时爬取：巨潮公告（公告研读）→ F1 标量 + F6 问询特征
             东财财报（财务异常）→ F2 实时现算
             腾讯行情（财务异常）→ F3 实时
             股吧舆情/股东治理（财务异常）→ F4/F5 实时（超时回退离线表）
    → backend/skills/feature_composer.py 按 models_manifest.json 组装向量
    → 缺失列用 fill/ 中位数兜底（离线数据仅作初始建模兜底）
    → PredictorAgent 三模型集成 → 概率 + SHAP
```

## 各目录职责与维护

| 目录/文件 | 职责 | 何时更新 | 谁负责 |
|---|---|---|---|
| `raw/` | 源头数据（训练输入） | 队友交付新表时导入 | 队长/队友 |
| `preprocessed/` | 离线特征表 F3-F6（训练同源） | 重新批处理特征时 | 队友（Wind/CNRDS 源） |
| `processed_dataset.csv` | 训练/验证/测试数据集 | 重新建模时由脚本重建 | 队长 |
| `fill/` | 预测兜底字典（Train 中位数） | 模型重训后重建 | 队长 |
| `output/` | 训练评估产物 | 每次训练后 | 队长 |
| `backend/models/predictor/` | 模型权重 + manifest（不在本目录，见 backend/models/） | 重训后 | 队长 |

## 已知缺口（需队员补充）

1. **F2 离线表**：✅ 已于 2026-08-23 导入（`preprocessed/F2_financial_anomaly.csv` 67 维新列名 37222 行 + `raw/` 备份与构建脚本）；
2. **F1 原始表**：❌ 待队员提供 `F1_semantic_features.parquet`（或 `F1_base_financial.csv`）——公告语义 50 维，导入 `raw/` 后即可重建 `processed_dataset.csv`；
3. 重建管线：`build_modeling_dataset.py → train_models.py`（见 backend/scripts/），当前脚本读取的 RAW 目录在 `C:\Users\86130\Desktop\预测建模agent\01_原数据`，导入项目 raw/ 后需同步该脚本路径。

## 预测为何"实时为主、离线兜底"

- **实时**：模型要预测的是"当下/近期"的问询概率，实时特征（最新财报、最新行情、最新舆情、最新问询）比过期离线表更贴近 T 时点；
- **兜底**：部分特征实时拿不到（F1 语义 50 维需离线语义模型、governance_year 需年报审计信息），用训练集中位数填充——只影响该列贡献，不改变模型；
- **审计**：每次预测记录 `data_source`（realtime/offline_lookup）与 `coverage`（实时覆盖率），可在预测建模页审计。
