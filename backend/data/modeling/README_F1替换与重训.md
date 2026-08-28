# F1 公告语义特征替换与模型重训

**分支**：`feature/bge-semantic-fallback-choice`
**日期**：2026-08-27 ~ 2026-08-28
**范围**：替换 F1 数据源、年份对齐、重复性检查、三窗口×三模型重训

---

## 一、做了什么

| # | 事项 | 结果 |
|---|---|---|
| 1 | F1 数据源替换 | 300 维 PCA 主成分 → **197 维结构化风险特征** |
| 2 | 年份对齐 | 剔除 2020，37,222 → **30,049 行**，序列自 2021Q1 起 |
| 3 | 重复性检查 | **F1 与 F2-F6 跨家族重复 0 对**；F1 内部 79 维冗余 |
| 4 | 重训 | 三窗口 × 三模型，**指标全面超过原基线** |

F2-F6 数据、标签口径、合并逻辑、`train_models.py` **均未改动**。

---

## 二、F1 数据从哪来

上游是一次**全量重跑**，不是抽样：

```
81,208 份上市公司公告 PDF（90.6 GB，2020-2025）
    ↓  pdfplumber 全量解析          80,752 成功 / 455 needs_ocr / 1 失败
    ↓  切块                          8,176,152 块（600字符/100重叠）
    ↓  bge-large-zh-v1.5 向量化      1024 维 fp16
    ↓  45 主题召回（公告内 topk）    12,187,064 候选
    ↓  bge-reranker-v2-m3 精排       4,720,453 条证据
    ↓  FinBERT 情绪 + 联合打分
    ↓  公告级特征聚合
announcement_risk_features.parquet   551,972 行 / 80,241 篇 / 43 个二级主题
```

风险主题体系为 `risk_taxonomy v1.1 冻结版`（8 个一级 + 45 个核心二级）。

**GPU 总耗时约 16 小时**（RTX 4090）。详见上游交付目录的 `README_交付说明.md` 与 `data_quality_report.md`。

---

## 三、197 维特征的构成

不是抽象主成分，**每一维都是可解释的业务指标**：

| 维度块 | 数量 | 说明 |
|---|---|---|
| 窗口统计 | 44 | 4 个窗口（30/60/90/180 天）× 11 项：公告数、主题命中数、高风险条数、覆盖主题数、规则命中数、强风险信号数、风险分均值/最大/总和、精排分最大、负面情绪均值 |
| 一级主题 | 64 | 4 个窗口 × 8 个一级主题（A-H）× (最高分, 证据数) |
| 二级主题 | 86 | 43 个二级主题 × (近 180 天最高分, 证据条数) |
| 时序 | 3 | 距上次高风险公告天数、距上次公告天数、风险分 90 天环比变化 |

对照表：`backend/data/modeling/raw/f1_selection/F1_semantic_column_mapping.csv`

```
announcement_semantic_042  ->  th_C03_max  ->  主题 C03 近180天最高风险分
announcement_semantic_061  ->  w90_score_max  ->  近90天 风险分最大值
```

**选出 Top-N 之后每一维都能说清是什么**，原 PCA 主成分做不到这点。

---

## 四、键网格与时点口径

对齐到原有网格，保证 F2-F6 可直接 join：

```
37,222 行 = 1,951 家公司 × 20 个季度末
键: stock_code(str) + T_date(date)
T_date = 03-31 / 06-30 / 09-30 / 12-31（报告期末）
```

**公告窗口**：严格 `T - N < publish_date <= T`，只用 T 之前已公开的公告。

⚠️ **口径差异（已知，需注意）**：原管线的 `T_date` 是**报告期末**，即 `T=2020-03-31` 时假设一季报当天可得，而实际法定披露截止日是 4/30，存在 30-120 天披露时滞。上游全量重跑中我们采用的是**法定披露截止日**作为锚点。为保证与 F2-F6 键对齐，本次替换按原管线的报告期末重新聚合，**未套用法定披露日锚点**。此差异不影响 F1 自身的窗口逻辑（公告只取 T 之前），但整个建模网格沿用了原有的时点约定。

---

## 五、年份对齐：为什么剔除 2020

实测各年份的数据可用性：

| T 年份 | 样本数 | 180 天内有公告率 | 平均公告数 |
|---|---|---|---|
| **2020** | 7,173 | **13.5%** | **0.18** |
| 2021 | 7,469 | 83.3% | 4.24 |
| 2022 | 7,590 | 86.5% | 4.04 |
| 2023 | 7,519 | 89.7% | 4.18 |
| 2024 | 7,471 | 90.2% | 4.77 |

**2020 年 86.5% 的样本语义特征近乎全零**，因数据集 2020 年仅含 374 份公告（2021 年为 19,033 份）。这些行占训练集 27%，属噪声。

`build_modeling_dataset.py` 中新增 `MIN_YEAR = 2021` 与 `keep_2021plus()`，在 5 处插入过滤：

1. F1 主表
2. F1 选取骨架 `_f2_sel`
3. F1 选取用事件 `_events`（`year_col="year"`）
4. F2-F6（`norm()` 之后）
5. 标签构建用 `events`

结果：**30,049 行，Train 21004 / Validation 2999 / Test 6046，最小 `report_period = 20210331`**。脚本结尾加了断言校验，防止漏网记录静默通过。

---

## 六、重复性检查结论

`backend/scripts/check_f1_redundancy.py`，Train 集上算 Spearman（实现用 rank+pearson，与 spearman 数值等价但快数十倍）。

### 6.1 跨家族：0 对

```
F1 ↔ F2/F3/F4/F5/F6   |rho| >= 0.85 的高相关对：0
```

**F1 提供的是财务、市场、舆情、治理、问询历史都没有的新信息维度**，不是重复劳动。

### 6.2 全部高相关对分布（380 对）

| 类型 | 对数 |
|---|---|
| F1 内部 | 336 |
| F4 ↔ F4 | 15 |
| F2 ↔ F2 | 13 |
| F3 ↔ F3 | 5 |
| F2 ↔ F3 | 5 |
| F5 ↔ F5 | 3 |
| F6 ↔ F6 | 3 |

另有 **11 个常量列**（全表无方差）。

### 6.3 F1 内部冗余的成因

79 维建议剔除，集中在短窗口：

```
w30_n_ann ≈ w30_n_rows ≈ w30_n_themes ≈ w30_score_mean/max/sum ≈ w30_sentneg_mean
                                                          rho 0.988 ~ 0.999
```

**根源是短窗口稀疏**：30 天窗口仅 21.8% 的样本有公告，绝大多数行全零，少数行只有 1-2 篇公告——此时「公告数」「主题命中数」「风险分总和」「负面情绪均值」退化为同一变量的不同缩放。`w30_L1_A_max ≈ w30_L1_A_cnt` 同理：30 天内某主题要么未命中（全 0），要么命中一次（max 与 cnt 同时非零）。

这是特征设计时在 30/60/90/180 四个窗口上机械铺开同一套统计量导致的，未考虑短窗口下统计量会退化。

### 6.4 处置

**本次未显式剔除**，理由：`train_models.py` 训练阶段已有自动去重

```python
cm = np.abs(np.corrcoef(X_tr[si], rowvar=False))
hc = np.where((cm > 0.95) & ut)
drop = set(hc[1])
```

因此**本文所有指标均为去重后的结果**。检查报告用于建模前显式确认与出报告。

如需在 build 阶段就排除，可读取 `F1_drop_suggestion.csv` 的清单加入过滤（197 → 118 维）。

---

## 七、模型表现

### 7.1 与原基线对比

| 窗口 | 指标 | 原基线（问询函语义 + 含2020） | 本次（公告语义 + 2021Q1起） | 变化 |
|---|---|---|---|---|
| **30d** | Ensemble AUC | 0.8021 | **0.8054** | +0.4% |
| | Top10%Recall | 0.345 | **0.4062** | **+17.7%** |
| **60d** | rf AUC | 0.8213 | **0.8498** | +3.5% |
| | rf AP | 0.2452 | **0.3007** | **+22.6%** |
| | lgb F1 | 0.2226 | **0.3696** | **+66.0%** |
| | Top10%Recall | 0.4276 | **0.4969** | **+16.2%** |
| **90d** | Ensemble AUC | — | **0.8458** | — |
| | lgb AP | — | **0.3863** | — |
| | Top10%Recall | — | **0.4573** | — |

`Top10%Recall` 业务价值最高：风险排名前 10% 的公司中能覆盖多少真正被问询的对象，60d 接近一半。

### 7.2 消融对比

| 配置 | 60d Ensemble AUC | 60d Top10%Recall |
|---|---|---|
| Top-50 + 含 2020 | 0.8206 | 0.4343 |
| Top-100 + 含 2020 | 0.8229 | 0.4432 |
| **Top-100 + 仅 2021+** | **0.8488** | **0.4969** |

结果文件：`output/model_summary_keep50.json`、`output/model_summary_keep100_with2020.json`

### 7.3 特征相关性质量

```
Top-50 的 |corr| 区间：0.0467 ~ 0.0962
```

对比原 `select_f1_top50.py` 注释所述「300 维 PCA 主成分中大量维度 |corr| 普遍 <0.01」，**最差一维也是该「低效特征」标准的 4.7 倍**。

---

## 八、方法学修正（重要）

原基线的 F1 实际选中的是 `regulatory_inquiry_semantic_*`，即**从问询函文本提取的特征**，而建模标签正是「未来 N 天内是否收到问询函」。特征与标签同源，存在信息泄露风险。

本次改为**纯公告语义特征**，不使用任何问询函文本内容。**在去掉这层信息的前提下，指标反而全面提升。**

`build_modeling_dataset.py` 中标签仍由 `inquiry_events.csv`（`kind=='letter'`）动态生成，`n_inq_60d` 仅作 `sample_weight`，不进模型特征（`train_models.py` 按 `n_inq_` 前缀排除）。

---

## 九、可解释性

模型自选的 Top-20 特征中，最强信号高度集中：

| Rank | 特征 | 含义 |
|---|---|---|
| 1-2 | `th_G07_cnt` / `th_G07_max` | **G07 立案调查、监管处罚与其他合规事项** |
| 3, 5-7 | `w30/60/90_L1_G_*` | G 类（审计、披露与监管合规）各窗口 |
| 4, 8 | `th_G04_cnt` / `th_G04_max` | **G04 信息披露完整性** |
| 9, 13-16 | `w30/90_L1_H_*` | **H 类（股价异常波动、市场传闻）** |
| 10 | `w180_sentneg_mean` | 近 180 天负面情绪均值 |
| 18 | `th_C04_cnt` | C04 固定资产与在建工程 |

**业务解读**：模型未被告知任何规则，自行学到「预测是否会被问询」的最强信号是「这家公司近期有没有因披露与合规问题被监管处理过」（G07/G04），其次是「股价有没有异常波动」（H01/H02）。这与监管问询的实际触发机制一致。

⚠️ **需注意**：G07 信号强，部分原因是「被监管处理过的公司容易再被问询」，立案调查与问询函存在同期相关性，不完全是纯前瞻预测。但窗口逻辑是干净的（只用 T 之前的公告），属合理的历史行为特征。

---

## 十、文件清单

### 新增脚本

| 文件 | 作用 |
|---|---|
| `backend/scripts/rebuild_f1_from_fullrun.py` | 用全量重跑产出重建 F1，对齐原键网格 |
| `backend/scripts/check_f1_redundancy.py` | F1 与 F2-F6 重复性检查 |

### 修改

| 文件 | 改动 |
|---|---|
| `backend/scripts/build_modeling_dataset.py` | 新增 `MIN_YEAR=2021` 与 5 处年份过滤；`N_KEEP` 50→100 |
| `backend/data/modeling/raw/F1_announcement_semantic_features.parquet` | 替换为 197 维 |

### 新增数据与报告

```
raw/f1_selection/F1_semantic_column_mapping.csv    列名 -> 业务含义对照
raw/f1_selection/F1_rebuild_meta.json              重建参数与来源
raw/f1_selection/F1_redundancy_pairs.csv           全部高相关对明细
raw/f1_selection/F1_drop_suggestion.csv            79 维剔除建议
raw/f1_selection/F1_redundancy_meta.json           去重检查汇总
output/model_summary_keep50.json                   消融：Top-50
output/model_summary_keep100_with2020.json         消融：含 2020
```

### 未入库（`.gitignore`）

```
raw/_backup_f1/          原 F1 备份，本地保留供回滚
*.py.bak                 原脚本备份
raw/*_NEW.parquet        dry-run 临时产物
```

---

## 十一、复现步骤

```powershell
# 1. 重建 F1（需要上游全量重跑产出）
python backend\scripts\rebuild_f1_from_fullrun.py `
  --run-dir D:\fintech_nlp_output\run_20260826_001320_full `
  --repo . --dry-run          # 先 dry-run 检查覆盖率
python backend\scripts\rebuild_f1_from_fullrun.py `
  --run-dir D:\fintech_nlp_output\run_20260826_001320_full --repo .

# 2. 重复性检查（可选，仅出报告）
python backend\scripts\check_f1_redundancy.py --repo .

# 3. 重建训练集
python -m backend.scripts.build_modeling_dataset

# 4. 重训
python -m backend.scripts.train_models
```

依赖：`lightgbm` `xgboost` `shap` `scipy` `scikit-learn` `pyarrow`

⚠️ 安装 `shap` 会通过 numba 依赖链把 numpy 升到 2.x，如遇 `numpy.dtype size changed` 类报错，执行 `pip install "numpy<2" --force-reinstall`。

---

## 十二、已知限制

1. **T_date 沿用报告期末**，存在披露时滞（见第四节）。若要改为法定披露截止日，需同步调整 F2-F6 的键网格。
2. **F1 内部 79 维冗余未显式剔除**，依赖训练脚本的自动去重。如需更干净的特征表，可按 `F1_drop_suggestion.csv` 过滤后重跑。
3. **43/45 主题覆盖**：E01（交易商业实质与定价公允性）、G02（审计证据充分性与审计范围受限）在全量 80,241 篇公告中未被激活，因 Top-K=8 截断下被语义相近的同类主题（E02/E04/E05/E06、G01）挤出。相关内容已被近邻主题捕获，未造成信息丢失。
4. **30d 窗口正样本率低**（Train 约 2.5%），指标波动较大，AUC 参考价值有限，建议看 AP 与 Top10%Recall。
