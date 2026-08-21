# 上市公司监管问询扫雷预警系统

基于 Agentic AI 的上市公司监管问询概率预测与扫雷预警（东吴证券赛题）。

## 目录结构

```
backend/          后端核心
├── config.py     全局配置（路径/阈值/模型开关，单一来源）
├── llm.py        共享 LLM 客户端（deepseek-v4-flash，一行 import 即用）
├── context.py    共享 Context（Agent 间唯一通信）
├── agents/       7 个 Agent + 基类 + 主控编排
│   ├── base.py                AgentBase（统一 execute(company, ctx) + trace）
│   ├── announcement_reader.py 公告研读（巨潮在线 + 规则/FinBERT/LLM + 可审计 F1）
│   ├── financial_detector.py  财务检测（F2 67 维 + F3 35 维 + 双负兜底 + F2 规则）
│   ├── case_retriever.py      案例检索（4785 案例库 + RRF + 三源标签通道 + 维度守卫）
│   ├── attributor.py          归因解释（SHAP + 证据白名单 + 案例链接）
│   └── orchestrator.py        主控编排（完整流水线 + 确定性 ReAct）
│   └── predictor.py           （待接入队友训练好的 LightGBM 模型）
├── skills/       原子能力
│   ├── announcement_search.py 巨潮在线公告检索、官方 PDF 下载与本地副本兼容
│   ├── ocr_extract.py         扫描型 PDF 按页 OCR 与审计元数据
│   ├── finbert_classify.py    FinBERT2-base 风险粗分类（门控）
│   ├── rule_risk_extract.py   规则风险抽取（官方 risk_dictionary.yaml）
│   ├── risk_labels.py         官方标签体系加载（任务1交付包 A-H/45 二级）
│   ├── concern_store.py       关注点词典（规则类别→官方关注点词汇，覆盖率 55.9%）
│   ├── embedding.py           统一 Embedding（bge 1024 维 / fallback 兜底）
│   ├── vector_store.py        向量库（numpy，案例库读写 + meta 校验）
│   ├── f2_calc.py             F2 67 维特征计算（Beneish/Piotroski/Benford/行业偏离）
│   ├── market_fetch.py        F3 35 维市场特征（腾讯 K 线在线）
│   └── financial_data_fetch.py 东财+腾讯行情爬虫
├── models/       模型权重与训练产物（embedding/finbert/predictor）
├── data/
│   ├── vector_db/             ★ 案例库：case_db.json（4785 份问询函）+ case_vectors.npy（BGE 1024 维）+ case_meta.json
│   ├── labels/                ★ 官方标签资产：risk_taxonomy / risk_dictionary / 分类结果 / concern_dict
│   ├── index/                 公告索引缓存
│   └── output/                报告 / 样例 context / 交付文档
├── scripts/      离线任务
│   ├── build_case_db.py       ★ 重建案例库（02_监管问询 源头 JSONL + 官方 GT）
│   ├── build_case_vector_db.py  （旧版，按公司构建，保留参考）
│   ├── train_predictor.py     训练预测模型（骨架）
│   └── build_concern_dict.py  关注点词典构建（骨架）
└── tests/        单元测试
app.py            全流程 Streamlit 演示入口
streamlit_app.py  公告研读 Agent 独立可审计展示页
公告研读agents/    旧版独立实现（保留参考，功能已迁入 backend/agents/）
context/          公告研读输出的共享 context 样例（000004.SZ）
无关文件夹/        与系统无关的历史文件
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt        # 含 pandas/numpy/scipy/requests/pymupdf
# 2.（可选）配置 LLM：复制 .env.example 为 .env，填 DEEPSEEK_API_KEY
# 3. 一键演示（公告研读 → 财务检测 → 案例检索 → 归因）
python -m backend.agents.orchestrator   # 000004.SZ 全流程自测
# 4. 公告研读 Agent 独立页面（输入代码或公司名称）
streamlit run streamlit_app.py
```

## 数据产物（非官方源文件，已内置）

| 数据 | 位置 | 说明 |
|------|------|------|
| 案例库（**4785 份问询函**） | `backend/data/vector_db/` | case_db.json + case_vectors.npy（BGE 1024 维，队友交付）+ case_meta.json |
| 关注点词典 | `backend/data/labels/concern_dict.json` | 规则类别 → 官方关注点词汇（标签通道覆盖率 1.3%→55.9%） |
| 官方标签体系 | `backend/data/labels/` | risk_taxonomy（A-H/45 二级）、risk_dictionary、分类结果 10481 条 |
| 公告索引（000004） | `backend/data/index/` | 其余公司首次运行自动建 |
| 样例 context | `backend/data/output/sample_context_000004.json` | 共享 Context 结构参考 |

> 源头文件在 `D:\BaiduNetdiskDownload` 与 `D:\新建文件夹\02_监管问询`（结构化 JSONL），
> 重建案例库：`python -m backend.scripts.build_case_db --force --backend bge`

## 开发状态

- [x] 公告研读 Agent（近一年巨潮公告 + OCR + 三通道抽取 + 30/60/90 天 F1）
- [x] 财务检测 Agent（F2 67 维 + F3 35 维 + 双负兜底，110 特征）
- [x] 案例检索 Agent（4785 案例库 + RRF + 三源标签通道 + 维度守卫）
- [x] 归因解释 Agent（SHAP + 证据白名单 + 案例链接）
- [x] 主控编排 Agent（完整流水线 + trace）
- [ ] 预测建模 Agent（待接入队友 LightGBM 模型：predictor.py + modeling_dataset.parquet + models/）
- [ ] 语义通道 BGE（`EMBEDDING_BACKEND=bge`，需下载权重；当前维度守卫自动禁用语义通道、仅标签通道）
- [x] 公告研读 Streamlit 独立展示页（证据、数据质量、窗口和主题对比）

## 说明

- **案例检索维度守卫**：案例库为 BGE 1024 维，当前 embedding 后端为 fallback（65536 维）时语义通道自动禁用、仅标签通道工作（打印提示，不报错）；切 `EMBEDDING_BACKEND=bge` 后自动恢复语义检索，无需改代码。
- **防幻觉约定**：所有 LLM 生成内容绑定原文证据 ID；证据一律取原文（公告/问询函原句），不存 LLM 转述；`evaluation_ground_truth` 仅用于归因评估，不作预测特征。
