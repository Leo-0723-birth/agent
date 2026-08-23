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
│   ├── chunk_retriever.py     chunk 级段落检索（段落证据召回）
│   ├── predictor.py           预测建模（三模型集成 30/60/90d + SHAP，查表推理）
│   ├── attributor.py          归因解释（SHAP + 证据白名单 + validate_narrative 防幻觉）
│   └── orchestrator.py        主控编排（LangGraph 图编排首选，确定性串行兜底）
│   └── graph.py               LangGraph 7 节点 StateGraph（可选 checkpointer 断点续跑）
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
│   ├── vector_db/             ★ 案例库：case_db.json（4785 份问询函）+ case_vectors.npy（BGE 1024 维）+ case_meta.json + chunk 段落库
│   ├── labels/                ★ 官方标签资产：risk_taxonomy / risk_dictionary / 分类结果 / concern_dict
│   ├── modeling/              ★ 建模数据：processed_dataset.csv + 三窗口预测/SHAP/风险排序输出
│   ├── index/                 公告索引缓存
│   └── output/                报告 / 样例 context / 交付文档
├── scripts/      离线任务
│   ├── build_case_db.py       重建案例库（02_监管问询 源头 JSONL + 官方 GT）
│   ├── enrich_case_db_reply.py 案例库补回复要点（4714/4785）
│   ├── build_chunk_index.py   构建 chunk 段落向量索引
│   ├── evaluate_case_retriever.py  案例检索 Top-5 命中率评测
│   ├── build_modeling_dataset.py   建模数据合并（F1 Top-50 + F2-F6 + 标签）
│   ├── train_models.py        三模型 × 三窗口训练 + SHAP（RF/LGB/XGB）
│   ├── select_f1_top50.py     F1 语义特征 Spearman Top-50 选取
│   └── build_concern_dict.py  关注点词典构建（骨架）
└── tests/        单元测试
导航入口.py       单入口导航壳（st.navigation，一个端口聚合全部页面，默认打开主控）
启动导航入口.bat   一键启动导航入口（双击即用，终端显示 Local/Network/External 网址）
主控agent.py       全流程 Streamlit 演示页（主控编排 Agent，经 导航入口.py 打开，也可独立运行）
公告研读agent.py   公告研读 Agent 独立可审计展示页（端口 8502）
财务异常agent.py   财务异常 Agent 独立审计页（端口 8503）
预测建模agent.py   预测建模 Agent 独立审计页（端口 8504）
案例匹配agent.py   案例匹配 Agent 独立审计页（端口 8505）
归因分析agent.py   归因分析 Agent 独立审计页（端口 8506）
报告生成agent.py   报告生成 Agent 独立审计页（端口 8507）
公告研读agents/    旧版独立实现（保留参考，功能已迁入 backend/agents/）
context/          公告研读输出的共享 context 样例（000004.SZ）
无关文件夹/        与系统无关的历史文件
```
docs/竞赛技能包/    开发技能包（领域知识/数据特征/建模评估/Agent编排/开发验收/提示词模板）
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt        # 含 pandas/numpy/scipy/requests/pymupdf
# 2.（可选）配置 LLM：复制 .env.example 为 .env，填 DEEPSEEK_API_KEY
# 3. 一键演示（公告研读 → 财务检测 → 案例检索 → 归因）
python -m backend.agents.orchestrator   # 000004.SZ 全流程自测
# 4. 单入口导航（推荐，一个端口切换全部页面）
streamlit run 导航入口.py               # http://localhost:8501 侧边栏切换 7 页
#    或双击 启动导航入口.bat（终端窗口显示 Local/Network/External 网址，便于分享给他人）
#    或在 PyCharm 运行配置中选择「导航入口 (streamlit)」点 ▶
# 5. 各 Agent 独立审计页面（端口固定，便于联调）：
streamlit run 公告研读agent.py --server.port 8502
streamlit run 财务异常agent.py --server.port 8503
streamlit run 预测建模agent.py --server.port 8504
streamlit run 案例匹配agent.py --server.port 8505
streamlit run 归因分析agent.py --server.port 8506
streamlit run 报告生成agent.py --server.port 8507
#    独立页面仅在需要"逐 Agent 审计"时使用；日常演示只用 8501 导航入口即可
```

## 数据产物（非官方源文件，已内置）

| 数据 | 位置 | 说明 |
|------|------|------|
| 案例库（**4785 份问询函**） | `backend/data/vector_db/` | case_db.json + case_vectors.npy（BGE 1024 维，队友交付）+ case_meta.json |
| 关注点词典 | `backend/data/labels/concern_dict.json` | 规则类别 → 官方关注点词汇（标签通道覆盖率 1.3%→55.9%） |
| 官方标签体系 | `backend/data/labels/` | risk_taxonomy（A-H/45 二级）、risk_dictionary、分类结果 10481 条 |
| 公告索引（000004） | `backend/data/index/` | 其余公司首次运行自动建 |
| 样例 context | `backend/data/output/sample_context_000004.json` | 共享 Context 结构参考 |
| 建模数据集 | `backend/data/modeling/processed_dataset.csv` | 37222×204（F1 Top-50 + F2-F6 + 30/60/90d 标签） |
| 预测模型 | `backend/models/predictor/` | RF/LGB/XGB × 30/60/90d + models_manifest.json |

> 源头文件在 `D:\BaiduNetdiskDownload` 与 `D:\新建文件夹\02_监管问询`（结构化 JSONL），
> 重建案例库：`python -m backend.scripts.build_case_db --force --backend bge`

## 开发状态

- [x] 公告研读 Agent（近一年巨潮公告 + OCR + 规则/FinBERT/LLM 三通道 + 30/60/90 天 F1）
- [x] 财务检测 Agent（F2 67 维 + F3 35 维 + 双负兜底 + F2 规则，110+ 特征）
- [x] 预测建模 Agent（三模型集成 RF/LGB/XGB × 30/60/90d，60d AUC 0.8335 / Top10% 46.1%）
- [x] 案例检索 Agent（4785 案例库 + RRF + 三源标签通道 + 时间穿越控制 + 回复要点）
- [x] chunk 段落检索 Agent（段落级证据召回）
- [x] 归因解释 Agent（SHAP + 证据白名单 + validate_narrative 防幻觉 + 单测）
- [x] 报告生成 Agent（Markdown/JSON 六章报告）
- [x] 主控编排 Agent（7 环节完整流水线 + trace，7-Agent 全闭环）
- [x] 公告研读 Streamlit 独立展示页 + 全流程 主控agent.py
- [ ] 正样本率提升（月度采样/多标签，见 backend/data/output/正样本率提升方案.md）

## 环境修复（常见问题）

| 问题 | 修复 |
|------|------|
| 首次启动卡在 `Welcome to Streamlit! ... Email:` 输入（服务不启动、网页拒绝连接） | 已内置 `.streamlit/config.toml`（`gatherUsageStats=false`）关闭引导；若仍出现，删除 `C:\Users\<你>\.streamlit\credentials.toml` 后重试 |
| `.bat` 双击后中文乱码/启动失败 | 批处理必须 CRLF 换行 + GBK 编码（本项目的 启动导航入口.bat 已按要求生成；不要用编辑器改成 UTF-8/LF） |
| `rapidocr` / `onnxruntime` 未安装（扫描型 PDF OCR 不可用） | `pip install "rapidocr>=3.9,<4" "onnxruntime>=1.29,<2"` |
| `xgboost` 未安装（预测三模型集成缺第三腿） | `pip install "xgboost>=2.0"`（lightgbm 同理已内置） |
| FinBERT 权重不完整（`models/embedding/hub` 下存在 `.incomplete`） | 删除该模型目录后重下：`Remove-Item backend\models\embedding\hub\models--valuesimplex-ai-lab--FinBERT2-base -Recurse -Force`，然后运行一次任意启用 FinBERT 的页面（走 hf-mirror.com，约 1.3GB），或直接 `python -c "from backend.skills.finbert_classify import FinBERTClient; FinBERTClient()"` |
| 预测显示"未预测" | 确认公司代码在 `backend/data/modeling/processed_dataset.csv` 内（如 000004.SZ）；`models/predictor/` 下 9 个模型文件齐全时自动推理 |
| Windows 下随机森林推理报 `WinError 5` | predictor.py 已内置 `n_jobs=1` 修复（单条样本推理不需要多线程） |

## 说明

- **Embedding**：默认 `EMBEDDING_BACKEND=bge`（BGE-large-zh-v1.5，1024 维，权重在 backend/models/embedding/，加载失败自动回落 fallback 且维度守卫拦截语义通道）。
- **防幻觉约定**：所有 LLM 生成内容绑定原文证据 ID；证据一律取原文（公告/问询函原句），不存 LLM 转述；`evaluation_ground_truth` 仅用于归因评估，不作预测特征。
- **预测模型指标**（测试集集成）：30d AUC 0.805 / 60d 0.8335 / 90d 0.830；Top10% 召回 38.0%/46.1%/43.2%。

## 更新记录（2026-08-22）

1. **预测建模接入完善**：修复 `backend/agents/predictor.py`（RF 推理固定 `n_jobs=1` 规避 Windows WinError 5 + 逐模型 try/except 容错），补装 `xgboost`，三模型集成（RF/LGB/XGB）实测生效（000004.SZ 60d 概率 0.3822，Predictor 环节由 skipped → done，归因升级为 SHAP 归因）。
2. **文件重命名**：`app.py` → `主控agent.py`；`streamlit_app.py` → `公告研读agent.py`（中文文件名不影响运行；对应测试 `backend/tests/test_streamlit_app.py` 已同步）。
3. **新增 5 个 Agent 独立审计页**（项目根目录，与 公告研读agent.py 同层）：财务异常 / 预测建模 / 案例匹配 / 归因分析 / 报告生成，每页可独立 `streamlit run --server.port XXXX` 运行。
4. **单入口导航**：新增 `导航入口.py`（`st.navigation` 聚合全部 7 页，一个端口侧边栏切换），端口方案：主控 8501 / 公告研读 8502 / 财务异常 8503 / 预测建模 8504 / 案例匹配 8505 / 归因分析 8506 / 报告生成 8507。
5. **启动方式精简**：新增 `启动导航入口.bat`（双击启动，终端显示 Local/Network/External 网址）；删除冗余的 `启动全部Agent页面.ps1`；新增 PyCharm 运行配置 `.run/导航入口.run.xml`（配置名「导航入口 (streamlit)」）。
6. **环境修复**：FinBERT2-base 权重完整重下（451MB，修复 `.incomplete`）；安装 `rapidocr 3.9.2` + `onnxruntime 1.29.0`（扫描 PDF OCR 可用）；新增 `.streamlit/config.toml` 关闭 Streamlit 首次运行邮箱引导（否则非 headless 启动会卡在 `Email:` 输入导致端口不绑定）。
7. **requirements.txt**：`lightgbm` / `xgboost` 转正为正式依赖（预测集成所需）。
8. **财务检测不再跳过金融/地产**：`SPECIAL_INDUSTRY_PROFILES` 中金融业/房地产业改为参与常规异常检测（实测平安银行 000001.SZ：跳过=False、风险等级=中、2 条异常）。
9. **页面日期默认值修正**：预测建模 / 报告生成页面 `date_input` 默认值由 2025-12-02 改为当天，可直接选择近期日期（`max_value=date.today()`）。
10. **LangChain + LangGraph 接入**（锁 1.x，未推送前本地已验证）：
    - `backend/llm.py` 双通道：LangChain `ChatDeepSeek` 首选（异常自动回落 requests 直连），`chat/chat_json` 签名不变 → 全部 Agent 调用方零改动；
    - 新增 `chat_structured`：Pydantic + JSON Schema 约束的结构化输出（DeepSeek thinking 模式不支持 tool_choice，故走 `response_format=json_object` 兼容路径）；
    - 新增 `backend/agents/graph.py`：LangGraph `StateGraph` 7 节点流水线（announcement→financial→predictor→case→chunk→attribution→report），`ctx` 即 State，trace 格式不变；可选 `MemorySaver` checkpointer（断点续跑/回放，实测 9 个历史快照）；
    - `orchestrator.py` 改薄封装：`sweep_one/execute` 首选 `graph.invoke()`，LangGraph 不可用时回落原确定性串行链——**对外接口与页面全部零改动**；
    - 依赖锁定（requirements.txt）：`langchain-core==1.5.3` / `langchain-deepseek==1.1.0` / `langgraph==1.2.10` / `langgraph-checkpoint==4.2.0`。
    - 验证：全流程 7 节点 done（预测 0.3822）、6 页面 AppTest 通过、pytest 5 passed、单 Agent 独立可用。
