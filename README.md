# 上市公司监管问询扫雷预警系统

> 基于 Agentic AI 的上市公司监管问询概率预测与扫雷预警 · 东吴证券「智能风控与量化建模」赛题
>
> FastAPI 后端 + 多 Agent 流水线 + 单文件前端 + WebSocket 实时进度，支持离线快照秒出与实时扫雷双模式。

---

## 一、整体架构

```
┌──────────────────────────────────────────────────────────┐
│  浏览器 api/static/index.html （单文件前端）            │
│  全局状态 + 发布订阅 · 6 Agent 流水线可视化 · 风险仪表盘 │
│  WebSocket 进度 · 实时扫雷开关 · 离线兜底提示条          │
└───────────────┬────────────────────────┬─────────────────┘
        fetch /api/*               ws /ws/pipeline/{id}
                ▼                            ▼
┌──────────────────────────────────────────────────────────┐
│  FastAPI 后端  api/main.py                              │
│  路由 · CORS · 静态挂载 · WebSocket 推送                │
└───────────────┬──────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────┐
│  任务调度层  api/pipeline.py                             │
│  串行锁 · 任务取消 · 内存缓存(LRU) · 离线兜底 · 终态机   │
│  ProgressMessage 消息：progress/complete/error/fallback/ │
│  cancelled · agent_key · progress_percent · fatal       │
└───────────────┬──────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────┐
│  Agent 流水线  backend/agents/  （6 个 Agent + 编排器）   │
│  公告研读 → 财务异常 → 预测建模 → 案例匹配 → 归因分析     │
│  → 报告生成  （SweepingOrchestrator 串行编排 + 进度回调） │
└───────────────┬──────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────┐
│  Skills 能力层  backend/skills/                          │
│  公告检索/OCR/Embedding/案例向量/财务抓取/特征加载…      │
└──────────────────────────────────────────────────────────┘
```

- **离线模式**：POST `/api/scan` 默认秒返离线快照（`cached:true`），不建任务、不连 WebSocket。
- **实时模式**：`realtime:true` 创建 Agent 任务，WebSocket 实时推送每个 Agent 的开始/进度/完成；失败自动回退离线快照（`fallback` 消息）。
- **串行锁**：同一时间只跑一个实时任务，切换公司时先 `DELETE` 取消旧任务，冲突返回 `409` 由前端弹框确认后 `force:true` 重发。

## 二、目录结构

```
competition_agent/
├── api/                      # FastAPI 后端
│   ├── main.py               # 路由入口（10 个端点 + WebSocket + 静态挂载）
│   ├── pipeline.py           # 任务调度：串行锁/取消/缓存/兜底/消息推送
│   ├── models.py             # Pydantic 数据模型（与前端 JS 结构对齐）
│   ├── requirements.txt      # 后端依赖
│   ├── static/
│   │   └── index.html        # 前端单文件（全局状态 + 6 Agent 可视化）
│   └── tests/                # 后端运行时测试
├── backend/
│   ├── config.py             # 全局配置（路径/参数/模型，单一来源）
│   ├── context.py            # 公司分析上下文
│   ├── llm.py                # LLM 调用封装
│   ├── dashboard_utils.py    # 仪表盘数据组装
│   ├── agents/               # Agent 流水线
│   │   ├── base.py           # AgentBase（含 progress_callback 进度回调）
│   │   ├── orchestrator.py   # SweepingOrchestrator 串行编排
│   │   ├── announcement_reader.py   # 公告研读 Agent
│   │   ├── financial_detector.py    # 财务异常 Agent
│   │   ├── predictor.py            # 预测建模 Agent（SHAP 归因）
│   │   ├── case_retriever.py       # 案例匹配 Agent（RRF 融合）
│   │   ├── attributor.py           # 归因分析 Agent
│   │   ├── reporter.py             # 报告生成 Agent
│   │   ├── graph.py / risk_mapper.py / label_keywords_v2.py
│   ├── skills/               # 能力层（公告检索/OCR/embedding/财务/特征…）
│   │   └── stock_code.py     # 股票代码归一化
│   ├── data/                 # 离线快照/向量库/模型/标签
│   ├── models/               # 预测模型（predictor/生存模型）
│   └── tests/                # 各 Agent 单测
├── docs/                     # 设计文档与 specs
├── .env / .env.example       # 环境变量（密钥与公告研读配置）
├── requirements.txt          # 项目依赖
└── README.md
```

## 三、后端模块

| 模块 | 职责 |
|------|------|
| `api/main.py` | FastAPI 入口，10 个路由 + WebSocket + 静态挂载，CORS 全开 |
| `api/pipeline.py` | 任务调度核心：`_scan_lock` 串行锁、`cancel_task` 取消、`_result_cache` LRU 缓存、`get_offline_result` 离线快照、`_fallback_after_error` 失败兜底、`run_scan_task` 实时流水线、`ProgressMessage` 消息推送 |
| `api/models.py` | Pydantic 模型：`AnalyzeRequest/Response`、`ScanRequest{realtime,force}`、`ScanResponse{cached,result}`、`ProgressMessage{type,agent_key,progress_percent,fatal}` |
| `backend/agents/*` | 6 个 Agent + 编排器，`AgentBase` 提供 `_report_progress` 细粒度进度回调 |
| `backend/skills/*` | 公告检索(cninfo)、OCR、Embedding(bge/fallback)、案例向量、财务抓取、特征加载、风险词典 |
| `backend/config.py` | 全局配置单一来源，路径/阈值/模型路径集中，业务代码禁止硬编码 |

## 四、前端模块（`api/static/index.html`）

单 HTML 文件（原生 JS，无构建工具），核心机制：

- **全局状态** `AppState`：`currentCompany`/`scanStatus`/`currentTaskId`/`agentStates`/`agentResults`/`resultCache`/`realtimeMode`，带发布订阅 `subscribe/set`。
- **API 调用**：`fetchSingleCompany`(→GET `/api/company/{code}`)、`fetchAnalysis`(→POST `/api/analyze`)、`startRealtimeScan`(→POST `/api/scan` + WS + DELETE 取消)。
- **WebSocket**：`/ws/pipeline/{taskId}`，`onmessage` 按 `type` 分发（progress/complete/error/fallback/cancelled/heartbeat），断线指数退避重连，消息 id 去重。
- **6 Agent 流水线可视化**：左侧 Agent 卡片灰→蓝(脉冲)→绿(完成)，顶部 5 步步骤条，全局进度条。
- **主控仪表盘**：问询概率大数字(变色)、风险等级徽章、SHAP 因子卡、风险证据表(可排序)、执行摘要、4 项模型指标卡。
- **实时扫雷开关 + 离线兜底**：默认离线秒出，开关开启走真实流水线，失败黄色兜底条 + 自动回退离线。
- **前端结果缓存** `resultCache`：看过的公司秒出（与后端 LRU 双重保障）。

## 五、环境配置

复制 `.env.example` 为 `.env`，填入真实密钥与配置：

| 变量 | 说明 | 默认/示例 |
|------|------|-----------|
| `LLM_PROVIDER` | LLM 供应商：deepseek / anthropic / openai | deepseek |
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | sk-xxx |
| `DEEPSEEK_BASE_URL` | DeepSeek 接口地址（可接本地代理） | https://api.deepseek.com |
| `DEEPSEEK_MODEL` | 模型名 | deepseek-v4-flash |
| `ANTHROPIC_API_KEY` | Anthropic 密钥（选用） | sk-ant-xxx |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | OpenAI 密钥/地址（选用） | — |
| `ANNOUNCE_SOURCE` | 公告数据源 | cninfo |
| `ANNOUNCE_MAX_DOCUMENTS` | 公告研读最大文档数 | 120 |
| `ANNOUNCE_OFFLINE_ENABLED` | 启用离线公告快照 | true |
| `OCR_ENABLED` | 启用 OCR | true |
| `EMBEDDING_BACKEND` | 向量后端：bge(需权重,与案例库兼容) / fallback(零依赖) | fallback |
| `EMBEDDING_ALLOW_DOWNLOAD` | 是否允许下载大模型 | false |
| `FINBERT_ENABLED` / `FINBERT_GATE_ENABLED` | FinBERT 门控（未校准前关闭） | false |
| `API_KEY` | 可选 API 鉴权密钥；设置后 `/api/*` 需带 `X-API-Key` 头、WS 握手需带 `?token=` | 空（不启用） |

> 数据路径默认读仓库内「公告解析」可移植数据包；本机持有完整外部数据集时可设 `COMPETITION_DATA_ROOT` 覆盖。

## 六、启动方式

官方要求交付包集中在 [`官方要求交付物/`](官方要求交付物/)，包含 API、批量结果脚本、审计日志规范、典型案例、算法架构、评估和部署说明；技术答辩 PPT 后续放入其中的 `00_技术答辩PPT/`。

```bash
# 1. 安装依赖
cd D:/competition_agent
pip install -r requirements.txt

# 2. 配置 .env（见上节）

# 3. 启动后端（FastAPI + WebSocket）
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
#   或 python api/main.py

# 4. 打开前端
#    浏览器访问 http://127.0.0.1:8000/
#    接口文档 http://127.0.0.1:8000/docs
```

## 七、API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 返回前端 `index.html` |
| GET | `/api/health` | 健康检查 |
| GET | `/api/companies` | 所有有离线报告的公司列表 |
| GET | `/api/agents` | 7 个 Agent 的元数据（前端流水线清单单一来源） |
| POST | `/api/analyze` | 批量风险分析（离线） |
| GET | `/api/company/{code}` | 单公司离线查询 |
| POST | `/api/scan` | 实时扫雷：`realtime=false` 秒返离线快照(cached)，`true` 创建任务，冲突返回 409 |
| DELETE | `/api/scan/{task_id}` | 取消实时任务 |
| GET | `/api/scan/{task_id}` | 查询任务状态（断线重连补历史） |
| GET | `/api/mock/{code}` | 联调用：有离线快照优先返回真实，否则 404 |
| WS | `/ws/pipeline/{task_id}` | 实时推送进度（progress/complete/error/fallback/cancelled/heartbeat） |

## 八、核心功能

1. **6 Agent 流水线**：公告研读→财务异常→预测建模→案例匹配→归因分析→报告生成，串行编排 + 细粒度进度回调。
2. **问询概率预测**：XGBoost 三模型集成（30/60/90d），SHAP 因子归因，可选 XGBoost-Cox 生存模型。
3. **风险证据表**：公告研读 Agent 输出日期/等级/L1L2/描述/原文证据，支持排序筛选。
4. **案例语义匹配**：BGE 向量检索 + RRF 融合得分，Top-5 相似历史问询案例。
5. **离线 + 实时双模**：离线秒出保证演示流畅，实时模式展示真实流水线，失败自动兜底。
6. **任务管理**：串行锁防并发、取消接口、内存 LRU 缓存、断线重连。

## 九、演示流程

1. 打开页面 → 默认离线模式秒出五粮液主控仪表盘。
2. 顶部「实时扫雷」开关 → 蓝色脉冲，进入实时模式。
3. 切换公司（搜索/快捷标签）→ 旧任务自动取消，新任务 WebSocket 推送 Agent 进度。
4. 连续点击「开始扫雷」→ 后续触发 409 弹框确认。
5. 实时失败（断网）→ 黄色兜底条 + 自动回退离线快照。
6. 各 Agent 详情页：财务异常指标卡、案例匹配 Top-5 卡片、归因贡献度。

---

## 说明

- **Embedding**：默认 `EMBEDDING_BACKEND=bge`（BGE-large-zh-v1.5，1024 维）；离线演示默认不自动下载缺失权重，语义通道不可用时回退标签检索。确需下载时设置 `EMBEDDING_ALLOW_DOWNLOAD=true`。
- **公告误报过滤**：公司章程、议事规则、候选人声明和通用管理制度只保留官方元数据，不下载 PDF、不进入风险抽取；真实处罚、立案、辞职、冻结等风险标题优先保留。
- **历史与当前分层**：比赛历史库是 2020—2024 年历史研究产物，旧规则命中只作待复核候选；当前事实和 30/60/90 天统计仍只使用巨潮官方公告及 PDF。
- **事实语境校验**：法规引用、董监高职责/任职资格、禁止性或假设性条款、会计政策及报表模板会记录过滤原因，但不计入风险事件。LLM 是可选精细通道，输出仍需通过逐字证据和事实语境双重校验。
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
11. **预测建模实时化（查表 → 实时推理）**：
    - `PredictorAgent` 主路径改为**实时特征推理**：以公告研读 F1 标量 + 财务异常 F2-F6 实时值为数据源，按 `models_manifest.json` 特征清单组装向量，缺失列（如 F1 语义 50 维、governance_year）用训练集中位数兜底（新增 `backend/data/modeling/fill_median_{30,60,90}d.csv`）；
    - 新增 `backend/skills/feature_composer.py`（实时特征组装器），预测结果带审计字段 `data_source=realtime/offline_lookup` 与 `coverage`（实时覆盖率）；无实时财务数据时自动回落查表路径。
12. **F2 特征列与训练表对齐**：`f2_calc.py` 删除恒 NaN 占位列（f2_gross_margin/f2_current_ratio/f2_interest_coverage），新增 `f2_neg_pe_flag/f2_neg_pb_flag/f2_market_cap_quintile`（负 PE/PB 标志 + 市值五分位），与建模数据集/模型 manifest 完全一致。
13. **F4/F5/F6 实时化分工**：F4 股吧舆情 / F5 股东治理在线爬取优先（每族 30s 超时，失败回退离线表）；**F6 监管问询函特征改由公告研读 Agent 提供**（新增 `backend/skills/inquiry_features.py`：从巨潮公告列表识别问询函/关注函，口径与离线 F6 表一致），财务侧不再输出 F6。
14. **流水线稳定性**：公告 PDF 下载加总预算（180s/轮）；LangGraph 图节点级看门狗（daemon 线程 + join(timeout)，节点永不永久挂起）；BGE 模型改为**本地快照直载**（修复 HF 缓存元数据损坏导致的联网重试风暴；实测 000004.SZ 全流程约 3 分钟有界完成，预测 `data_source=realtime`、60d 概率 0.314）。

*本项目为研究生创新大赛参赛作品，东吴证券赞助。*
