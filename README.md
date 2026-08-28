# 上市公司监管问询扫雷预警系统

> 基于 Agentic AI 的上市公司监管问询概率预测与扫雷预警 · 东吴证券「智能风控与量化建模」赛题
>
> FastAPI 后端 + 7 Agent 流水线 + 单文件前端 + WebSocket 实时进度，支持离线快照秒出与实时扫雷双模式。

---

## 一、整体架构

```
┌──────────────────────────────────────────────────────────┐
│  浏览器 api/static/index.html （单文件前端）            │
│  全局状态 + 发布订阅 · 7 Agent 流水线可视化 · 风险仪表盘 │
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
│  Agent 流水线  backend/agents/  （7 个 Agent + 编排器）   │
│  公告研读 → 财务异常 → 预测建模 → 案例匹配 → 段落召回     │
│  → 归因分析 → 报告生成  （SweepingOrchestrator 串行编排 + 进度回调） │
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
│   │   ├── chunk_retriever.py      # 段落召回 Agent（证据级 chunk 检索）
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
| `api/models.py` | Pydantic 模型：`AnalyzeRequest/Response`、`ScanRequest{as_of,realtime,force}`、`ScanResponse{cached,result}`、`ProgressMessage{type,agent_key,progress_percent,fatal}` |
| `backend/agents/*` | 6 个 Agent + 编排器，`AgentBase` 提供 `_report_progress` 细粒度进度回调 |
| `backend/skills/*` | 公告检索(cninfo)、OCR、Embedding(bge/fallback)、案例向量、财务抓取、特征加载、风险词典 |
| `backend/config.py` | 全局配置单一来源，路径/阈值/模型路径集中，业务代码禁止硬编码 |

## 四、前端模块（`api/static/index.html`）

单 HTML 文件（原生 JS，无构建工具），核心机制：

- **全局状态** `AppState`：`currentCompany`/`scanStatus`/`currentTaskId`/`agentStates`/`agentResults`/`resultCache`/`realtimeMode`，带发布订阅 `subscribe/set`。
- **API 调用**：`fetchSingleCompany`(→GET `/api/company/{code}`)、`fetchAnalysis`(→POST `/api/analyze`)、`startRealtimeScan`(→POST `/api/scan` + WS + DELETE 取消)。
- **WebSocket**：`/ws/pipeline/{taskId}`，`onmessage` 按 `type` 分发（progress/complete/error/fallback/cancelled/heartbeat），断线指数退避重连，消息 id 去重。
- **7 Agent 流水线可视化**：左侧 Agent 卡片灰→蓝(脉冲)→绿(完成)，顶部 5 步步骤条，全局进度条。
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

### 一键启动（推荐）

在资源管理器中进入项目根目录，双击：

- `start_api_verify.bat` —— 启动 FastAPI 服务并自动打开浏览器访问 `http://127.0.0.1:8000`
- `start_api.bat` —— 仅启动服务（不自动开浏览器，适合开发/调试）

> 两个脚本均通过环境变量 `API_PORT` / `API_HOST` 控制监听端口（默认 `0.0.0.0:8000`），避免与历史脚本 `8080` 端口冲突。

### 命令行启动

```bash
# 1. 安装依赖
cd D:/competition_agent
pip install -r requirements.txt

# 2. 配置 .env（见上节）

# 3. 启动后端（FastAPI + WebSocket）
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
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
| POST | `/api/scan` | 实时扫雷：`realtime=false` 秒返离线快照(cached)，`true` 创建任务，冲突返回 409；支持 `as_of` 指定截止日期 |
| DELETE | `/api/scan/{task_id}` | 取消实时任务 |
| GET | `/api/scan/{task_id}` | 查询任务状态（断线重连补历史） |
| GET | `/api/mock/{code}` | 联调用：有离线快照优先返回真实，否则 404 |
| WS | `/ws/pipeline/{task_id}` | 实时推送进度（progress/complete/error/fallback/cancelled/heartbeat） |

## 八、核心功能

1. **7 Agent 流水线**：公告研读→财务异常→预测建模→案例匹配→段落召回→归因分析→报告生成，串行编排 + 细粒度进度回调。
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

*本项目为研究生创新大赛参赛作品，东吴证券赞助。*
