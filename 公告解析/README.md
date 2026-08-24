# 公告解析可移植运行包

本目录只包含公告研读 Agent 实际运行所需的两份 2020—2024 年历史研究产物，
用于解决其他成员电脑上没有 `D:\codex work\Announcement_NLP_Project_Final` 的问题。

## 内置文件

- `02_风险标签抽取模块/规则风险结果/announcement_rule_risks.jsonl.gz`
  - 原始 `announcement_rule_risks.jsonl` 的无损 gzip 压缩版；程序逐行读取，不会
    解压出临时大文件。
- `04_语义特征生成模块/核心Parquet/semantic_features.parquet`
  - BAAI/bge-large-zh-v1.5 + train-only incremental PCA 的历史语义特征。

文件来源、字节数和 SHA-256 记录在 `manifest.json`，便于核验数据未被替换。

## 默认路径

`backend/config.py` 默认从本目录读取，因此克隆仓库并安装依赖后即可运行：

```powershell
streamlit run 公告研读agent.py --server.port 8502
```

不需要再创建 `COMPETITION_DATA_ROOT`。如本机持有完整交付目录，可在 `.env` 中
设置该变量覆盖默认路径；代码同时兼容未压缩 `.jsonl` 与 `.jsonl.gz`。

## 范围说明

完整交付目录约 20.77 GiB，包含公告全文、问询全文、Chroma 向量数据库、嵌入
索引和流水线日志，超出 GitHub 普通仓库限制，也不是公告研读页面运行的必要项，
因此未纳入 Git。历史数据只用于候选检索和历史语义特征，不替代巨潮当前公告，
也不计入当前 30/60/90 天风险事件。
