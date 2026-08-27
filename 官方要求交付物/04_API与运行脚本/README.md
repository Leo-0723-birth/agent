# API 与运行脚本

## API

启动：

```powershell
.\.venv\Scripts\python.exe 官方要求交付物\api_service.py
```

接口：

- `GET /health`：健康检查。
- `GET /`：接口说明。
- `POST /predict`：接受 JSON `{"company":"000004.SZ","window":60,"as_of":"2025-12-02"}`，或 `{"companies":["000004.SZ","000063.SZ"],"window":60}`。

返回包含 `request_id`、`results`、`prediction`、`attribution`、`cases`、`report` 和 `trace_log`。请求结束后 trace 会写入 `../02_Agent推理日志/`。

## 批量脚本

```powershell
.\.venv\Scripts\python.exe 官方要求交付物\04_API与运行脚本\batch_predict.py --companies 000004.SZ 000063.SZ --as-of 2025-12-02
```

脚本调用同一个 `SweepingOrchestrator`，输出完整 JSONL 和便于评估的 CSV。

## 已验证的返回内容

API 健康端点已验证返回 `status=ok`。`POST /predict` 成功时，每家公司结果包含 `prediction`、`attribution`、`cases`、`report` 和 `trace_log`；非法窗口返回 400，分析异常返回 500。

API 使用 Python 标准库 HTTP 服务，默认监听 127.0.0.1:8787，适合答辩现场本机演示。批量 CSV 服务于快速浏览和评估，JSONL 保留完整上下文以便审计复核。
