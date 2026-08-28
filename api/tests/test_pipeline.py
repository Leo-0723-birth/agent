"""pipeline 核心调度逻辑的单元测试（第二批 #28）。

覆盖：终态任务清理、manifest 缓存、取消边界、离线同步接口、结果缓存命中。
"""
import asyncio
import inspect
import json
import time

import pytest

from api import pipeline
from api.models import ScanRequest


@pytest.fixture(autouse=True)
def clean_runtime_state():
    """每个测试前清理全局状态，测试后回收挂起的任务句柄。"""
    pipeline._task_store.clear()
    pipeline._task_handles.clear()
    pipeline._result_cache.clear()
    pipeline._manifest_cache = None
    pipeline._manifest_cache_mtime = None
    yield
    for handle in list(pipeline._task_handles.values()):
        handle.cancel()
    pipeline._task_store.clear()
    pipeline._task_handles.clear()
    pipeline._result_cache.clear()
    pipeline._manifest_cache = None
    pipeline._manifest_cache_mtime = None


def _make_terminal_task(code="000001.SZ", finished_offset=0.0):
    state = pipeline.create_task(code)
    state.status = "completed"
    state.finished_at = time.time() + finished_offset
    return state


def test_prune_removes_expired_terminal_tasks():
    # 过期任务（finished_at 距今很久）应被清理
    expired = _make_terminal_task("000001.SZ", finished_offset=-10000)
    active = pipeline.create_task("000063.SZ")
    active.status = "running"

    pipeline._prune_finished_tasks()

    assert expired.task_id not in pipeline._task_store
    assert active.task_id in pipeline._task_store


def test_prune_keeps_recent_terminal_tasks_up_to_limit():
    for i in range(pipeline.MAX_TERMINAL_TASKS + 5):
        _make_terminal_task(f"{i:06d}.SZ", finished_offset=0)

    pipeline._prune_finished_tasks()

    terminal = [s for s in pipeline._task_store.values()
                if s.status in pipeline.TERMINAL_STATUSES]
    assert len(terminal) == pipeline.MAX_TERMINAL_TASKS


def test_prune_ignores_running_and_pending():
    running = pipeline.create_task("000001.SZ")
    running.status = "running"
    pending = pipeline.create_task("000063.SZ")
    pending.status = "pending"

    pipeline._prune_finished_tasks()

    assert running.task_id in pipeline._task_store
    assert pending.task_id in pipeline._task_store


def test_cancel_terminal_task_is_noop():
    state = _make_terminal_task("000001.SZ")
    result = asyncio.run(pipeline.cancel_task(state.task_id))

    assert result is state
    assert state.status == "completed"


def test_run_pipeline_is_sync_function():
    assert not inspect.iscoroutinefunction(pipeline.run_pipeline)
    result = pipeline.run_pipeline(["000001.SZ"], 60, False, True)
    assert isinstance(result, list)
    assert result and result[0].code == "000001.SZ"


def test_manifest_cache_reuses_loaded_payload():
    first = pipeline._load_manifest()
    second = pipeline._load_manifest()
    assert first is second
    assert isinstance(first, list)


def test_get_offline_result_populates_lru_cache():
    result = pipeline.get_offline_result("000001.SZ", 60)
    assert result is not None
    assert pipeline.get_cached_result("000001.SZ", 60) is not None


def test_get_offline_result_unknown_code_returns_none():
    assert pipeline.get_offline_result("999999.SZ", 60) is None


def test_run_agent_ok():
    streamer = pipeline.StreamingOrchestrator(callback=lambda m: None)
    outcome, error = streamer._run_agent(lambda c, x: None, "000001.SZ", None, 5)
    assert outcome == "ok"
    assert error is None


def test_run_agent_error_propagates():
    def boom(c, x):
        raise ValueError("boom")

    streamer = pipeline.StreamingOrchestrator(callback=lambda m: None)
    outcome, error = streamer._run_agent(boom, "000001.SZ", None, 5)
    assert outcome == "error"
    assert isinstance(error, ValueError)


def test_run_agent_timeout_returns_timeout():
    def slow(c, x):
        time.sleep(1.0)

    streamer = pipeline.StreamingOrchestrator(callback=lambda m: None)
    outcome, error = streamer._run_agent(slow, "000001.SZ", None, 0.05)
    assert outcome == "timeout"
    assert error is None


def test_cancel_event_interrupts_long_loop():
    """Agent 内部长循环应响应 Context.cancel_event，取消后尽快退出。"""
    import threading

    from backend.context import Context

    class SlowAgent:
        def execute(self, company, ctx):
            for _ in range(1000):
                if ctx.cancel_event and ctx.cancel_event.is_set():
                    raise RuntimeError("已取消")
                time.sleep(0.001)

    cancel_event = threading.Event()
    ctx = Context(company="000001.SZ")
    ctx.cancel_event = cancel_event

    def run_slow():
        with pytest.raises(RuntimeError, match="已取消"):
            SlowAgent().execute("000001.SZ", ctx)

    t = threading.Thread(target=run_slow)
    t.start()
    time.sleep(0.02)
    cancel_event.set()
    t.join(timeout=1.0)
    assert not t.is_alive()


def test_orchestrator_pool_lru_evicts_oldest():
    """Orchestrator 缓存超过容量时应按 LRU 淘汰最早未命中的实例。"""
    pipeline.clear_orchestrator_pool()
    # 创建 ORCHESTRATOR_POOL_SIZE + 2 个不同 key 的实例
    size = pipeline.ORCHESTRATOR_POOL_SIZE
    keys = []
    for i in range(size + 2):
        orch = pipeline.get_orchestrator(use_llm=i % 2 == 0, use_finbert=True, max_documents=i + 1)
        keys.append(orch)

    assert len(pipeline._orchestrator_pool) == size
    # 最早的两个 key 应被淘汰
    assert keys[0] not in pipeline._orchestrator_pool.values()
    assert keys[1] not in pipeline._orchestrator_pool.values()
    # 最近使用的 key 应保留
    assert keys[-1] in pipeline._orchestrator_pool.values()
    # 命中队尾后再新增，不应淘汰该 key
    pipeline.get_orchestrator(use_llm=keys[2] in pipeline._orchestrator_pool.values(), use_finbert=True, max_documents=9999)


def test_report_cache_reuses_and_invalidates_on_mtime_change(tmp_path, monkeypatch):
    """报告 JSON 缓存按 mtime 失效：文件未变时复用，修改后重新读取。"""
    # 构造临时 manifest 和报告目录结构
    company_dir = tmp_path / "000001.SZ"
    company_dir.mkdir()
    report_file = company_dir / "report.json"
    report_file.write_text('{"code": "000001.SZ", "risk": 0.5}', encoding="utf-8")

    manifest = [{"company": "000001.SZ", "json_file": "000001.SZ/report.json", "generated_at": "2026-08-28T00:00:00"}]
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(pipeline, "REPORTS_DIR", tmp_path)
    pipeline._manifest_cache = None
    pipeline._manifest_cache_mtime = None
    pipeline._report_cache.clear()

    first = pipeline._load_report("000001.SZ")
    assert first == {"code": "000001.SZ", "risk": 0.5}

    # 直接调用应命中缓存，对象引用相同
    second = pipeline._load_report("000001.SZ")
    assert second is first

    # 修改文件 mtime 和内容，缓存应失效并重新读取
    time.sleep(0.05)
    report_file.write_text('{"code": "000001.SZ", "risk": 0.9}', encoding="utf-8")
    third = pipeline._load_report("000001.SZ")
    assert third["risk"] == 0.9
    assert third is not first


def test_as_of_used_in_cache_key():
    """不同 as_of 应产生不同的缓存 key，相同 as_of 命中缓存。"""
    template = pipeline.offline_to_response("000001.SZ")
    assert template is not None
    pipeline.cache_result("000001.SZ", 60, template.model_copy(update={"risk": 0.5}), as_of="2026-08-27")
    pipeline.cache_result("000001.SZ", 60, template.model_copy(update={"risk": 0.9}), as_of="2026-08-28")

    assert pipeline.get_cached_result("000001.SZ", 60, as_of="2026-08-27").risk == 0.5
    assert pipeline.get_cached_result("000001.SZ", 60, as_of="2026-08-28").risk == 0.9
