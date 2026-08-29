"""pipeline 核心调度逻辑的单元测试（第二批 #28）。

覆盖：终态任务清理、manifest 缓存、取消边界、离线同步接口、结果缓存命中。
"""
import asyncio
import inspect
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


def _make_terminal_task(code="300577.SZ", finished_offset=0.0):
    state = pipeline.create_task(code)
    state.status = "completed"
    state.finished_at = time.time() + finished_offset
    return state


def test_prune_removes_expired_terminal_tasks():
    # 过期任务（finished_at 距今很久）应被清理
    expired = _make_terminal_task("300577.SZ", finished_offset=-10000)
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
    running = pipeline.create_task("300577.SZ")
    running.status = "running"
    pending = pipeline.create_task("000063.SZ")
    pending.status = "pending"

    pipeline._prune_finished_tasks()

    assert running.task_id in pipeline._task_store
    assert pending.task_id in pipeline._task_store


def test_cancel_terminal_task_is_noop():
    state = _make_terminal_task("300577.SZ")
    result = asyncio.run(pipeline.cancel_task(state.task_id))

    assert result is state
    assert state.status == "completed"


def test_run_pipeline_is_sync_function():
    assert not inspect.iscoroutinefunction(pipeline.run_pipeline)
    result = pipeline.run_pipeline(["300577.SZ"], 60, False, True)
    assert isinstance(result, list)
    assert result and result[0].code == "300577.SZ"


def test_manifest_cache_reuses_loaded_payload():
    first = pipeline._load_manifest()
    second = pipeline._load_manifest()
    assert first is second
    assert isinstance(first, list)


def test_get_offline_result_populates_lru_cache():
    result = pipeline.get_offline_result("300577.SZ", 60)
    assert result is not None
    assert pipeline.get_cached_result("300577.SZ", 60) is not None


def test_get_offline_result_unknown_code_returns_none():
    assert pipeline.get_offline_result("999999.SZ", 60) is None


def test_run_agent_ok():
    streamer = pipeline.StreamingOrchestrator(callback=lambda m: None)
    outcome, error = streamer._run_agent(lambda c, x: None, "300577.SZ", None, 5)
    assert outcome == "ok"
    assert error is None


def test_run_agent_error_propagates():
    def boom(c, x):
        raise ValueError("boom")

    streamer = pipeline.StreamingOrchestrator(callback=lambda m: None)
    outcome, error = streamer._run_agent(boom, "300577.SZ", None, 5)
    assert outcome == "error"
    assert isinstance(error, ValueError)


def test_run_agent_timeout_returns_timeout():
    def slow(c, x):
        time.sleep(1.0)

    streamer = pipeline.StreamingOrchestrator(callback=lambda m: None)
    outcome, error = streamer._run_agent(slow, "300577.SZ", None, 0.05)
    assert outcome == "timeout"
    assert error is None
