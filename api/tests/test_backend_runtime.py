import asyncio

import pytest
from fastapi.testclient import TestClient

from api import main, pipeline
from api.models import ProgressMessage, ScanRequest


@pytest.fixture(autouse=True)
def clean_runtime_state():
    pipeline._task_store.clear()
    pipeline._task_handles.clear()
    pipeline._result_cache.clear()
    yield
    for handle in list(pipeline._task_handles.values()):
        handle.cancel()
    pipeline._task_store.clear()
    pipeline._task_handles.clear()
    pipeline._result_cache.clear()


def test_offline_scan_returns_snapshot_without_task():
    with TestClient(main.app) as client:
        response = client.post("/api/scan", json={"code": "000001"})

    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == "cached"
    assert body["cached"] is True
    assert body["status"] == "completed"
    assert body["result"]["code"] == "000001.SZ"
    assert pipeline.active_tasks() == []


def test_realtime_conflict_and_cancel(monkeypatch):
    started = asyncio.Event()

    async def fake_run(state, req):
        state.status = "running"
        started.set()
        try:
            while not state.cancel_event.is_set():
                await asyncio.sleep(0.01)
        finally:
            state.status = "cancelled"

    monkeypatch.setattr(main, "run_scan_task", fake_run)
    with TestClient(main.app) as client:
        first = client.post("/api/scan", json={"code": "000001.SZ", "realtime": True})
        assert first.status_code == 200
        task_id = first.json()["task_id"]

        second = client.post("/api/scan", json={"code": "000063.SZ", "realtime": True})
        assert second.status_code == 409
        assert second.json()["detail"]["task_id"] == task_id

        cancelled = client.delete(f"/api/scan/{task_id}")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"


def test_websocket_replays_producer_history(monkeypatch):
    async def fake_run(state, req):
        state.status = "running"
        pipeline.emit_message(state, ProgressMessage(
            type="progress", step=1, total=pipeline.AGENT_TOTAL,
            agent="AnnouncementReader", agent_key="announcement",
            status="running", progress_percent=45, message="正在匹配风险词典",
        ))
        result = pipeline.get_offline_result(req.code, req.window)
        state.result = result
        state.status = "completed"
        pipeline.emit_message(state, ProgressMessage(
            type="complete", step=pipeline.AGENT_TOTAL, total=pipeline.AGENT_TOTAL,
            agent="Reporter", agent_key="report", status="done",
            progress_percent=100, message="报告生成完成", result=result,
        ))

    monkeypatch.setattr(main, "run_scan_task", fake_run)
    with TestClient(main.app) as client:
        created = client.post("/api/scan", json={"code": "000001.SZ", "realtime": True})
        task_id = created.json()["task_id"]
        with client.websocket_connect(f"/ws/pipeline/{task_id}") as websocket:
            first = websocket.receive_json()
            second = websocket.receive_json()

    assert first["type"] == "progress"
    assert first["agent_key"] == "announcement"
    assert second["type"] == "complete"
    assert second["result"]["code"] == "000001.SZ"


def test_failure_emits_error_then_offline_fallback(monkeypatch):
    def fail_run(*args, **kwargs):
        raise OSError("WinError 10060")

    monkeypatch.setattr(pipeline.StreamingOrchestrator, "run", fail_run)
    state = pipeline.create_task("000001.SZ")
    request = ScanRequest(code="000001.SZ", realtime=True)

    asyncio.run(pipeline.run_scan_task(state, request))

    assert state.status == "fallback"
    assert state.result is not None
    message_types = [message.type for message in state.progress]
    assert "error" in message_types
    assert "fallback" in message_types
    assert message_types.index("error") < message_types.index("fallback")
    fallback = next(message for message in state.progress if message.type == "fallback")
    assert fallback.result is not None
    assert fallback.agent_key == "orchestrator"


def test_result_cache_is_lru_bounded_to_twenty():
    template = pipeline.offline_to_response("000001.SZ")
    assert template is not None
    for index in range(21):
        code = f"{index:06d}.SZ"
        pipeline.cache_result(code, 60, template.model_copy(update={"code": code}))

    assert len(pipeline._result_cache) == 20
    assert pipeline.get_cached_result("000000.SZ", 60) is None
    assert pipeline.get_cached_result("000020.SZ", 60).code == "000020.SZ"


def test_progress_messages_have_stable_frontend_fields_and_history():
    state = pipeline.create_task("000001.SZ")
    message = ProgressMessage(
        agent="AnnouncementReader",
        agent_key="announcement",
        total=pipeline.AGENT_TOTAL,
        progress_percent=45,
        message="正在匹配风险词典",
    )
    pipeline.emit_message(state, message)

    assert state.progress == [message]
    dumped = message.model_dump()
    assert dumped["id"]
    assert dumped["agent_key"] == "announcement"
    assert dumped["progress_percent"] == 45
    assert dumped["total"] == pipeline.AGENT_TOTAL
