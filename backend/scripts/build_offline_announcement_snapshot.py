#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""构建可提交到仓库的巨潮官方公告与分析结果离线快照。"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from backend.agents.announcement_reader import AnnouncementReaderAgent
from backend.config import ANNOUNCE_OFFLINE_SNAPSHOT_DIR, BASE_DIR
from backend.context import Context
from backend.skills.announcement_context_filter import FILTER_VERSION
from backend.skills.announcement_search import CninfoAnnouncementSource
from backend.skills.competition_history import CompetitionAwareAnnouncementSource
from backend.skills.offline_announcement_snapshot import (
    ANALYSIS_SCHEMA,
    SNAPSHOT_SCHEMA,
)


def _write_gzip_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                json.dump(payload, text, ensure_ascii=False, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable(value):
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable(item) for item in value]
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        root = str(BASE_DIR).replace("\\", "/") + "/"
        if normalized.lower().startswith(root.lower()):
            return normalized[len(root):]
    return value


def build(company_query: str, as_of: str, days: int, output_dir: Path) -> list[Path]:
    online = CninfoAnnouncementSource(
        max_documents=None,
        ocr_enabled=True,
        prefer_offline=False,
    )
    company, announcements = online.search(company_query, days=days, as_of=as_of)
    for item in announcements:
        item["cache_path"] = ""
        item["extraction_cache_path"] = ""
    code_token = company["secucode"].replace(".", "_")
    created_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    cutoff = date.fromisoformat(as_of)
    aliases = sorted(
        {
            company_query,
            company["code"],
            company["secucode"],
            company["secucode"].replace(".", ""),
            company["company_name"],
            *(item.get("company_name", "") for item in announcements),
        }
        - {""}
    )
    raw_path = output_dir / f"{code_token}_announcements_{as_of}.json.gz"
    raw_payload = {
        "schema_version": SNAPSHOT_SCHEMA,
        "snapshot_id": f"{code_token}_{as_of}",
        "created_at": created_at,
        "source_name": "巨潮资讯网",
        "source_endpoints": [company.get("source_url", "")],
        "company": company,
        "aliases": aliases,
        "coverage": {
            "query_start": (cutoff - timedelta(days=days - 1)).isoformat(),
            "query_end": cutoff.isoformat(),
            "lookback_days": days,
            "announcement_date_min": min(item["date"] for item in announcements),
            "announcement_date_max": max(item["date"] for item in announcements),
        },
        "announcement_count": len(announcements),
        "announcements": announcements,
    }
    _write_gzip_json(raw_path, raw_payload)

    source = CompetitionAwareAnnouncementSource(
        CninfoAnnouncementSource(
            max_documents=None,
            ocr_enabled=True,
            offline_snapshot_dir=output_dir,
            prefer_offline=True,
        )
    )
    agent = AnnouncementReaderAgent(source=source, use_finbert=False, use_llm=False)
    result, trace = agent.run(company_query, Context(company=company_query, as_of=as_of))
    result_payload = result.to_dict()
    result_payload["announcement_filter_version"] = FILTER_VERSION
    result_payload["run_trace"] = trace
    result_payload = _portable(result_payload)
    analysis_path = output_dir / f"{code_token}_analysis_{as_of}.json.gz"
    analysis_payload = {
        "schema_version": ANALYSIS_SCHEMA,
        "snapshot_id": f"{code_token}_{as_of}_rule_analysis",
        "created_at": created_at,
        "company": company,
        "aliases": aliases,
        "analysis_options": {
            "as_of": as_of,
            "use_ocr": True,
            "use_finbert": False,
            "use_llm": False,
            "filter_version": FILTER_VERSION,
        },
        "result": result_payload,
    }
    _write_gzip_json(analysis_path, analysis_payload)

    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema_version": "offline_announcement_manifest_v1",
        "created_at": created_at,
        "company": company["secucode"],
        "source": "巨潮资讯网官方公告及PDF正文缓存",
        "snapshot_as_of": as_of,
        "lookback_days": days,
        "files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (raw_path, analysis_path)
        ],
        "warning": "离线快照只代表锚点日期；锚点之后的公告必须联网刷新。",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return [raw_path, analysis_path, manifest_path]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--company", default="000004SZ")
    parser.add_argument("--as-of", default=date.today().isoformat())
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--output-dir", type=Path, default=ANNOUNCE_OFFLINE_SNAPSHOT_DIR)
    args = parser.parse_args()
    for path in build(args.company, args.as_of, args.days, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
