from pathlib import Path

from streamlit.testing.v1 import AppTest

from backend.dashboard_utils import (
    risk_interval_comparison_rows,
    risk_monthly_severity_rows,
    risk_theme_distribution_rows,
    risk_theme_heatmap_rows,
    risk_theme_name,
    risk_window_comparison_rows,
)


def test_streamlit_app_initial_view_has_query_form():
    app_path = Path(__file__).resolve().parents[2] / "公告研读agent.py"
    app = AppTest.from_file(str(app_path), default_timeout=60).run()

    assert not app.exception
    assert app.title[0].value == "公告研读 Agent"
    assert app.text_input[0].value == "000001"
    assert app.button[0].label == "开始研读"


def test_risk_theme_distribution_uses_selected_window_and_readable_names():
    counts = {
        "30d": {"A03": 1},
        "60d": {"A03": 2},
        "90d": {"C-CANDIDATE": 3, "A03": 1},
    }

    rows = risk_theme_distribution_rows(counts, 90)

    assert [row["主题代码"] for row in rows] == ["C-CANDIDATE", "A03"]
    assert rows[0]["风险主题"] == "资产质量与减值（待精分类）"
    assert rows[0]["事件数"] == 3
    assert rows[0]["占比"] == 0.75
    assert risk_theme_name("A03") == "利润、扣非利润与业绩波动"


def test_risk_theme_distribution_ignores_invalid_or_empty_counts():
    counts = {"90d": {"A03": "2", "G07": 0, "BAD": {"nested": 1}}}

    rows = risk_theme_distribution_rows(counts, 90)

    assert len(rows) == 1
    assert rows[0]["主题代码"] == "A03"
    assert rows[0]["事件数"] == 2


def test_streamlit_result_view_renders_selected_window_chart():
    app_path = Path(__file__).resolve().parents[2] / "公告研读agent.py"
    app = AppTest.from_file(str(app_path), default_timeout=15)
    app.session_state["announcement_analysis"] = {
        "name": "测试公司",
        "company": "000001",
        "as_of": "2026-08-21",
        "semantic": {
            "stats": {},
            "data_quality": {
                "lookback_days": 365,
                "source": "巨潮资讯网",
                "title_excluded_count": 1,
            },
            "channel_summary": {
                "rule": {"suppressed_count": 1},
                "llm": {"rejected_nonfactual_context": 1},
            },
            "risk_factors": [
                {
                    "event_key": "event-1",
                    "risk_id": "risk-1",
                    "announcement_date": "2026-08-20",
                    "severity": 5,
                    "taxonomy_l2": "A03",
                    "evidence_valid": True,
                },
                {
                    "event_key": "event-2",
                    "risk_id": "risk-2",
                    "announcement_date": "2026-07-10",
                    "severity": 3,
                    "taxonomy_l2": "C-CANDIDATE",
                    "evidence_valid": True,
                },
            ],
            "announcements": [
                {
                    "id": "policy",
                    "date": "2026-08-20",
                    "title": "独立董事候选人声明",
                    "analysis_status": "excluded_by_title",
                    "analysis_skip_reason": "candidate_declaration",
                    "source_url": "https://www.cninfo.com.cn/policy",
                    "pdf_url": "https://static.cninfo.com.cn/policy.pdf",
                }
            ],
            "per_announcement": {
                "policy": {
                    "suppressed_rule_hits": [
                        {
                            "label": "G07",
                            "matched_keyword": "立案调查",
                            "suppression_reason": "governance_eligibility_clause",
                            "evidence": "被中国证监会立案调查的不得担任董事",
                        }
                    ]
                }
            },
            "f1_features": {
                "category_event_counts": {
                    "30d": {"A03": 1},
                    "60d": {"A03": 2},
                    "90d": {"C-CANDIDATE": 3, "A03": 1},
                },
                "scalar_features": {
                    "announcement_count_30d": 5,
                    "announcement_count_60d": 8,
                    "announcement_count_90d": 10,
                    "risk_event_count_30d": 1,
                    "risk_event_count_60d": 2,
                    "risk_event_count_90d": 4,
                    "high_risk_event_count_30d": 0,
                    "high_risk_event_count_60d": 1,
                    "high_risk_event_count_90d": 2,
                },
            },
        },
    }

    app.run()

    assert not app.exception
    assert app.segmented_control[0].value == "月份热力图"
    assert any(item.value == "近一年风险事件时间轴" for item in app.subheader)
    assert any(item.value == "最近 90 天风险节奏" for item in app.subheader)
    assert any(item.value == "风险主题分析" for item in app.subheader)
    assert len(app.get("vega_lite_chart")) == 3
    assert any(item.label == "标题过滤公告" for item in app.metric)
    assert any(item.label == "规则语境过滤" for item in app.metric)


def test_risk_window_comparison_keeps_all_three_windows_and_real_denominator():
    rows = risk_window_comparison_rows(
        {
            "announcement_count_30d": 4,
            "risk_event_count_30d": 2,
            "high_risk_event_count_30d": 1,
            "announcement_count_60d": 8,
            "risk_event_count_60d": 3,
            "high_risk_event_count_60d": 1,
            "announcement_count_90d": 10,
            "risk_event_count_90d": 5,
            "high_risk_event_count_90d": 2,
        }
    )

    assert [row["时间窗口"] for row in rows] == [
        "最近 30 天",
        "最近 60 天",
        "最近 90 天",
    ]
    assert rows[0]["公告总数"] == 4
    assert rows[0]["风险事件"] == 2
    assert rows[0]["每份公告风险事件"] == 0.5


def test_non_overlapping_intervals_do_not_double_count_windows():
    announcements = [
        {"date": "2026-08-20"},
        {"date": "2026-07-20"},
        {"date": "2026-06-20"},
    ]
    factors = [
        {"event_key": "a", "announcement_date": "2026-08-20", "severity": 5},
        {"event_key": "b", "announcement_date": "2026-07-20", "severity": 3},
        {"event_key": "c", "announcement_date": "2026-06-20", "severity": 2},
    ]

    rows = risk_interval_comparison_rows(announcements, factors, "2026-08-23")

    assert [row["风险事件"] for row in rows] == [1, 1, 1]
    assert [row["高风险事件"] for row in rows] == [1, 0, 0]
    assert sum(row["风险事件"] for row in rows) == 3


def test_monthly_timeline_has_twelve_months_and_deduplicates_events():
    factors = [
        {"event_key": "same", "announcement_date": "2026-08-20", "severity": 5},
        {"event_key": "same", "announcement_date": "2026-08-20", "severity": 5},
        {"event_key": "other", "announcement_date": "2026-07-10", "severity": 3},
    ]

    rows = risk_monthly_severity_rows(factors, "2026-08-23")

    assert len(rows) == 12
    assert rows[-1]["月份"] == "2026-08"
    assert rows[-1]["高风险"] == 1
    assert rows[-2]["中风险"] == 1


def test_theme_heatmap_fills_zero_months_for_top_themes():
    factors = [
        {"event_key": "a", "announcement_date": "2026-08-20", "taxonomy_l2": "A03"},
        {"event_key": "b", "announcement_date": "2026-07-10", "taxonomy_l2": "G07"},
    ]

    rows = risk_theme_heatmap_rows(factors, "2026-08-23", max_themes=8)

    assert len(rows) == 24
    assert {row["主题代码"] for row in rows} == {"A03", "G07"}
    assert sum(row["事件数"] for row in rows) == 2
