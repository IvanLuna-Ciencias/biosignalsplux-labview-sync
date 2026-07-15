"""Tests for integrated synchronized-session GUI helpers."""

import json

from biosignalsplux_labview_sync.synchronized_acquisition_window import (
    calculate_session_window_layout,
    format_acquisition_summary,
    format_force_summary,
    runtime_settings_from_config,
    write_sync_report,
)


def test_runtime_settings_include_prestart() -> None:
    settings = runtime_settings_from_config(
        {
            "session": {
                "flush_interval_samples": 1000,
                "prestart_seconds": 3.0,
            },
            "biosignalsplux": {
                "api_path": "C:/local/plux",
            },
        }
    )

    assert settings.api_path.as_posix() == "C:/local/plux"
    assert settings.flush_interval_samples == 1000
    assert settings.prestart_seconds == 3.0
    assert settings.live_window_seconds == 5.0


def test_runtime_settings_reject_negative_prestart() -> None:
    try:
        runtime_settings_from_config(
            {
                "session": {
                    "flush_interval_samples": 1000,
                    "prestart_seconds": -1.0,
                },
                "biosignalsplux": {
                    "api_path": "C:/local/plux",
                },
            }
        )
    except ValueError as exc:
        assert "prestart_seconds" in str(exc)
    else:
        raise AssertionError(
            "Negative prestart duration was accepted."
        )


def test_acquisition_summary_contains_core_results() -> None:
    text = format_acquisition_summary(
        {
            "session_id": "P-ABC123_elbow_20260713_120000",
            "termination_reason": "stop_requested",
            "samples_received": 30525,
            "samples_written": 30525,
            "sequence_gap_events": 0,
            "missing_sequence_count": 0,
            "non_increasing_sequence_count": 0,
            "runtime_error": None,
            "request_stop_error": None,
            "stop_error": None,
            "close_error": None,
            "emg_csv": "outputs/session/emg.csv",
            "metadata_json": "outputs/session/metadata.json",
        }
    )

    assert "Muestras recibidas: 30525" in text
    assert "Muestras escritas: 30525" in text
    assert "Eventos de gap: 0" in text
    assert "stop_requested" in text


def test_sync_report_is_written(tmp_path) -> None:
    path = write_sync_report(
        path=tmp_path / "sync_report.json",
        session_id="P-ABC123_elbow_20260713_120000",
        session_duration_s=35.5,
        acquisition_summary={
            "samples_received": 30000,
            "runtime_error": None,
        },
        trajectory_summary={
            "termination_reason": "trajectory_completed",
            "setpoints_generated": 2500,
            "runtime_error": None,
        },
    )

    saved = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert saved["session_duration_s"] == 35.5
    assert saved["acquisition"]["samples_received"] == 30000
    assert (
        saved["trajectory"]["termination_reason"]
        == "trajectory_completed"
    )

def test_force_summary_contains_core_results() -> None:
    text = format_force_summary(
        {
            "termination_reason": "stop_requested",
            "samples_received": 12000,
            "samples_written": 12000,
            "bias_samples_used": 1000,
            "force_csv": "outputs/session/force.csv",
        }
    )

    assert "Sensor ATI:" in text
    assert "Muestras recibidas: 12000" in text
    assert "Muestras de bias: 1000" in text


def test_sync_report_includes_force_summary(tmp_path) -> None:
    path = write_sync_report(
        path=tmp_path / "sync_report.json",
        session_id="P-ABC123_elbow_20260713_120000",
        session_duration_s=35.5,
        acquisition_summary={
            "samples_received": 30000,
            "runtime_error": None,
        },
        trajectory_summary=None,
        force_summary={
            "samples_received": 29500,
            "samples_written": 29500,
            "runtime_error": None,
        },
    )

    saved = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert (
        saved["force_sensor"]["samples_received"]
        == 29500
    )


def test_multimodal_window_layout_has_no_overlap() -> None:
    layout = calculate_session_window_layout(
        screen_x=0,
        screen_y=0,
        screen_width=1920,
        screen_height=1040,
        force_enabled=True,
    )

    control = layout["control"]
    emg = layout["emg"]
    force = layout["force"]

    assert control.x + control.width < emg.x
    assert emg.x == force.x
    assert emg.width == force.width
    assert emg.y + emg.height < force.y
    assert force.y + force.height == 1040


def test_window_layout_uses_full_right_side_without_ati() -> None:
    layout = calculate_session_window_layout(
        screen_x=100,
        screen_y=50,
        screen_width=1366,
        screen_height=768,
        force_enabled=False,
    )

    assert set(layout) == {"control", "emg"}
    assert layout["emg"].height == 768
    assert layout["emg"].y == 50
