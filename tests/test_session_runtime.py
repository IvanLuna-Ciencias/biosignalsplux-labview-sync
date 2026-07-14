"""Tests for synchronized acquisition runtime preparation."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from biosignalsplux_labview_sync.session_config import (
    SessionFormData,
    build_session_metadata,
)
from biosignalsplux_labview_sync.session_runtime import (
    SessionRuntimeError,
    build_acquisition_summary,
    prepare_acquisition_runtime,
)


class FakeAcquisition:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        class Results:
            termination_reason = "stop_requested"
            samples_received = 2500
            writer_stats = None
            runtime_error = None
            request_stop_error = None
            stop_error = None
            close_error = None

        self.results = Results()


def make_metadata():
    form = SessionFormData(
        participant_id="P-ABC123",
        name_or_alias="Participant 01",
        age_years=25,
        sex_assigned_at_birth="prefer_not_to_say",
        instrumented_arm="right",
        notes="Runtime preparation test.",
        device_address="B00:07:80:4D:2F:0B",
        joint="elbow",
        start_deg=0.0,
        end_deg=20.0,
        frequency_hz=0.1,
        cycles=2,
        setpoint_rate_hz=50.0,
        move_to_start_s=3.0,
        return_duration_s=3.0,
        neutral_hold_s=0.5,
        labview_host="172.22.11.2",
        labview_port=5005,
        labview_enabled=False,
    )

    return build_session_metadata(
        form,
        timestamp=datetime(
            2026,
            7,
            13,
            18,
            30,
            tzinfo=timezone.utc,
        ),
    )


def test_runtime_prepares_paths_metadata_and_controller(
    tmp_path,
) -> None:
    metadata = make_metadata()
    loaded_paths = []

    def fake_plux_loader(api_path):
        loaded_paths.append(api_path)
        return object()

    runtime = prepare_acquisition_runtime(
        metadata=metadata,
        output_root=tmp_path,
        api_path=tmp_path / "plux_api",
        flush_interval_samples=1000,
        live_window_seconds=5.0,
        session_origin=123.456,
        plux_loader=fake_plux_loader,
        acquisition_factory=FakeAcquisition,
    )

    assert runtime.metadata is metadata
    assert runtime.session_origin == pytest.approx(123.456)
    assert len(loaded_paths) == 1

    assert runtime.paths.directory.exists()
    assert runtime.paths.metadata_json.exists()

    saved = json.loads(
        runtime.paths.metadata_json.read_text(
            encoding="utf-8"
        )
    )

    assert saved["session_id"] == metadata.session_id
    assert saved["biosignalsplux"]["sampling_rate_hz"] == 1000
    assert saved["trajectory"]["joint"] == "elbow"

    controller_kwargs = runtime.acquisition.kwargs

    assert controller_kwargs["sampling_rate_hz"] == 1000
    assert controller_kwargs["resolution_bits"] == 16
    assert controller_kwargs["channel_mask"] == 3
    assert controller_kwargs["channel_names"] == (
        "emg_0",
        "emg_1",
    )
    assert controller_kwargs["session_origin"] == pytest.approx(
        123.456
    )
    assert (
        controller_kwargs["csv_path"]
        == runtime.paths.emg_csv
    )


def test_runtime_live_buffer_retains_five_seconds(
    tmp_path,
) -> None:
    runtime = prepare_acquisition_runtime(
        metadata=make_metadata(),
        output_root=tmp_path,
        api_path=tmp_path / "plux_api",
        live_window_seconds=5.0,
        session_origin=1.0,
        plux_loader=lambda path: object(),
        acquisition_factory=FakeAcquisition,
    )

    for index in range(5100):
        runtime.live_buffer.append_sample(
            index / 1000.0,
            (index, index + 1),
        )

    snapshot = runtime.live_buffer.snapshot()

    assert snapshot.total_samples_received == 5100
    assert len(snapshot.time_monotonic_s) == 5000
    assert len(snapshot.channels[0]) == 5000
    assert len(snapshot.channels[1]) == 5000


def test_invalid_live_window_duration_is_rejected(
    tmp_path,
) -> None:
    with pytest.raises(
        SessionRuntimeError,
        match="live_window_seconds",
    ):
        prepare_acquisition_runtime(
            metadata=make_metadata(),
            output_root=tmp_path,
            api_path=tmp_path / "plux_api",
            live_window_seconds=0.0,
            plux_loader=lambda path: object(),
            acquisition_factory=FakeAcquisition,
        )


def test_acquisition_summary_contains_output_paths(
    tmp_path,
) -> None:
    runtime = prepare_acquisition_runtime(
        metadata=make_metadata(),
        output_root=tmp_path,
        api_path=tmp_path / "plux_api",
        session_origin=1.0,
        plux_loader=lambda path: object(),
        acquisition_factory=FakeAcquisition,
    )

    summary = build_acquisition_summary(runtime)

    assert summary["termination_reason"] == "stop_requested"
    assert summary["samples_received"] == 2500
    assert summary["samples_written"] == 0
    assert summary["emg_csv"] == str(runtime.paths.emg_csv)
    assert summary["metadata_json"] == str(
        runtime.paths.metadata_json
    )