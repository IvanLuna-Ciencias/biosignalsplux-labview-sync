"""Tests for optional force-sensor configuration."""

from biosignalsplux_labview_sync.force_runtime import force_settings_from_config


def test_missing_force_section_disables_sensor() -> None:
    assert force_settings_from_config({}) is None


def test_force_settings_build_default_channels() -> None:
    settings = force_settings_from_config(
        {
            "force_sensor": {
                "enabled": True,
                "calibration_file": "C:/ATI/FT11495.cal",
                "device": "Dev2",
                "sampling_rate_hz": 1000,
                "bias_seconds": 1.0,
                "flush_interval_samples": 1000,
                "live_window_seconds": 5.0,
            }
        }
    )
    assert settings is not None
    assert settings.channels[0] == "Dev2/ai0"
    assert settings.channels[-1] == "Dev2/ai5"
    assert settings.bias_samples == 1000
