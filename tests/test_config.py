"""Tests for configuration loading and validation."""

import copy
import json
from pathlib import Path

import pytest

from biosignalsplux_labview_sync.config import ConfigError, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = PROJECT_ROOT / "configs" / "acquisition.example.json"


def test_example_config_is_valid() -> None:
    config = load_config(EXAMPLE_CONFIG)

    assert config["biosignalsplux"]["sampling_rate_hz"] == 1000
    assert config["biosignalsplux"]["resolution_bits"] == 16
    assert config["biosignalsplux"]["channel_mask"] == 3
    assert config["biosignalsplux"]["channels"] == ["emg_0", "emg_1"]
    assert config["labview"]["command_channel"]["port"] == 5005


def test_channel_mask_must_match_channel_names(tmp_path: Path) -> None:
    config = load_config(EXAMPLE_CONFIG)
    invalid_config = copy.deepcopy(config)

    invalid_config["biosignalsplux"]["channel_mask"] = 1
    invalid_path = tmp_path / "invalid_config.json"
    invalid_path.write_text(
        json.dumps(invalid_config),
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigError,
        match="number of channel names",
    ):
        load_config(invalid_path)
