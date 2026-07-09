"""Configuration loading and validation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a configuration file is missing or invalid."""


def _require_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)

    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid configuration section: {name}")

    return value


def _validate_port(value: Any, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{field_name} must be an integer.")

    if not 1 <= value <= 65535:
        raise ConfigError(f"{field_name} must be between 1 and 65535.")


def validate_config(config: dict[str, Any]) -> None:
    """Validate the minimum project configuration structure."""

    session = _require_section(config, "session")
    biosignalsplux = _require_section(config, "biosignalsplux")
    labview = _require_section(config, "labview")
    _require_section(config, "logging")

    output_root = session.get("output_root")
    if not isinstance(output_root, str) or not output_root.strip():
        raise ConfigError("session.output_root must be a non-empty string.")

    prestart_seconds = session.get("prestart_seconds")
    if not isinstance(prestart_seconds, (int, float)) or isinstance(
        prestart_seconds, bool
    ):
        raise ConfigError("session.prestart_seconds must be numeric.")

    if prestart_seconds < 0:
        raise ConfigError("session.prestart_seconds cannot be negative.")

    sampling_rate = biosignalsplux.get("sampling_rate_hz")
    if not isinstance(sampling_rate, int) or isinstance(sampling_rate, bool):
        raise ConfigError(
            "biosignalsplux.sampling_rate_hz must be an integer."
        )

    if sampling_rate <= 0:
        raise ConfigError(
            "biosignalsplux.sampling_rate_hz must be greater than zero."
        )

    resolution_bits = biosignalsplux.get("resolution_bits")
    if not isinstance(resolution_bits, int) or isinstance(
        resolution_bits, bool
    ):
        raise ConfigError(
            "biosignalsplux.resolution_bits must be an integer."
        )

    if resolution_bits <= 0:
        raise ConfigError(
            "biosignalsplux.resolution_bits must be greater than zero."
        )

    channel_mask = biosignalsplux.get("channel_mask")
    if not isinstance(channel_mask, int) or isinstance(channel_mask, bool):
        raise ConfigError("biosignalsplux.channel_mask must be an integer.")

    if channel_mask <= 0:
        raise ConfigError(
            "biosignalsplux.channel_mask must be greater than zero."
        )

    channels = biosignalsplux.get("channels")
    if not isinstance(channels, list) or not channels:
        raise ConfigError(
            "biosignalsplux.channels must be a non-empty list."
        )

    if not all(isinstance(channel, str) and channel for channel in channels):
        raise ConfigError(
            "Every item in biosignalsplux.channels must be a non-empty string."
        )

    enabled_channel_count = bin(channel_mask).count("1")
    if enabled_channel_count != len(channels):
        raise ConfigError(
            "The number of channel names must match the enabled bits "
            "in biosignalsplux.channel_mask."
        )

    host = labview.get("host")
    if not isinstance(host, str) or not host.strip():
        raise ConfigError("labview.host must be a non-empty string.")

    command_channel = labview.get("command_channel")
    if not isinstance(command_channel, dict):
        raise ConfigError("labview.command_channel must be an object.")

    transport = command_channel.get("transport")
    if transport not in {"udp", "tcp"}:
        raise ConfigError(
            "labview.command_channel.transport must be 'udp' or 'tcp'."
        )

    _validate_port(
        command_channel.get("port"),
        "labview.command_channel.port",
    )

    for section_name in ("status_channel", "trajectory_stream"):
        section = labview.get(section_name)

        if not isinstance(section, dict):
            raise ConfigError(f"labview.{section_name} must be an object.")

        if not isinstance(section.get("enabled"), bool):
            raise ConfigError(
                f"labview.{section_name}.enabled must be true or false."
            )

        section_transport = section.get("transport")
        if section_transport not in {"udp", "tcp"}:
            raise ConfigError(
                f"labview.{section_name}.transport must be 'udp' or 'tcp'."
            )

        _validate_port(
            section.get("port"),
            f"labview.{section_name}.port",
        )


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a JSON configuration file."""

    config_path = Path(path)

    if not config_path.is_file():
        raise ConfigError(
            f"Configuration file not found: {config_path}"
        )

    try:
        text = config_path.read_text(encoding="utf-8-sig")
        config = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"Invalid JSON in {config_path}: {exc}"
        ) from exc
    except OSError as exc:
        raise ConfigError(
            f"Could not read configuration file {config_path}: {exc}"
        ) from exc

    if not isinstance(config, dict):
        raise ConfigError(
            "The root of the configuration file must be a JSON object."
        )

    validate_config(config)
    return config
