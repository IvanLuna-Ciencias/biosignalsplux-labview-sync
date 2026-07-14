#!/usr/bin/env python
"""Launch the synchronized-session configuration GUI."""

from biosignalsplux_labview_sync.session_config_window import (
    run_configuration_window,
)


if __name__ == "__main__":
    raise SystemExit(run_configuration_window())