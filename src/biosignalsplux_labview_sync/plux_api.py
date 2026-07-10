"""Utilities for loading the external PLUX Python API on Windows."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType


class PluxAPIError(RuntimeError):
    """Raised when the external PLUX Python API cannot be loaded."""


_DLL_DIRECTORY_HANDLES: dict[str, object] = {}


def load_plux_api(api_path: str | Path) -> ModuleType:
    """Load the external PLUX API from a local SDK directory."""

    directory = Path(api_path).expanduser()

    if not directory.is_dir():
        raise PluxAPIError(
            f"PLUX API directory was not found: {directory}"
        )

    directory = directory.resolve()
    directory_text = str(directory)

    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        if directory_text not in _DLL_DIRECTORY_HANDLES:
            try:
                handle = os.add_dll_directory(directory_text)
            except OSError as exc:
                raise PluxAPIError(
                    f"Could not register the PLUX DLL directory: {exc}"
                ) from exc

            # Keep the handle alive while the process is running.
            _DLL_DIRECTORY_HANDLES[directory_text] = handle

    if directory_text not in sys.path:
        sys.path.insert(0, directory_text)

    importlib.invalidate_caches()

    existing_module = sys.modules.get("plux")
    if existing_module is not None:
        existing_file = getattr(existing_module, "__file__", None)

        if existing_file:
            existing_path = Path(existing_file).resolve()

            if (
                existing_path != directory
                and directory not in existing_path.parents
            ):
                raise PluxAPIError(
                    "A different plux module is already loaded from: "
                    f"{existing_path}"
                )

        return existing_module

    try:
        module = importlib.import_module("plux")
    except Exception as exc:
        raise PluxAPIError(
            "The PLUX API could not be imported from "
            f"{directory}: {type(exc).__name__}: {exc}"
        ) from exc

    if not hasattr(module, "SignalsDev"):
        raise PluxAPIError(
            "The imported plux module does not provide SignalsDev."
        )

    return module