"""Integrated GUI for Biosignalsplux acquisition and LabVIEW trajectories."""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PyQt5 import QtCore, QtGui, QtWidgets

from biosignalsplux_labview_sync.config import load_config
from biosignalsplux_labview_sync.live_window import EMGLiveWindow
from biosignalsplux_labview_sync.session_config_window import (
    SessionConfigurationWindow,
)
from biosignalsplux_labview_sync.session_runtime import (
    PreparedAcquisitionRuntime,
    build_acquisition_summary,
    prepare_acquisition_runtime,
)
from biosignalsplux_labview_sync.trajectory import (
    CosineTrajectory,
)
from biosignalsplux_labview_sync.trajectory_session import (
    TrajectoryExecutionSession,
)


@dataclass(frozen=True)
class AcquisitionRuntimeSettings:
    """Local hardware settings not stored in the public repository."""

    api_path: Path
    flush_interval_samples: int
    prestart_seconds: float
    live_window_seconds: float = 5.0


def runtime_settings_from_config(
    config: dict[str, Any],
) -> AcquisitionRuntimeSettings:
    """Extract local PLUX and session settings."""

    try:
        api_path_value = config["biosignalsplux"]["api_path"]
        flush_interval = config["session"][
            "flush_interval_samples"
        ]
        prestart_seconds = config["session"].get(
            "prestart_seconds",
            3.0,
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "La configuración local no contiene los campos requeridos."
        ) from exc

    if (
        not isinstance(api_path_value, str)
        or not api_path_value.strip()
    ):
        raise ValueError(
            "biosignalsplux.api_path debe contener una ruta."
        )

    if (
        not isinstance(flush_interval, int)
        or isinstance(flush_interval, bool)
        or flush_interval <= 0
    ):
        raise ValueError(
            "flush_interval_samples debe ser un entero positivo."
        )

    if (
        not isinstance(prestart_seconds, (int, float))
        or isinstance(prestart_seconds, bool)
        or not math.isfinite(float(prestart_seconds))
        or float(prestart_seconds) < 0
    ):
        raise ValueError(
            "prestart_seconds debe ser un número no negativo."
        )

    return AcquisitionRuntimeSettings(
        api_path=Path(api_path_value),
        flush_interval_samples=flush_interval,
        prestart_seconds=float(prestart_seconds),
        live_window_seconds=5.0,
    )


def format_acquisition_summary(
    summary: dict[str, object],
) -> str:
    """Format completed acquisition results."""

    return "\n".join(
        [
            f"Sesión: {summary['session_id']}",
            (
                "Motivo de terminación: "
                f"{summary['termination_reason']}"
            ),
            (
                "Muestras recibidas: "
                f"{summary['samples_received']}"
            ),
            (
                "Muestras escritas: "
                f"{summary['samples_written']}"
            ),
            (
                "Eventos de gap: "
                f"{summary['sequence_gap_events']}"
            ),
            (
                "Muestras faltantes: "
                f"{summary['missing_sequence_count']}"
            ),
            (
                "Secuencias no crecientes: "
                f"{summary['non_increasing_sequence_count']}"
            ),
            f"CSV sEMG: {summary['emg_csv']}",
            f"Metadatos: {summary['metadata_json']}",
        ]
    )


def _json_safe(value: object) -> object:
    if isinstance(value, Exception):
        return str(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item)
            for item in value
        ]

    return value


def write_sync_report(
    *,
    path: str | Path,
    session_id: str,
    session_duration_s: float,
    acquisition_summary: dict[str, object],
    trajectory_summary: dict[str, object] | None,
) -> Path:
    """Write the final synchronized-session report."""

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "schema_version": 1,
        "session_id": session_id,
        "finished_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "session_duration_s": float(
            session_duration_s
        ),
        "acquisition": _json_safe(
            acquisition_summary
        ),
        "trajectory": _json_safe(
            trajectory_summary
        ),
    }

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(output_path)
    return output_path


class SynchronizedSessionWindow(
    SessionConfigurationWindow
):
    """Configure and run one complete synchronized session."""

    def __init__(
        self,
        *,
        config_path: str | Path,
    ) -> None:
        self.config_path = Path(config_path)

        self.runtime: PreparedAcquisitionRuntime | None = None
        self.trajectory_session: (
            TrajectoryExecutionSession | None
        ) = None
        self.live_window: EMGLiveWindow | None = None

        self.settings: AcquisitionRuntimeSettings | None = None
        self.required_baseline_samples = 0

        self.pending_full_stop = False
        self.trajectory_completion_handled = False
        self.trajectory_started_once = False
        self.programmatic_live_close = False

        super().__init__()

        self.setWindowTitle(
            "Sesión sincronizada Biosignalsplux–LabVIEW"
        )
        self.resize(900, 900)

        self._load_local_defaults()
        self._add_runtime_controls()
        self._create_shortcuts()

        self.runtime_timer = QtCore.QTimer(self)
        self.runtime_timer.timeout.connect(
            self.check_runtime_state
        )

    def _load_local_defaults(self) -> None:
        config = load_config(self.config_path)
        settings = runtime_settings_from_config(config)
        self.settings = settings

        device_config = config.get(
            "biosignalsplux",
            {},
        )
        session_config = config.get(
            "session",
            {},
        )
        labview_config = config.get(
            "labview",
            {},
        )

        address = device_config.get(
            "device_address",
            "",
        )

        if address:
            self.device_address_edit.setText(
                str(address)
            )

        output_root = session_config.get(
            "output_root",
            "outputs",
        )
        self.output_dir_edit.setText(
            str(output_root)
        )

        command_channel = labview_config.get(
            "command_channel",
            {},
        )

        port = command_channel.get(
            "port",
            5005,
        )

        if isinstance(port, int):
            self.labview_port_spin.setValue(
                port
            )

    def _add_runtime_controls(self) -> None:
        group = QtWidgets.QGroupBox(
            "Control de la sesión"
        )
        layout = QtWidgets.QGridLayout(group)

        self.session_state_value = QtWidgets.QLabel(
            "SIN SESIÓN"
        )
        self.samples_value = QtWidgets.QLabel(
            "0"
        )
        self.baseline_value = QtWidgets.QLabel(
            "Pendiente"
        )
        self.trajectory_state_value = QtWidgets.QLabel(
            "IDLE"
        )
        self.references_value = QtWidgets.QLabel(
            "Hombro 0.000° · Codo 0.000° · Rotación 0.000°"
        )

        layout.addWidget(
            QtWidgets.QLabel("Estado general:"),
            0,
            0,
        )
        layout.addWidget(
            self.session_state_value,
            0,
            1,
        )
        layout.addWidget(
            QtWidgets.QLabel("Muestras sEMG:"),
            1,
            0,
        )
        layout.addWidget(
            self.samples_value,
            1,
            1,
        )
        layout.addWidget(
            QtWidgets.QLabel("Periodo basal:"),
            2,
            0,
        )
        layout.addWidget(
            self.baseline_value,
            2,
            1,
        )
        layout.addWidget(
            QtWidgets.QLabel("Trayectoria:"),
            3,
            0,
        )
        layout.addWidget(
            self.trajectory_state_value,
            3,
            1,
        )
        layout.addWidget(
            QtWidgets.QLabel("Referencias:"),
            4,
            0,
        )
        layout.addWidget(
            self.references_value,
            4,
            1,
        )

        self.start_session_button = QtWidgets.QPushButton(
            "1. Iniciar adquisición sEMG"
        )
        self.start_trajectory_button = QtWidgets.QPushButton(
            "2. Iniciar trayectoria"
        )
        self.pause_button = QtWidgets.QPushButton(
            "Pausar trayectoria"
        )
        self.stop_trajectory_button = QtWidgets.QPushButton(
            "Detener trayectoria y retornar"
        )
        self.stop_session_button = QtWidgets.QPushButton(
            "Detener adquisición y finalizar sesión"
        )

        self.start_session_button.setMinimumHeight(42)
        self.start_trajectory_button.setMinimumHeight(42)
        self.stop_session_button.setMinimumHeight(42)

        self.start_trajectory_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.stop_trajectory_button.setEnabled(False)
        self.stop_session_button.setEnabled(False)

        layout.addWidget(
            self.start_session_button,
            5,
            0,
            1,
            2,
        )
        layout.addWidget(
            self.start_trajectory_button,
            6,
            0,
            1,
            2,
        )
        layout.addWidget(
            self.pause_button,
            7,
            0,
        )
        layout.addWidget(
            self.stop_trajectory_button,
            7,
            1,
        )
        layout.addWidget(
            self.stop_session_button,
            8,
            0,
            1,
            2,
        )

        self.start_session_button.clicked.connect(
            self.start_session
        )
        self.start_trajectory_button.clicked.connect(
            self.start_trajectory
        )
        self.pause_button.clicked.connect(
            self.toggle_trajectory_pause
        )
        self.stop_trajectory_button.clicked.connect(
            self.request_trajectory_stop
        )
        self.stop_session_button.clicked.connect(
            self.request_full_stop
        )

        main_layout = self.layout()
        summary_index = main_layout.indexOf(
            self.summary_label
        )

        main_layout.insertWidget(
            summary_index + 1,
            group,
        )

    def _create_shortcuts(self) -> None:
        self.pause_shortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence(
                QtCore.Qt.Key_Space
            ),
            self,
        )
        self.pause_shortcut.setContext(
            QtCore.Qt.ApplicationShortcut
        )
        self.pause_shortcut.activated.connect(
            self.toggle_trajectory_pause
        )
        self.pause_shortcut.setEnabled(False)

        self.stop_shortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence(
                QtCore.Qt.Key_Escape
            ),
            self,
        )
        self.stop_shortcut.setContext(
            QtCore.Qt.ApplicationShortcut
        )
        self.stop_shortcut.activated.connect(
            self.request_trajectory_stop
        )
        self.stop_shortcut.setEnabled(False)

    def set_configuration_enabled(
        self,
        enabled: bool,
    ) -> None:
        widgets = (
            self.participant_id_edit,
            self.generate_id_button,
            self.name_edit,
            self.age_spin,
            self.sex_combo,
            self.arm_combo,
            self.notes_edit,
            self.device_address_edit,
            self.joint_combo,
            self.start_spin,
            self.end_spin,
            self.frequency_spin,
            self.cycles_spin,
            self.setpoint_rate_spin,
            self.move_to_start_spin,
            self.return_duration_spin,
            self.neutral_hold_spin,
            self.labview_host_edit,
            self.labview_port_spin,
            self.labview_enabled_check,
            self.output_dir_edit,
            self.browse_button,
            self.validate_button,
            self.save_button,
            self.start_session_button,
        )

        for widget in widgets:
            widget.setEnabled(enabled)

    def start_session(self) -> None:
        """Start continuous Biosignalsplux acquisition."""

        if self.runtime is not None:
            QtWidgets.QMessageBox.warning(
                self,
                "Sesión activa",
                "Ya existe una sesión activa.",
            )
            return

        try:
            metadata = self.build_metadata()
            config = load_config(
                self.config_path
            )
            settings = runtime_settings_from_config(
                config
            )

            output_root = (
                self.output_dir_edit.text().strip()
            )

            if not output_root:
                raise ValueError(
                    "El directorio de salida no puede estar vacío."
                )

            runtime = prepare_acquisition_runtime(
                metadata=metadata,
                output_root=output_root,
                api_path=settings.api_path,
                flush_interval_samples=(
                    settings.flush_interval_samples
                ),
                live_window_seconds=(
                    settings.live_window_seconds
                ),
            )

            live_window = EMGLiveWindow(
                live_buffer=runtime.live_buffer,
                channel_names=list(
                    metadata.biosignalsplux.channels
                ),
                sampling_rate_hz=(
                    metadata.biosignalsplux.sampling_rate_hz
                ),
                window_seconds=(
                    settings.live_window_seconds
                ),
                refresh_interval_ms=50,
                stop_callback=self.request_full_stop,
            )
            live_window.setWindowTitle(
                f"sEMG en vivo · {metadata.session_id}"
            )

        except Exception as exc:
            self.status_label.setText(
                f"No se pudo iniciar: {exc}"
            )
            QtWidgets.QMessageBox.critical(
                self,
                "Error al iniciar la sesión",
                str(exc),
            )
            return

        self.runtime = runtime
        self.settings = settings
        self.live_window = live_window

        self.trajectory_session = None
        self.pending_full_stop = False
        self.trajectory_completion_handled = False
        self.trajectory_started_once = False

        sampling_rate = (
            metadata.biosignalsplux.sampling_rate_hz
        )
        self.required_baseline_samples = max(
            1,
            int(
                math.ceil(
                    settings.prestart_seconds
                    * sampling_rate
                )
            ),
        )

        self.set_configuration_enabled(False)
        self.stop_session_button.setEnabled(True)

        self.session_state_value.setText(
            "ABRIENDO DISPOSITIVO"
        )
        self.baseline_value.setText(
            (
                f"0 / {self.required_baseline_samples} "
                "muestras"
            )
        )
        self.status_label.setText(
            "Iniciando adquisición continua a 1000 Hz..."
        )

        self.live_window.show()
        self.runtime.acquisition.start()
        self.runtime_timer.start(100)

    def start_trajectory(self) -> None:
        """Start the configured trajectory without stopping sEMG."""

        if self.runtime is None:
            return

        if self.trajectory_started_once:
            QtWidgets.QMessageBox.warning(
                self,
                "Trayectoria ya ejecutada",
                "Esta sesión admite una trayectoria completa. "
                "Finalice la adquisición para iniciar otra sesión.",
            )
            return

        snapshot = self.runtime.live_buffer.snapshot()

        if (
            snapshot.total_samples_received
            < self.required_baseline_samples
        ):
            QtWidgets.QMessageBox.warning(
                self,
                "Periodo basal incompleto",
                "Aún no se ha completado el periodo basal.",
            )
            return

        metadata = self.runtime.metadata
        trajectory_metadata = metadata.trajectory

        try:
            trajectory = CosineTrajectory(
                joint=trajectory_metadata.joint,
                start_deg=trajectory_metadata.start_deg,
                end_deg=trajectory_metadata.end_deg,
                frequency_hz=(
                    trajectory_metadata.frequency_hz
                ),
                cycles=trajectory_metadata.cycles,
            )

            trajectory_session = TrajectoryExecutionSession(
                trajectory=trajectory,
                setpoint_rate_hz=(
                    trajectory_metadata.setpoint_rate_hz
                ),
                move_to_start_s=(
                    trajectory_metadata.move_to_start_s
                ),
                return_duration_s=(
                    trajectory_metadata.return_duration_s
                ),
                neutral_hold_s=(
                    trajectory_metadata.neutral_hold_s
                ),
                trajectories_csv=(
                    self.runtime.paths.trajectories_csv
                ),
                events_csv=(
                    self.runtime.paths.events_csv
                ),
                session_origin=(
                    self.runtime.session_origin
                ),
                labview_host=metadata.labview.host,
                labview_port=metadata.labview.port,
                udp_enabled=metadata.labview.enabled,
            )

        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Error de trayectoria",
                str(exc),
            )
            return

        self.trajectory_session = trajectory_session
        self.trajectory_started_once = True
        self.trajectory_completion_handled = False

        self.start_trajectory_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_trajectory_button.setEnabled(True)
        self.pause_shortcut.setEnabled(True)
        self.stop_shortcut.setEnabled(True)

        self.trajectory_state_value.setText(
            "MOVE_TO_START"
        )
        self.status_label.setText(
            "Trayectoria iniciada. "
            "Space pausa/reanuda · Esc retorna a neutral."
        )

        self.trajectory_session.start()

    def toggle_trajectory_pause(self) -> None:
        """Pause or resume trajectory time."""

        if (
            self.trajectory_session is None
            or not self.trajectory_session.is_running
        ):
            return

        paused = self.trajectory_session.toggle_pause()

        if paused:
            self.pause_button.setText(
                "Reanudar trayectoria"
            )
            self.status_label.setText(
                "Trayectoria pausada. "
                "La adquisición continúa a 1000 Hz."
            )
        else:
            self.pause_button.setText(
                "Pausar trayectoria"
            )
            self.status_label.setText(
                "Trayectoria reanudada."
            )

    def request_trajectory_stop(self) -> None:
        """Request a controlled return without stopping sEMG."""

        if (
            self.trajectory_session is None
            or not self.trajectory_session.is_running
        ):
            return

        self.trajectory_session.request_stop()
        self.pause_button.setEnabled(False)
        self.stop_trajectory_button.setEnabled(False)
        self.pause_shortcut.setEnabled(False)
        self.stop_shortcut.setEnabled(False)

        self.status_label.setText(
            "Retorno controlado solicitado. "
            "La adquisición sEMG continúa."
        )

    def request_full_stop(self) -> None:
        """Stop trajectory safely and then stop acquisition."""

        if self.programmatic_live_close:
            return

        if self.runtime is None:
            return

        if self.pending_full_stop:
            return

        self.pending_full_stop = True
        self.stop_session_button.setEnabled(False)

        if (
            self.trajectory_session is not None
            and self.trajectory_session.is_running
        ):
            self.trajectory_session.request_stop()
            self.status_label.setText(
                "Finalizando trayectoria y retornando a neutral "
                "antes de cerrar la adquisición..."
            )
        else:
            self.runtime.acquisition.request_stop()
            self.status_label.setText(
                "Deteniendo adquisición de forma segura..."
            )

    def _handle_trajectory_finished(self) -> None:
        if (
            self.trajectory_session is None
            or self.trajectory_completion_handled
        ):
            return

        self.trajectory_completion_handled = True

        results = self.trajectory_session.results

        self.pause_button.setEnabled(False)
        self.stop_trajectory_button.setEnabled(False)
        self.pause_shortcut.setEnabled(False)
        self.stop_shortcut.setEnabled(False)
        self.pause_button.setText(
            "Pausar trayectoria"
        )

        if results.runtime_error is not None:
            self.status_label.setText(
                "La trayectoria terminó con error. "
                "La adquisición continúa."
            )
            QtWidgets.QMessageBox.critical(
                self,
                "Error de trayectoria",
                str(results.runtime_error),
            )
        else:
            self.status_label.setText(
                "Trayectoria finalizada y referencia neutral alcanzada. "
                "La adquisición continúa hasta STOP."
            )

        if (
            self.pending_full_stop
            and self.runtime is not None
            and self.runtime.acquisition.is_running
        ):
            self.runtime.acquisition.request_stop()

    def check_runtime_state(self) -> None:
        """Refresh states and coordinate orderly shutdown."""

        if self.runtime is None:
            return

        snapshot = self.runtime.live_buffer.snapshot()
        sample_count = snapshot.total_samples_received

        self.samples_value.setText(
            str(sample_count)
        )

        if (
            not self.runtime.acquisition.is_finished
        ):
            self.session_state_value.setText(
                "ADQUIRIENDO A 1000 Hz"
            )

        if sample_count >= self.required_baseline_samples:
            self.baseline_value.setText(
                "COMPLETO"
            )

            if (
                not self.trajectory_started_once
                and not self.pending_full_stop
            ):
                self.start_trajectory_button.setEnabled(
                    True
                )
        else:
            self.baseline_value.setText(
                (
                    f"{sample_count} / "
                    f"{self.required_baseline_samples} muestras"
                )
            )

        if self.trajectory_session is not None:
            state = self.trajectory_session.state
            references = (
                self.trajectory_session.current_references_deg
            )

            self.trajectory_state_value.setText(
                state
            )
            self.references_value.setText(
                f"Hombro {references[0]:.3f}° · "
                f"Codo {references[1]:.3f}° · "
                f"Rotación {references[2]:.3f}°"
            )

            if (
                self.trajectory_session.is_finished
                and not self.trajectory_completion_handled
            ):
                self._handle_trajectory_finished()

        if self.runtime.acquisition.is_finished:
            if (
                self.trajectory_session is not None
                and self.trajectory_session.is_running
            ):
                self.pending_full_stop = True
                self.trajectory_session.request_stop()
                self.status_label.setText(
                    "La adquisición terminó antes de lo esperado. "
                    "Retornando la referencia a neutral..."
                )
                return

            self.finish_session()
            return

        if (
            self.pending_full_stop
            and (
                self.trajectory_session is None
                or self.trajectory_session.is_finished
            )
        ):
            self.runtime.acquisition.request_stop()

    def _trajectory_summary(
        self,
    ) -> dict[str, object] | None:
        if self.trajectory_session is None:
            return None

        results = self.trajectory_session.results

        return {
            "termination_reason": results.termination_reason,
            "runtime_error": results.runtime_error,
            "close_error": results.close_error,
            "setpoints_generated": results.setpoints_generated,
            "udp_packets_sent": results.udp_packets_sent,
            "pause_count": results.pause_count,
            "final_state": results.final_state,
            "trajectories_csv": str(
                self.runtime.paths.trajectories_csv
            ) if self.runtime is not None else None,
            "events_csv": str(
                self.runtime.paths.events_csv
            ) if self.runtime is not None else None,
        }

    def finish_session(self) -> None:
        """Close all resources, write report, and show results."""

        if self.runtime is None:
            return

        self.runtime_timer.stop()

        self.runtime.acquisition.join(
            timeout=1.0
        )

        if (
            self.trajectory_session is not None
            and self.trajectory_session.is_running
        ):
            self.trajectory_session.request_stop()
            self.trajectory_session.join(
                timeout=(
                    self.runtime.metadata.trajectory
                    .return_duration_s
                    + 5.0
                )
            )

        self.programmatic_live_close = True

        if self.live_window is not None:
            try:
                self.live_window.timer.stop()
            except Exception:
                pass

            self.live_window.close()
            self.live_window = None

        self.programmatic_live_close = False

        acquisition_summary = build_acquisition_summary(
            self.runtime
        )
        trajectory_summary = self._trajectory_summary()

        session_duration_s = max(
            0.0,
            time.perf_counter()
            - self.runtime.session_origin,
        )

        report_path = write_sync_report(
            path=self.runtime.paths.sync_report_json,
            session_id=self.runtime.metadata.session_id,
            session_duration_s=session_duration_s,
            acquisition_summary=acquisition_summary,
            trajectory_summary=trajectory_summary,
        )

        summary_text = (
            format_acquisition_summary(
                acquisition_summary
            )
            + "\n"
            + f"Reporte final: {report_path}"
        )

        has_error = any(
            acquisition_summary.get(name) is not None
            for name in (
                "runtime_error",
                "request_stop_error",
                "stop_error",
                "close_error",
            )
        )

        if (
            trajectory_summary is not None
            and (
                trajectory_summary.get(
                    "runtime_error"
                ) is not None
                or trajectory_summary.get(
                    "close_error"
                ) is not None
            )
        ):
            has_error = True

        self.set_configuration_enabled(True)
        self.start_trajectory_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.stop_trajectory_button.setEnabled(False)
        self.stop_session_button.setEnabled(False)

        self.pause_shortcut.setEnabled(False)
        self.stop_shortcut.setEnabled(False)

        self.session_state_value.setText(
            "FINALIZADA"
        )
        self.trajectory_state_value.setText(
            (
                self.trajectory_session.state
                if self.trajectory_session is not None
                else "NO EJECUTADA"
            )
        )

        if has_error:
            self.status_label.setText(
                "La sesión terminó con uno o más errores."
            )
            QtWidgets.QMessageBox.critical(
                self,
                "Resumen de sesión",
                summary_text,
            )
        else:
            self.status_label.setText(
                "Sesión finalizada correctamente."
            )
            QtWidgets.QMessageBox.information(
                self,
                "Resumen de sesión",
                summary_text,
            )

        self.runtime = None
        self.trajectory_session = None
        self.live_window = None
        self.pending_full_stop = False
        self.trajectory_completion_handled = False
        self.trajectory_started_once = False

    def closeEvent(
        self,
        event: QtGui.QCloseEvent,
    ) -> None:
        """Perform controlled shutdown before closing."""

        if (
            self.trajectory_session is not None
            and self.trajectory_session.is_running
        ):
            self.trajectory_session.request_stop()
            self.trajectory_session.join(
                timeout=(
                    self.trajectory_session
                    .return_duration_s
                    + 5.0
                )
            )

        if (
            self.runtime is not None
            and self.runtime.acquisition.is_running
        ):
            self.runtime.acquisition.request_stop()
            stopped = self.runtime.acquisition.join(
                timeout=10.0
            )

            if not stopped:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Cierre incompleto",
                    "El hilo de adquisición no terminó "
                    "dentro de 10 segundos.",
                )

        self.programmatic_live_close = True

        if self.live_window is not None:
            try:
                self.live_window.timer.stop()
            except Exception:
                pass

            self.live_window.close()

        event.accept()


def run_synchronized_session_window(
    *,
    config_path: str | Path,
) -> int:
    """Launch the complete synchronized-session GUI."""

    app = QtWidgets.QApplication.instance()
    owns_application = app is None

    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    window = SynchronizedSessionWindow(
        config_path=config_path,
    )
    window.show()

    if owns_application:
        return app.exec_()

    return 0