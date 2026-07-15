"""PyQtGraph monitor for six-axis ATI force/torque data."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from biosignalsplux_labview_sync.live_buffer import EMGLiveBuffer


class ForceLiveWindow(QtWidgets.QMainWindow):
    """Display forces and torques in two synchronized live panels."""

    def __init__(
        self,
        *,
        live_buffer: EMGLiveBuffer,
        sampling_rate_hz: int,
        window_seconds: float = 5.0,
        refresh_interval_ms: int = 50,
        stop_callback: Callable[[], None] | None = None,
        show_stop_button: bool = True,
    ) -> None:
        super().__init__()
        if live_buffer.channel_count != 6:
            raise ValueError("ForceLiveWindow requires a six-channel buffer.")

        self.live_buffer = live_buffer
        self.sampling_rate_hz = sampling_rate_hz
        self.window_seconds = float(window_seconds)
        self.stop_callback = stop_callback
        self.show_stop_button = bool(show_stop_button)
        self._stop_requested = False

        self.setWindowTitle("ATI force/torque monitor")
        self.resize(1100, 750)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        self.graph_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graph_widget, stretch=1)

        self.force_plot = self.graph_widget.addPlot(row=0, col=0, title="Fuerzas ATI")
        self.force_plot.showGrid(x=True, y=True)
        self.force_plot.setLabel("left", "Fuerza", units="N")
        self.force_plot.hideAxis("bottom")
        self.force_plot.setXRange(-self.window_seconds, 0.0, padding=0.0)

        self.torque_plot = self.graph_widget.addPlot(row=1, col=0, title="Torques ATI")
        self.torque_plot.showGrid(x=True, y=True)
        self.torque_plot.setLabel("left", "Torque", units="N·m")
        self.torque_plot.setLabel("bottom", "Tiempo", units="s")
        self.torque_plot.setXRange(-self.window_seconds, 0.0, padding=0.0)

        self.force_curves = [
            self.force_plot.plot(name=name)
            for name in ("Fx", "Fy", "Fz")
        ]
        self.torque_curves = [
            self.torque_plot.plot(name=name)
            for name in ("Mx", "My", "Mz")
        ]
        self.force_plot.addLegend()
        self.torque_plot.addLegend()

        for curve in (*self.force_curves, *self.torque_curves):
            curve.setClipToView(True)
            curve.setDownsampling(auto=True, method="peak")

        status_layout = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Esperando muestras ATI...")
        status_layout.addWidget(self.status_label, stretch=1)
        self.stop_button = QtWidgets.QPushButton("STOP sesión")
        self.stop_button.clicked.connect(self.request_stop)
        self.stop_button.setVisible(self.show_stop_button)
        status_layout.addWidget(self.stop_button)
        layout.addLayout(status_layout)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(refresh_interval_ms)

    def update_plot(self) -> None:
        try:
            snapshot = self.live_buffer.snapshot()
            times = np.asarray(snapshot.time_monotonic_s, dtype=np.float64)
            channels = [np.asarray(channel, dtype=np.float64) for channel in snapshot.channels]
            if times.size:
                times = times - times[-1]
            for curve, values in zip(self.force_curves, channels[:3]):
                curve.setData(times, values)
            for curve, values in zip(self.torque_curves, channels[3:]):
                curve.setData(times, values)
            self.status_label.setText(
                f"ATI adquiriendo | Total: {snapshot.total_samples_received} | "
                f"Buffer: {snapshot.buffered_samples} | Fs: {self.sampling_rate_hz} Hz"
            )
        except Exception as exc:
            self.status_label.setText(
                f"Advertencia de visualización ATI: {type(exc).__name__}: {exc}"
            )

    def request_stop(self) -> None:
        if self._stop_requested:
            return
        self._stop_requested = True
        self.stop_button.setEnabled(False)
        if self.stop_callback is not None:
            self.stop_callback()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.request_stop()
        self.timer.stop()
        event.accept()
