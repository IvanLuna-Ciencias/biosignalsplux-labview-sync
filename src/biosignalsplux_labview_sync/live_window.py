"""PyQtGraph window for live raw sEMG visualization."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from biosignalsplux_labview_sync.live_buffer import (
    EMGBufferSnapshot,
    EMGLiveBuffer,
)


class EMGLiveWindowError(ValueError):
    """Raised when live-plot data or window settings are invalid."""


def snapshot_to_plot_arrays(
    snapshot: EMGBufferSnapshot,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Convert one immutable buffer snapshot into aligned NumPy arrays."""

    time_values = np.asarray(
        snapshot.time_monotonic_s,
        dtype=np.float64,
    )

    channels = tuple(
        np.asarray(channel, dtype=np.float64)
        for channel in snapshot.channels
    )

    for channel in channels:
        if channel.size != time_values.size:
            raise EMGLiveWindowError(
                "Time and channel arrays must contain the same "
                "number of samples."
            )

    if time_values.size:
        # Display the newest sample at t = 0 s.
        time_values = time_values - time_values[-1]

    return time_values, channels


class EMGLiveWindow(QtWidgets.QMainWindow):
    """Display the most recent raw sEMG samples from an EMGLiveBuffer."""

    def __init__(
        self,
        live_buffer: EMGLiveBuffer,
        channel_names: list[str] | tuple[str, ...],
        sampling_rate_hz: int,
        window_seconds: float = 5.0,
        refresh_interval_ms: int = 50,
        stop_callback: Callable[[], None] | None = None,
        show_stop_button: bool = True,
    ) -> None:
        super().__init__()

        self.live_buffer = live_buffer
        self.channel_names = tuple(channel_names)
        self.sampling_rate_hz = sampling_rate_hz
        self.window_seconds = float(window_seconds)
        self.refresh_interval_ms = refresh_interval_ms
        self.stop_callback = stop_callback
        self.show_stop_button = bool(show_stop_button)

        if len(self.channel_names) != self.live_buffer.channel_count:
            raise EMGLiveWindowError(
                "channel_names must match the live-buffer channel count."
            )

        if sampling_rate_hz <= 0:
            raise EMGLiveWindowError(
                "sampling_rate_hz must be greater than zero."
            )

        if window_seconds <= 0:
            raise EMGLiveWindowError(
                "window_seconds must be greater than zero."
            )

        if refresh_interval_ms <= 0:
            raise EMGLiveWindowError(
                "refresh_interval_ms must be greater than zero."
            )

        self._stop_requested = False

        self.setWindowTitle("Biosignalsplux Live sEMG Monitor")
        self.resize(1100, 700)

        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        self.graph_widget = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.graph_widget, stretch=1)

        self.plots: list[pg.PlotItem] = []
        self.curves: list[pg.PlotDataItem] = []

        for channel_index, channel_name in enumerate(self.channel_names):
            plot = self.graph_widget.addPlot(
                row=channel_index,
                col=0,
                title=channel_name,
            )
            plot.showGrid(x=True, y=True)
            plot.setLabel("left", "Raw ADC")
            plot.setXRange(
                -self.window_seconds,
                0.0,
                padding=0.0,
            )

            if channel_index == len(self.channel_names) - 1:
                plot.setLabel("bottom", "Time", units="s")
            else:
                plot.hideAxis("bottom")

            curve = plot.plot()
            curve.setClipToView(True)
            curve.setDownsampling(
                auto=True,
                method="peak",
            )

            self.plots.append(plot)
            self.curves.append(curve)

        status_layout = QtWidgets.QHBoxLayout()

        self.status_label = QtWidgets.QLabel(
            "Status: waiting for samples"
        )
        status_layout.addWidget(self.status_label, stretch=1)

        self.stop_button = QtWidgets.QPushButton(
            "STOP acquisition safely"
        )
        self.stop_button.clicked.connect(self.request_stop)
        self.stop_button.setVisible(self.show_stop_button)
        status_layout.addWidget(self.stop_button)

        main_layout.addLayout(status_layout)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(self.refresh_interval_ms)

    def update_plot(self) -> None:
        """Refresh all displayed channels from one consistent snapshot."""

        try:
            snapshot = self.live_buffer.snapshot()
            time_values, channels = snapshot_to_plot_arrays(snapshot)

            for curve, channel_values in zip(
                self.curves,
                channels,
            ):
                curve.setData(time_values, channel_values)

            self.status_label.setText(
                "Status: acquiring | "
                f"Total samples: {snapshot.total_samples_received} | "
                f"Buffered: {snapshot.buffered_samples} | "
                f"Fs: {self.sampling_rate_hz} Hz"
            )

        except Exception as exc:
            # A display error must remain local to the GUI.
            self.status_label.setText(
                "Visualization warning: "
                f"{type(exc).__name__}: {exc}"
            )

    def request_stop(self) -> None:
        """Request acquisition termination only once."""

        if self._stop_requested:
            return

        self._stop_requested = True
        self.stop_button.setEnabled(False)
        self.status_label.setText(
            "Status: safe stop requested..."
        )

        if self.stop_callback is not None:
            try:
                self.stop_callback()
            except Exception as exc:
                self.status_label.setText(
                    "Stop callback warning: "
                    f"{type(exc).__name__}: {exc}"
                )

    def closeEvent(
        self,
        event: QtGui.QCloseEvent,
    ) -> None:
        """Request safe acquisition stop when the window is closed."""

        self.request_stop()
        self.timer.stop()
        event.accept()