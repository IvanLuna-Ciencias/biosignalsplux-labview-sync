"""Initial GUI for synchronized Biosignalsplux-LabVIEW sessions."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from biosignalsplux_labview_sync.session_config import (
    SessionFormData,
    build_session_metadata,
    save_session_metadata,
)
from biosignalsplux_labview_sync.session_metadata import (
    MetadataError,
    generate_participant_id,
)
from biosignalsplux_labview_sync.trajectory import (
    CosineTrajectory,
    TrajectoryError,
)


JOINT_LABELS = {
    "shoulder": "Hombro",
    "elbow": "Codo",
    "rotation": "Rotación",
}

JOINT_LIMITS = {
    "shoulder": (-20.0, 140.0),
    "elbow": (0.0, 100.0),
    "rotation": (-40.0, 40.0),
}

JOINT_DEFAULT_END = {
    "shoulder": 60.0,
    "elbow": 90.0,
    "rotation": 20.0,
}


class SessionConfigurationWindow(QtWidgets.QWidget):
    """Collect, validate, summarize, and save session configuration."""

    def __init__(self) -> None:
        super().__init__()

        self.last_saved_path: Path | None = None

        self.setWindowTitle(
            "Configuración de adquisición Biosignalsplux–LabVIEW"
        )
        self.resize(820, 780)

        self._build_interface()
        self._connect_signals()
        self.generate_new_participant_id()
        self.update_joint_limits()
        self.update_trajectory_summary()

    def _build_interface(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel(
            "Sesión sincronizada sEMG–trayectoria"
        )
        title_font = title.font()
        title_font.setPointSize(15)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(title)

        subtitle = QtWidgets.QLabel(
            "Biosignalsplux a 1000 Hz · "
            "Tres referencias de posición hacia LabVIEW"
        )
        subtitle.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(subtitle)

        main_layout.addWidget(
            self._create_participant_group()
        )
        main_layout.addWidget(
            self._create_biosignals_group()
        )
        main_layout.addWidget(
            self._create_trajectory_group()
        )
        main_layout.addWidget(
            self._create_labview_group()
        )
        main_layout.addWidget(
            self._create_output_group()
        )

        self.summary_label = QtWidgets.QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setFrameShape(
            QtWidgets.QFrame.StyledPanel
        )
        self.summary_label.setMinimumHeight(90)
        main_layout.addWidget(self.summary_label)

        button_layout = QtWidgets.QHBoxLayout()

        self.validate_button = QtWidgets.QPushButton(
            "Validar configuración"
        )
        self.save_button = QtWidgets.QPushButton(
            "Guardar metadatos"
        )
        self.close_button = QtWidgets.QPushButton(
            "Cerrar"
        )

        button_layout.addWidget(self.validate_button)
        button_layout.addWidget(self.save_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)

        main_layout.addLayout(button_layout)

        self.status_label = QtWidgets.QLabel(
            "Complete los campos y valide la configuración."
        )
        main_layout.addWidget(self.status_label)

    def _create_participant_group(
        self,
    ) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(
            "Datos del participante"
        )
        form = QtWidgets.QFormLayout(group)

        id_layout = QtWidgets.QHBoxLayout()
        self.participant_id_edit = QtWidgets.QLineEdit()
        self.generate_id_button = QtWidgets.QPushButton(
            "Generar"
        )
        id_layout.addWidget(self.participant_id_edit)
        id_layout.addWidget(self.generate_id_button)
        form.addRow(
            "Identificador:",
            id_layout,
        )

        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText(
            "Nombre, iniciales o alias local"
        )
        form.addRow(
            "Nombre o alias:",
            self.name_edit,
        )

        self.age_spin = QtWidgets.QSpinBox()
        self.age_spin.setRange(1, 120)
        self.age_spin.setValue(25)
        form.addRow(
            "Edad:",
            self.age_spin,
        )

        self.sex_combo = QtWidgets.QComboBox()
        self.sex_combo.addItem(
            "Mujer",
            "female",
        )
        self.sex_combo.addItem(
            "Hombre",
            "male",
        )
        self.sex_combo.addItem(
            "Intersexual",
            "intersex",
        )
        self.sex_combo.addItem(
            "Prefiero no decirlo",
            "prefer_not_to_say",
        )
        form.addRow(
            "Sexo asignado al nacer:",
            self.sex_combo,
        )

        self.arm_combo = QtWidgets.QComboBox()
        self.arm_combo.addItem(
            "Derecho",
            "right",
        )
        self.arm_combo.addItem(
            "Izquierdo",
            "left",
        )
        form.addRow(
            "Brazo instrumentado:",
            self.arm_combo,
        )

        self.notes_edit = QtWidgets.QPlainTextEdit()
        self.notes_edit.setMaximumHeight(65)
        self.notes_edit.setPlaceholderText(
            "Observaciones de la sesión"
        )
        form.addRow(
            "Notas:",
            self.notes_edit,
        )

        return group

    def _create_biosignals_group(
        self,
    ) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(
            "Adquisición Biosignalsplux"
        )
        form = QtWidgets.QFormLayout(group)

        self.device_address_edit = QtWidgets.QLineEdit(
            os.environ.get(
                "BIOSIGNALSPLUX_DEVICE_ADDRESS",
                "",
            )
        )
        self.device_address_edit.setPlaceholderText(
            "Ejemplo: B00:00:00:00:00:00"
        )
        form.addRow(
            "Dirección del dispositivo:",
            self.device_address_edit,
        )

        sampling_label = QtWidgets.QLabel(
            "1000 Hz"
        )
        form.addRow(
            "Frecuencia de adquisición:",
            sampling_label,
        )

        resolution_label = QtWidgets.QLabel(
            "16 bits"
        )
        form.addRow(
            "Resolución:",
            resolution_label,
        )

        channels_label = QtWidgets.QLabel(
            "emg_0, emg_1 · máscara 0x03"
        )
        form.addRow(
            "Canales:",
            channels_label,
        )

        return group

    def _create_trajectory_group(
        self,
    ) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(
            "Configuración de trayectoria"
        )
        form = QtWidgets.QFormLayout(group)

        self.joint_combo = QtWidgets.QComboBox()

        for joint, label in JOINT_LABELS.items():
            self.joint_combo.addItem(
                label,
                joint,
            )

        self.joint_combo.setCurrentIndex(
            self.joint_combo.findData("elbow")
        )
        form.addRow(
            "Articulación activa:",
            self.joint_combo,
        )

        self.start_spin = QtWidgets.QDoubleSpinBox()
        self.start_spin.setDecimals(3)
        self.start_spin.setSingleStep(1.0)
        form.addRow(
            "Posición inicial (°):",
            self.start_spin,
        )

        self.end_spin = QtWidgets.QDoubleSpinBox()
        self.end_spin.setDecimals(3)
        self.end_spin.setSingleStep(1.0)
        form.addRow(
            "Posición final (°):",
            self.end_spin,
        )

        self.frequency_spin = QtWidgets.QDoubleSpinBox()
        self.frequency_spin.setDecimals(4)
        self.frequency_spin.setRange(0.001, 5.0)
        self.frequency_spin.setSingleStep(0.01)
        self.frequency_spin.setValue(0.1)
        form.addRow(
            "Frecuencia (Hz):",
            self.frequency_spin,
        )

        self.cycles_spin = QtWidgets.QSpinBox()
        self.cycles_spin.setRange(1, 1000)
        self.cycles_spin.setValue(5)
        form.addRow(
            "Número de ciclos:",
            self.cycles_spin,
        )

        self.setpoint_rate_spin = QtWidgets.QDoubleSpinBox()
        self.setpoint_rate_spin.setDecimals(1)
        self.setpoint_rate_spin.setRange(1.0, 1000.0)
        self.setpoint_rate_spin.setValue(50.0)
        form.addRow(
            "Envío de referencias (Hz):",
            self.setpoint_rate_spin,
        )

        self.move_to_start_spin = QtWidgets.QDoubleSpinBox()
        self.move_to_start_spin.setRange(0.1, 60.0)
        self.move_to_start_spin.setValue(3.0)
        self.move_to_start_spin.setSuffix(" s")
        form.addRow(
            "Movimiento a posición inicial:",
            self.move_to_start_spin,
        )

        self.return_duration_spin = QtWidgets.QDoubleSpinBox()
        self.return_duration_spin.setRange(0.1, 60.0)
        self.return_duration_spin.setValue(3.0)
        self.return_duration_spin.setSuffix(" s")
        form.addRow(
            "Retorno a neutral:",
            self.return_duration_spin,
        )

        self.neutral_hold_spin = QtWidgets.QDoubleSpinBox()
        self.neutral_hold_spin.setRange(0.0, 30.0)
        self.neutral_hold_spin.setValue(0.5)
        self.neutral_hold_spin.setSuffix(" s")
        form.addRow(
            "Espera en neutral:",
            self.neutral_hold_spin,
        )

        return group

    def _create_labview_group(
        self,
    ) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(
            "Comunicación LabVIEW"
        )
        form = QtWidgets.QFormLayout(group)

        self.labview_host_edit = QtWidgets.QLineEdit(
            os.environ.get(
                "LABVIEW_HOST",
                "172.22.11.2",
            )
        )
        form.addRow(
            "Host:",
            self.labview_host_edit,
        )

        self.labview_port_spin = QtWidgets.QSpinBox()
        self.labview_port_spin.setRange(1, 65535)
        self.labview_port_spin.setValue(5005)
        form.addRow(
            "Puerto UDP:",
            self.labview_port_spin,
        )

        self.labview_enabled_check = QtWidgets.QCheckBox(
            "Habilitar transmisión UDP durante la sesión"
        )
        self.labview_enabled_check.setChecked(True)
        form.addRow(
            "",
            self.labview_enabled_check,
        )

        return group

    def _create_output_group(
        self,
    ) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(
            "Almacenamiento"
        )
        form = QtWidgets.QFormLayout(group)

        output_layout = QtWidgets.QHBoxLayout()

        self.output_dir_edit = QtWidgets.QLineEdit(
            "outputs"
        )
        self.browse_button = QtWidgets.QPushButton(
            "Examinar"
        )

        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(self.browse_button)

        form.addRow(
            "Directorio base:",
            output_layout,
        )

        return group

    def _connect_signals(self) -> None:
        self.generate_id_button.clicked.connect(
            self.generate_new_participant_id
        )
        self.joint_combo.currentIndexChanged.connect(
            self.update_joint_limits
        )
        self.joint_combo.currentIndexChanged.connect(
            self.update_trajectory_summary
        )

        trajectory_widgets = (
            self.start_spin,
            self.end_spin,
            self.frequency_spin,
            self.cycles_spin,
            self.setpoint_rate_spin,
        )

        for widget in trajectory_widgets:
            widget.valueChanged.connect(
                self.update_trajectory_summary
            )

        self.validate_button.clicked.connect(
            self.validate_configuration
        )
        self.save_button.clicked.connect(
            self.save_configuration
        )
        self.browse_button.clicked.connect(
            self.select_output_directory
        )
        self.close_button.clicked.connect(
            self.close
        )

    def generate_new_participant_id(self) -> None:
        self.participant_id_edit.setText(
            generate_participant_id()
        )

    def update_joint_limits(self) -> None:
        joint = self.joint_combo.currentData()
        minimum, maximum = JOINT_LIMITS[joint]

        self.start_spin.setRange(
            minimum,
            maximum,
        )
        self.end_spin.setRange(
            minimum,
            maximum,
        )

        self.start_spin.setValue(0.0)
        self.end_spin.setValue(
            JOINT_DEFAULT_END[joint]
        )

        self.update_trajectory_summary()

    def update_trajectory_summary(self) -> None:
        try:
            trajectory = CosineTrajectory(
                joint=self.joint_combo.currentData(),
                start_deg=self.start_spin.value(),
                end_deg=self.end_spin.value(),
                frequency_hz=self.frequency_spin.value(),
                cycles=self.cycles_spin.value(),
            )
        except TrajectoryError as exc:
            self.summary_label.setText(
                f"Configuración de trayectoria inválida: {exc}"
            )
            return

        minimum, maximum = JOINT_LIMITS[
            trajectory.joint
        ]

        self.summary_label.setText(
            "<b>Resumen del protocolo</b><br>"
            f"Articulación: {JOINT_LABELS[trajectory.joint]}<br>"
            f"Rango configurado: "
            f"{trajectory.start_deg:.2f}° → "
            f"{trajectory.end_deg:.2f}° "
            f"(límites de software: {minimum:.1f}° a "
            f"{maximum:.1f}°)<br>"
            f"Duración de trayectoria: "
            f"{trajectory.duration_s:.2f} s · "
            f"Velocidad máxima estimada: "
            f"{trajectory.peak_velocity_deg_s:.2f} °/s · "
            f"Aceleración máxima estimada: "
            f"{trajectory.peak_acceleration_deg_s2:.2f} °/s²"
        )

    def select_output_directory(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Seleccionar directorio de salida",
            self.output_dir_edit.text() or ".",
        )

        if selected:
            self.output_dir_edit.setText(selected)

    def collect_form_data(self) -> SessionFormData:
        return SessionFormData(
            participant_id=self.participant_id_edit.text(),
            name_or_alias=self.name_edit.text(),
            age_years=self.age_spin.value(),
            sex_assigned_at_birth=self.sex_combo.currentData(),
            instrumented_arm=self.arm_combo.currentData(),
            notes=self.notes_edit.toPlainText(),
            device_address=self.device_address_edit.text(),
            joint=self.joint_combo.currentData(),
            start_deg=self.start_spin.value(),
            end_deg=self.end_spin.value(),
            frequency_hz=self.frequency_spin.value(),
            cycles=self.cycles_spin.value(),
            setpoint_rate_hz=self.setpoint_rate_spin.value(),
            move_to_start_s=self.move_to_start_spin.value(),
            return_duration_s=self.return_duration_spin.value(),
            neutral_hold_s=self.neutral_hold_spin.value(),
            labview_host=self.labview_host_edit.text(),
            labview_port=self.labview_port_spin.value(),
            labview_enabled=(
                self.labview_enabled_check.isChecked()
            ),
        )

    def build_metadata(self):
        return build_session_metadata(
            self.collect_form_data()
        )

    def validate_configuration(self) -> None:
        try:
            metadata = self.build_metadata()
        except (
            MetadataError,
            TrajectoryError,
            ValueError,
        ) as exc:
            self.status_label.setText(
                f"Error de validación: {exc}"
            )
            QtWidgets.QMessageBox.warning(
                self,
                "Configuración inválida",
                str(exc),
            )
            return

        self.status_label.setText(
            f"Configuración válida. "
            f"Sesión prevista: {metadata.session_id}"
        )

        QtWidgets.QMessageBox.information(
            self,
            "Configuración válida",
            "Todos los campos obligatorios y los parámetros "
            "de trayectoria son válidos.",
        )

    def save_configuration(self) -> None:
        try:
            metadata = self.build_metadata()

            base_output_dir = self.output_dir_edit.text().strip()

            if not base_output_dir:
                raise ValueError(
                    "El directorio de salida no puede estar vacío."
                )

            saved_path = save_session_metadata(
                base_output_dir=base_output_dir,
                metadata=metadata,
            )

        except (
            MetadataError,
            TrajectoryError,
            ValueError,
            OSError,
        ) as exc:
            self.status_label.setText(
                f"No se pudo guardar: {exc}"
            )
            QtWidgets.QMessageBox.critical(
                self,
                "Error al guardar",
                str(exc),
            )
            return

        self.last_saved_path = saved_path

        self.status_label.setText(
            f"Metadatos guardados en: {saved_path}"
        )

        QtWidgets.QMessageBox.information(
            self,
            "Configuración guardada",
            "La configuración de la sesión fue guardada en:\n\n"
            f"{saved_path}",
        )


def run_configuration_window() -> int:
    """Launch the standalone configuration window."""

    app = QtWidgets.QApplication.instance()

    owns_application = app is None

    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    window = SessionConfigurationWindow()
    window.show()

    if owns_application:
        return app.exec_()

    return 0