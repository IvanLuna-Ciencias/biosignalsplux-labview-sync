# ATI force/torque acquisition layer

This patch adds an optional ATI six-axis force/torque acquisition layer to
`biosignalsplux-labview-sync`.

## Design

- One NI-DAQmx continuous task remains active for the whole recording.
- Sampling frequency defaults to 1000 Hz.
- The first configurable interval is averaged to obtain the voltage bias.
- All six raw voltages and six calibrated force/torque axes are preserved.
- Session-relative timestamps use the same `time.perf_counter()` origin that
  can later be shared with Biosignalsplux and trajectory acquisition.
- The acquisition remains active until explicit STOP, device error, or safe
  shutdown.
- Live display uses two synchronized panels: forces and torques.

## Files

- `ati_force.py`: calibration parsing and NI-DAQ device wrapper.
- `force_writer.py`: incremental CSV storage.
- `force_acquisition_session.py`: background continuous acquisition.
- `force_runtime.py`: optional JSON configuration and session preparation.
- `force_live_window.py`: live PyQtGraph display.
- `test_ati_force_acquisition.py`: standalone hardware validation.
- `plot_force_session.py`: post-session force/torque plot.

## Local configuration section

Merge the object from `configs/force_sensor.example.json` into
`configs/acquisition.local.json`. Keep the calibration file outside the public
repository.

## Expected output

`force_<session_id>.csv` contains:

- sample and session timing,
- six raw analog voltages,
- Fx, Fy, Fz in N,
- Mx, My, Mz in N·m,
- force and torque magnitudes.
