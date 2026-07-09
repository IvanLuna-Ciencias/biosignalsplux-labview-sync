# biosignalsplux-labview-sync

Synchronized acquisition of surface electromyography signals from Biosignalsplux and joint trajectories from LabVIEW.

## Project objective

This project coordinates synchronized acquisition between:

- Biosignalsplux for two-channel sEMG acquisition.
- LabVIEW for robot or exoskeleton trajectory execution and joint-position recording.
- Additional external sensors, such as a force sensor, in future versions.

Python acts as the session coordinator and data acquisition host. LabVIEW remains responsible for robot control and trajectory execution.

## Planned acquisition flow

1. Create a shared session identifier.
2. Start Biosignalsplux acquisition.
3. Record a pre-start sEMG interval.
4. Send a synchronized START event to LabVIEW.
5. Execute and record the passive trajectory in LabVIEW.
6. Record timestamps and synchronization events.
7. Continue acquisition until an explicit STOP or safe interruption.
8. Preserve raw files and generate alignment reports afterward.

## Critical acquisition behavior

Acquisition and streaming must remain active until an explicit STOP command, user interruption, device error, or safe shutdown.

The acquisition must not stop automatically only because a theoretical protocol duration has elapsed.

## Planned session files

```text
outputs/
└── session_ID/
    ├── emg_ID.csv
    ├── trajectories_ID.csv
    ├── events_ID.csv
    ├── metadata_ID.json
    ├── sync_report_ID.json
    └── aligned_ID.csv
```

Raw files will be preserved. Aligned or combined files will be generated separately and will not overwrite the original recordings.

## Project status

Early development.

The current work focuses on:

- Minimal Biosignalsplux acquisition.
- Safe indefinite streaming.
- Timestamp and sequence validation.
- Simulated communication with LabVIEW.
- Reproducible session storage.

## Hardware and environment

- Windows
- Python 3.10.11
- Biosignalsplux
- Two sEMG channels
- Expected sEMG sampling frequency: 1000 Hz
- LabVIEW-based trajectory control and acquisition

## Data privacy

This public repository must not contain:

- Human participant recordings.
- Personal identifiers.
- Clinical information.
- Consent forms.
- Private laboratory configurations.
- External SDKs or proprietary installers.

Only synthetic or explicitly approved anonymized example data may be included.

## License

This project's original source code is released under the MIT License.

External SDKs, APIs, drivers, and LabVIEW components remain subject to their respective licenses.
