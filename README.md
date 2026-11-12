# Adaptive Realtime Multifacial Attendance System

A Python computer vision prototype by Nsikak Charles Etuk, developed as part of MSc work in Artificial Intelligence and Robotics. It recognises multiple faces from a webcam, records attendance locally and lets the operator register an unknown face during a session.

## Features

- HOG face detection and dlib-based face encodings through `face_recognition`.
- Nearest-face matching with a distance threshold of 0.5.
- CSV attendance: Name, Time and Date; one entry per person per local calendar day.
- Voice greetings and a crowd alert when more than five faces are detected.
- Operator-labelled registration of unknown faces without restarting.

## Setup

Use a local desktop with Python 3.10 or 3.11 and a webcam. These are suggested starting versions, not a tested compatibility matrix. The application needs a graphical desktop and audio support.

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\activate` on Windows, or `source .venv/bin/activate` on macOS/Linux, then:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python attendance.py
```

`face-recognition` depends on dlib. Installation may require CMake and a C++ compiler, especially on Windows. Linux speech output may need an eSpeak system package. Dependency versions are currently unpinned; a fully tested environment has not yet been established.

## Use

1. Put consented reference photos directly in `Attendee_images`, such as `Alex.jpg`. Use one clearly visible face per image. The filename becomes the person's label. Non-image files and images without a detectable face are skipped.
2. Run `python attendance.py` from your activated environment.
3. With an unknown-face preview focused, press **s**, then enter a label in the terminal. The terminal prompt pauses processing. Labels accept letters, numbers, spaces, hyphens and underscores. Existing files are not overwritten.
4. Press **q** with the main video window focused to stop.

Attendance is written beside the script. Unknown faces are shown locally; they are saved only when the operator chooses registration. Recognition is probabilistic: check results manually where errors matter.

## Data handling

Obtain informed permission before enrolling or recording anyone. Reference images and attendance records stay on the local computer and are excluded by `.gitignore`. Check staged changes before publishing. Set an appropriate retention period and delete records and reference images when no longer needed. No face data is included in this repository.

## Current limitations

This is a learning prototype, not a validated access-control or production attendance system. Lighting, pose and image quality affect matching. Speech and registration prompts block the camera loop; crowd alerts may repeat each frame. Multiple unknown faces can make registration selection ambiguous: enrol one person at a time. Existing malformed CSV records from the original script should be backed up and moved before running this version. No accuracy or timing claims have been independently reproduced.

## Verification

The uploaded source was cleaned and checked for Python syntax. Name/encoding alignment and CSV structure were corrected. Camera, face recognition, speech output and end-to-end behaviour still require local testing with installed dependencies.

## Development ideas

- Add non-blocking voice alerts and an alert cooldown.
- Add explicit face selection and a consent-led registration screen.
- Measure precision, recall, latency and performance under different lighting.
- Add automated tests around recognition mapping and attendance persistence.

## Licence

No licence has been selected yet. Choose a licence before inviting reuse or contributions.
