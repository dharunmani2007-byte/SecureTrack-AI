# SecureTrack AI

SecureTrack AI is an educational prototype for reviewing video and flagging **potentially unattended bags**. It uses object detection and tracking to help a human operator review CCTV footage.

> **Safety note:** AI alerts are not proof that an object is abandoned or dangerous. A security professional must verify every alert. Use only footage you own or are authorized to process, follow local privacy rules, and avoid uploading identifiable footage to public repositories.

## Features

- Detects people, backpacks, handbags, and suitcases with a pretrained YOLO model.
- Uses ByteTrack tracking through Ultralytics to associate detections across frames.
- Uses a simple proximity heuristic to estimate whether a person is near a tracked bag.
- Flags tracked bags that remain away from detected people for a configurable time.
- Shows an annotated video and review prompts in a Streamlit dashboard.
- Includes unit tests for the core logic.

## Limitations

- This is a proof of concept, not a production security system.
- “Near a person” is a rough geometric heuristic; it does not establish ownership.
- Detection, tracking, lighting, occlusion, camera angle, and crowding can cause false positives or missed detections.
- Only bags with a usable tracking ID are considered by this prototype.
- The model downloads its weights on first use; internet access may be required.
- Processing may be slower than real time, depending on the computer and video resolution.

## Requirements

- Python 3.10 or newer (Python 3.11 recommended)
- Internet access for the first model-weight download
- A computer capable of running OpenCV and a small YOLO model

## Setup

### 1. Download or clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/SecureTrack-AI.git
cd SecureTrack-AI
```

Or download the ZIP from GitHub and extract it.

### 2. Create a virtual environment

Windows:

```bash
py -m venv .venv
.venv\\Scripts\\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

A browser tab should open. Upload a short MP4, AVI, MOV, or MKV video and click **Start monitoring**. On the first run, the YOLO model weights may download automatically.

## Run tests

```bash
pytest -q
```

The tests check the proximity heuristic and unattended timer without needing a GPU or video file.

## Sample data

The repository intentionally does not include CCTV footage. Add only public sample footage with a suitable license or footage you have permission to use. Keep the video small and avoid committing personal or sensitive recordings.

Suggested folder:

```text
sample_data/
  README.md
  demo_video.mp4   # optional; only if you have permission to redistribute
```

See `sample_data/README.md` for guidance.

## Project structure

```text
SecureTrack-AI/
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── src/
│   ├── __init__.py
│   └── logic.py
├── tests/
│   └── test_logic.py
├── docs/
│   └── architecture.md
└── sample_data/
    └── README.md
```

## Architecture

Read [`docs/architecture.md`](docs/architecture.md).

## Privacy and responsible use

- Get permission before recording or processing video.
- Do not commit private CCTV clips, faces, names, or personal information.
- Restrict access to uploaded footage and delete temporary files after use.
- Treat alerts as prompts for human review, not automated conclusions.
- Before real-world deployment, add access controls, secure storage, audit logging, retention limits, and formal testing.

## License

The project code is offered under the MIT License in `LICENSE`. The pretrained model and third-party packages may have separate terms; review their licenses before redistribution or commercial use.
