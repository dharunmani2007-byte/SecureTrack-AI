import time
from datetime import datetime
from pathlib import Path

import cv2
import streamlit as st
from ultralytics import YOLO

from src.logic import is_object_near_person, update_unattended_state

st.set_page_config(page_title="SecureTrack AI", page_icon="🛡️", layout="wide")
st.title("🛡️ SecureTrack AI")
st.caption("Prototype: person tracking and potentially unattended-belonging alerts")

st.warning(
    "Prototype only. AI can miss objects or raise false alarms. A trained security "
    "operator must verify every alert. Do not use this demo as the sole basis for action."
)

with st.sidebar:
    st.header("Settings")
    model_name = st.text_input("YOLO model", value="yolo11n.pt",
                               help="The model weights download automatically the first time.")
    unattended_seconds = st.slider("Unattended threshold (seconds)", 3, 60, 10)
    process_every_n = st.slider("Process every Nth frame", 1, 5, 2)
    st.caption("Lower frame skipping can improve detail but may run more slowly.")

uploaded = st.file_uploader("Upload a short CCTV/sample video", type=["mp4", "avi", "mov", "mkv"])

if uploaded is None:
    st.info("Upload a short video to begin. For privacy, use footage you own or are permitted to use.")
    st.markdown("""
    **What this prototype does**
    - Detects people, backpacks, handbags, and suitcases.
    - Tracks detected objects across processed frames when track IDs are available.
    - Flags a tracked bag-like object when no person is nearby for the selected time.
    - Shows an annotated video preview and timestamped alerts.
    """)
    st.stop()

if st.button("Start monitoring", type="primary"):
    suffix = Path(uploaded.name).suffix or ".mp4"
    temp_input = Path("securetrack_input" + suffix)
    temp_output = Path("securetrack_annotated.mp4")
    temp_input.write_bytes(uploaded.getvalue())

    cap = cv2.VideoCapture(str(temp_input))
    if not cap.isOpened():
        st.error("Could not open this video. Try an MP4 file.")
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    writer = cv2.VideoWriter(
        str(temp_output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (width, height)
    )

    progress = st.progress(0)
    status = st.empty()
    preview = st.empty()
    alert_area = st.container()
    alerts = []
    object_states = {}
    model = None
    frame_index = 0
    latest_frame = None
    camera_name = uploaded.name

    try:
        model = YOLO(model_name)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if (frame_index - 1) % process_every_n != 0:
                # Keep the output video smooth by writing skipped frames unchanged.
                writer.write(frame)
                continue

            result = model.track(
                frame, persist=True, tracker="bytetrack.yaml", verbose=False
            )[0]
            persons = []
            tracked_objects = []

            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().tolist()
                classes = result.boxes.cls.cpu().tolist()
                ids = result.boxes.id
                ids = ids.int().cpu().tolist() if ids is not None else [None] * len(xyxy)

                for box, cls_id, track_id in zip(xyxy, classes, ids):
                    label = model.names[int(cls_id)]
                    if label == "person":
                        persons.append(box)
                    elif label in {"backpack", "handbag", "suitcase"} and track_id is not None:
                        tracked_objects.append((int(track_id), label, box))

            now = time.monotonic()
            for track_id, label, box in tracked_objects:
                near_person = any(is_object_near_person(box, p) for p in persons)
                state = object_states.get(track_id)
                unattended, state = update_unattended_state(
                    state=state,
                    person_nearby=near_person,
                    now=now,
                    threshold_seconds=unattended_seconds,
                )
                object_states[track_id] = state

                x1, y1, x2, y2 = [int(v) for v in box]
                color = (0, 180, 0) if near_person else (0, 165, 255)
                text = f"{label} ID:{track_id}"
                if unattended:
                    color = (0, 0, 255)
                    text += " - CHECK"
                    if not state.get("alert_logged", False):
                        video_seconds = frame_index / fps
                        alert = {
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "video_time": f"{int(video_seconds // 60):02d}:{int(video_seconds % 60):02d}",
                            "camera": camera_name,
                            "object": label,
                            "track_id": track_id,
                            "message": "Potentially unattended - verify manually",
                        }
                        alerts.append(alert)
                        state["alert_logged"] = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, max(22, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Draw person boxes for clarity.
            for p in persons:
                x1, y1, x2, y2 = [int(v) for v in p]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 160, 0), 2)
                cv2.putText(frame, "person", (x1, max(22, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 160, 0), 2)

            writer.write(frame)
            latest_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_index % max(1, int(fps)) == 0:
                preview.image(latest_frame, caption=f"Processed frame {frame_index}")
            if total_frames:
                progress.progress(min(frame_index / total_frames, 1.0))
            status.write(f"Processing frame {frame_index} of {total_frames or 'unknown'}")
    except Exception as exc:
        st.error(f"Processing failed: {exc}")
        st.info("Check that dependencies are installed and the model name is valid.")
    finally:
        cap.release()
        writer.release()

    if temp_output.exists() and temp_output.stat().st_size > 0:
        st.subheader("Annotated video")
        st.video(str(temp_output))
        with open(temp_output, "rb") as f:
            st.download_button("Download annotated video", f, file_name="securetrack_annotated.mp4",
                               mime="video/mp4")

    st.subheader("Potentially unattended-object alerts")
    if alerts:
        st.dataframe(alerts, use_container_width=True)
        st.caption("These are review prompts, not confirmed security incidents.")
    else:
        st.success("No potential unattended-object alerts were generated in this run.")

    try:
        temp_input.unlink(missing_ok=True)
    except Exception:
        pass
