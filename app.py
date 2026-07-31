import time
import threading
from collections import deque
from datetime import datetime

import cv2
from flask import Flask, Response, jsonify, render_template
from ultralytics import YOLO

MODEL_PATH = "best.pt"
CAMERA_INDEX = 0
CAMERA_NAME = "Camera 1 - Main Floor"
CONFIDENCE_THRESHOLD = 0.4
VIOLATION_COOLDOWN_SECONDS = 120
MAX_LOGS = 300
ROLLING_WINDOW_SECONDS = 4

CLASS_COLORS = {
    "helmet": (0, 200, 0),
    "gloves": (200, 130, 0),
    "vest": (0, 165, 255),
    "goggles": (200, 200, 0),
    "none": (120, 120, 120),
    "Person": (255, 200, 0),
    "no_helmet": (0, 0, 230),
    "no_goggle": (0, 60, 220),
    "no_gloves": (0, 100, 220),
    "no_vest": (0, 0, 200),
}

POSITIVE_CLASSES = {"helmet", "gloves", "vest", "goggles"}
VIOLATION_CLASSES = {"no_helmet", "no_goggle", "no_gloves", "no_vest", "none"}
REQUIRED_PPE = ["helmet", "vest", "goggles", "gloves"]
VIOLATION_TO_ITEM = {
    "no_helmet": "helmet",
    "no_vest": "vest",
    "no_goggle": "goggles",
    "no_gloves": "gloves",
}

VIOLATION_META = {
    "no_helmet": {
        "severity": "High",
        "description": "Worker detected without safety helmet in designated PPE zone",
    },
    "no_vest": {
        "severity": "High",
        "description": "Worker detected without safety vest in designated PPE zone",
    },
    "no_goggle": {
        "severity": "Medium",
        "description": "Worker detected without safety goggles in designated PPE zone",
    },
    "no_gloves": {
        "severity": "Medium",
        "description": "Worker detected without safety gloves in designated PPE zone",
    },
    "none": {
        "severity": "High",
        "description": "Worker detected with no PPE equipment worn in designated PPE zone",
    },
}

# Live on-frame banner text shown only while the violation is present in the
# CURRENT frame (no cooldown, no smoothing) — disappears the instant the
# item is detected again, same behavior as the reference video.
BANNER_LABELS = {
    "no_helmet": "HELMET MISSING",
    "no_vest": "VEST MISSING",
    "no_goggle": "GOGGLES MISSING",
    "no_gloves": "GLOVES MISSING",
    "none": "PPE MISSING",
}

app = Flask(__name__)

state_lock = threading.Lock()
state = {
    "current_persons": 0,
    "total_persons_seen": 0,
    "total_logs": 0,
    "active_alerts": [],
    "ppe_status": {item: "unknown" for item in REQUIRED_PPE},
    "camera_status": "connecting",
}
logs = deque(maxlen=MAX_LOGS)
last_logged_at = {}
ppe_status = {item: "unknown" for item in REQUIRED_PPE}
frame_history = deque()

model = YOLO(MODEL_PATH)


def make_log_entry(violation_class):
    meta = VIOLATION_META[violation_class]
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "camera": CAMERA_NAME,
        "violation_type": "PPE Non-Compliance",
        "description": meta["description"],
        "severity": meta["severity"],
        "assigned_to": "Site Supervisor",
    }
    return entry


def process_frame(frame):
    results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

    persons = 0
    now = time.time()
    new_logs = []
    active_violations = set()  # real-time, this frame only — drives the banner

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls_name == "Person":
                persons += 1
            elif cls_name in POSITIVE_CLASSES:
                ppe_status[cls_name] = "detected"
            elif cls_name in VIOLATION_CLASSES:
                active_violations.add(cls_name)
                if cls_name == "none":
                    for item in REQUIRED_PPE:
                        ppe_status[item] = "missing"
                else:
                    item = VIOLATION_TO_ITEM.get(cls_name)
                    if item:
                        ppe_status[item] = "missing"
                last_time = last_logged_at.get(cls_name, 0)
                if now - last_time >= VIOLATION_COOLDOWN_SECONDS:
                    last_logged_at[cls_name] = now
                    new_logs.append(make_log_entry(cls_name))

            color = CLASS_COLORS.get(cls_name, (255, 255, 255))
            label = f"{cls_name} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Live "X MISSING" banner — only shows while the violation is present in
    # this exact frame, gone the instant the item is detected again.
    for i, cls_name in enumerate(sorted(active_violations)):
        banner_text = BANNER_LABELS.get(cls_name, f"{cls_name.upper()} MISSING")
        y = 40 + i * 38
        (tw, th), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (14, y - th - 10), (14 + tw + 16, y + 8), (0, 0, 0), -1)
        cv2.putText(frame, banner_text, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 230), 2)

    with state_lock:
        frame_history.append((now, persons))
        while frame_history and now - frame_history[0][0] > ROLLING_WINDOW_SECONDS:
            frame_history.popleft()

        smoothed_persons = max(p for _, p in frame_history)

        state["current_persons"] = smoothed_persons
        state["total_persons_seen"] += persons
        state["active_alerts"] = [BANNER_LABELS.get(c, c) for c in sorted(active_violations)]
        state["ppe_status"] = dict(ppe_status)

        for entry in new_logs:
            logs.appendleft(entry)
        state["total_logs"] = len(logs)

    return frame


def generate_frames():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        with state_lock:
            state["camera_status"] = "offline"
        return

    with state_lock:
        state["camera_status"] = "live"

    while True:
        ret, frame = cap.read()
        if not ret:
            with state_lock:
                state["camera_status"] = "offline"
            break

        frame = process_frame(frame)
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html", camera_name=CAMERA_NAME)


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stats")
def api_stats():
    with state_lock:
        return jsonify(dict(state))


@app.route("/api/logs")
def api_logs():
    with state_lock:
        return jsonify(list(logs))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)