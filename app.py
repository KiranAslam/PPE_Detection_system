import os
import time
import threading
from collections import deque
from datetime import datetime

import cv2
from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO

MODEL_PATH = "best.pt"
CAMERA_INDEX = 0
CAMERA_NAME = "Camera 1 - Main Floor"
CONFIDENCE_THRESHOLD = 0.4
CAPTURE_SETTLE_SECONDS = 5  # wait this long after a person first appears before checking/capturing them
MAX_LOGS = 300
ROLLING_WINDOW_SECONDS = 4

# Person tracking tuning
TRACK_GRACE_SECONDS = 3       # how long a person can vanish from a frame before losing their ID
TRACK_IOU_THRESHOLD = 0.3     # minimum overlap to consider a detection "the same person" as a track

PHOTO_DIR = os.path.join("static", "violation_photos")
PHOTO_URL_PREFIX = "/static/violation_photos"
os.makedirs(PHOTO_DIR, exist_ok=True)

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
VIOLATION_TO_ITEM = {
    "no_helmet": "helmet",
    "no_vest": "vest",
    "no_goggle": "goggles",
    "no_gloves": "gloves",
}

ITEM_META = {
    "helmet": {"severity": "High", "description": "Worker detected without safety helmet"},
    "vest": {"severity": "High", "description": "Worker detected without safety vest"},
    "goggles": {"severity": "Medium", "description": "Worker detected without safety goggles"},
    "gloves": {"severity": "Medium", "description": "Worker detected without safety gloves"},
}

# Per-area PPE requirement checklist — only these items are checked while
# that area is active.
AREAS = {
    "area1": {"name": "Process / Refinery Unit", "required": ["helmet", "goggles", "gloves", "vest"]},
    "area2": {"name": "Tank Farm / Storage Yard", "required": ["helmet", "vest", "goggles"]},
    "area3": {"name": "Loading Gantry", "required": ["helmet", "vest", "gloves"]},
    "area4": {"name": "Workshop / Maintenance Bay", "required": ["helmet", "goggles", "gloves"]},
}
DEFAULT_AREA = "area1"

app = Flask(__name__)

state_lock = threading.Lock()
active_area = DEFAULT_AREA

# id -> {"bbox", "last_seen", "first_seen", "ppe_status", "saw_none", "logged"}
person_tracks = {}
frame_history = deque()
logs = deque(maxlen=MAX_LOGS)

state = {
    "current_persons": 0,
    "total_persons_seen": 0,
    "total_logs": 0,
    "active_area": DEFAULT_AREA,
    "active_area_name": AREAS[DEFAULT_AREA]["name"],
    "required_items": AREAS[DEFAULT_AREA]["required"],
    "ppe_status": {item: "unknown" for item in AREAS[DEFAULT_AREA]["required"]},
    "camera_status": "connecting",
}

model = YOLO(MODEL_PATH)


def make_log_entry(area_id, person_id, violation_text, severity, photo_url):
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "area": AREAS[area_id]["name"],
        "person_id": f"{person_id:02d}",
        "violation_type": violation_text,
        "severity": severity,
        "photo": photo_url,
    }


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def next_free_id():
    """Lowest ID not currently in use by a live track."""
    i = 1
    while i in person_tracks:
        i += 1
    return i


def match_persons(person_boxes, now, required_items):
    """Match this frame's person boxes to existing tracks (by IoU), spin up
    new tracks (lowest free ID) for anyone unmatched, and drop tracks that
    have been gone longer than the grace period (freeing their ID).
    Returns the list of track IDs present in THIS frame."""
    pairs = []
    for tid, track in person_tracks.items():
        for di, box in enumerate(person_boxes):
            score = iou(track["bbox"], box)
            if score >= TRACK_IOU_THRESHOLD:
                pairs.append((score, tid, di))
    pairs.sort(key=lambda p: p[0], reverse=True)

    used_tracks, used_dets = set(), set()
    matched_ids = []
    for score, tid, di in pairs:
        if tid in used_tracks or di in used_dets:
            continue
        used_tracks.add(tid)
        used_dets.add(di)
        person_tracks[tid]["bbox"] = person_boxes[di]
        person_tracks[tid]["last_seen"] = now
        matched_ids.append(tid)

    for di, box in enumerate(person_boxes):
        if di in used_dets:
            continue
        tid = next_free_id()
        person_tracks[tid] = {
            "bbox": box,
            "last_seen": now,
            "first_seen": now,
            "ppe_status": {item: "unknown" for item in required_items},
            "saw_none": False,
            "logged": False,
        }
        matched_ids.append(tid)

    stale_ids = [tid for tid, t in person_tracks.items() if now - t["last_seen"] > TRACK_GRACE_SECONDS]
    for tid in stale_ids:
        del person_tracks[tid]

    return matched_ids


def assign_gear_to_person(gear_box):
    """Return the track ID whose person box contains this gear box's center."""
    cx = (gear_box[0] + gear_box[2]) / 2
    cy = (gear_box[1] + gear_box[3]) / 2
    for tid, track in person_tracks.items():
        x1, y1, x2, y2 = track["bbox"]
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return tid
    return None


def process_frame(frame):
    results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    now = time.time()

    person_boxes = []
    gear_detections = []   # (cls_name, bbox)
    draw_items = []        # (cls_name, conf, bbox)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            bbox = tuple(map(int, box.xyxy[0]))
            draw_items.append((cls_name, conf, bbox))

            if cls_name == "Person":
                person_boxes.append(bbox)
            elif cls_name in POSITIVE_CLASSES or cls_name in VIOLATION_CLASSES:
                gear_detections.append((cls_name, bbox))

    pending_logs = []  # (track_id, item)

    with state_lock:
        current_area = active_area
        required_items = AREAS[current_area]["required"]

        matched_ids = match_persons(person_boxes, now, required_items)
        box_to_id = {person_tracks[tid]["bbox"]: tid for tid in matched_ids if tid in person_tracks}

        for cls_name, gbox in gear_detections:
            tid = assign_gear_to_person(gbox)
            if tid is None or tid not in person_tracks:
                continue
            status = person_tracks[tid]["ppe_status"]
            if cls_name in POSITIVE_CLASSES:
                status[cls_name] = "detected"
                person_tracks[tid]["saw_none"] = False
            elif cls_name == "none":
                for item in required_items:
                    status[item] = "missing"
                person_tracks[tid]["saw_none"] = True
            else:
                item = VIOLATION_TO_ITEM.get(cls_name)
                if item:
                    status[item] = "missing"

        aggregate_missing = set()
        for tid in matched_ids:
            track = person_tracks.get(tid)
            if not track:
                continue
            for item in required_items:
                if track["ppe_status"].get(item) == "missing":
                    aggregate_missing.add(item)

        # ---- One-shot capture per person: at most one photo/log per track ----
        for tid in matched_ids:
            track = person_tracks.get(tid)
            if not track or track["logged"]:
                continue
            if now - track["first_seen"] < CAPTURE_SETTLE_SECONDS:
                continue  # give detection a moment to stabilize after they appear

            if track.get("saw_none"):
                pending_logs.append((tid, "None Detected (No PPE Worn)", "High"))
                track["logged"] = True
                continue

            missing_items = [item for item in required_items if track["ppe_status"].get(item) == "missing"]
            if missing_items:
                violation_text = ", ".join(item.capitalize() for item in missing_items) + " Missing"
                severity = "High" if any(ITEM_META[i]["severity"] == "High" for i in missing_items) else "Medium"
                pending_logs.append((tid, violation_text, severity))
                track["logged"] = True
            # else: fully compliant so far — keep watching, don't mark logged yet

        persons_this_frame = len(person_boxes)

    # ---------- Draw annotations (no lock needed — frame is local) ----------
    for cls_name, conf, bbox in draw_items:
        x1, y1, x2, y2 = bbox
        color = CLASS_COLORS.get(cls_name, (255, 255, 255))
        if cls_name == "Person":
            tid = box_to_id.get(bbox)
            label = f"Person {tid:02d}" if tid else "Person"
        else:
            label = f"{cls_name} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # (on-frame "X MISSING" banner removed — status now lives only in the
    # dashboard PPE Status panel, not overlaid on the camera feed)

    # ---------- Save the one-time evidence photo for anyone just logged ----------
    new_log_entries = []
    for tid, violation_text, severity in pending_logs:
        photo_name = f"{current_area}_p{tid:02d}_{int(now)}.jpg"
        cv2.imwrite(os.path.join(PHOTO_DIR, photo_name), frame)
        photo_url = f"{PHOTO_URL_PREFIX}/{photo_name}"
        new_log_entries.append(make_log_entry(current_area, tid, violation_text, severity, photo_url))

    # ---------- Publish state ----------
    with state_lock:
        frame_history.append((now, persons_this_frame))
        while frame_history and now - frame_history[0][0] > ROLLING_WINDOW_SECONDS:
            frame_history.popleft()
        smoothed_persons = max(p for _, p in frame_history)

        state["current_persons"] = smoothed_persons
        state["total_persons_seen"] += persons_this_frame
        state["active_area"] = current_area
        state["active_area_name"] = AREAS[current_area]["name"]
        state["required_items"] = required_items
        if matched_ids:
            state["ppe_status"] = {
                item: ("missing" if item in aggregate_missing else "detected") for item in required_items
            }
        else:
            state["ppe_status"] = {item: "unknown" for item in required_items}

        for entry in new_log_entries:
            logs.appendleft(entry)
        state["total_logs"] = len(logs)

    return frame


def camera_loop():
    """Runs on the server only — opens the camera, processes frames, and shows
    a local preview window. The dashboard never receives this video; it only
    reads the stats/logs/photos this loop produces."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        with state_lock:
            state["camera_status"] = "offline"
        return

    with state_lock:
        state["camera_status"] = "live"

    window_name = "Site Guard - Server Preview (press q to close preview)"

    while True:
        ret, frame = cap.read()
        if not ret:
            with state_lock:
                state["camera_status"] = "offline"
            break

        frame = process_frame(frame)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    with state_lock:
        state["camera_status"] = "offline"


@app.route("/")
def index():
    with state_lock:
        current_area = active_area
    return render_template("index.html", camera_name=CAMERA_NAME, areas=AREAS, active_area=current_area)


@app.route("/api/stats")
def api_stats():
    with state_lock:
        return jsonify(dict(state))


@app.route("/api/logs")
def api_logs():
    with state_lock:
        return jsonify(list(logs))


@app.route("/api/area", methods=["POST"])
def set_area():
    global active_area
    data = request.get_json(force=True, silent=True) or {}
    area_id = data.get("area")
    if area_id not in AREAS:
        return jsonify({"error": "invalid area"}), 400

    with state_lock:
        active_area = area_id
        person_tracks.clear()  # IDs restart at 01 for the newly active area

    return jsonify({"active_area": area_id, "active_area_name": AREAS[area_id]["name"]})


if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)