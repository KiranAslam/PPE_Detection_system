import cv2
from ultralytics import YOLO

MODEL_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.4
CAMERA_INDEX = 0

CLASS_COLORS = {
    "helmet": (0, 255, 0),
    "gloves": (255, 0, 0),
    "vest": (0, 165, 255),
    "goggles": (255, 255, 0),
    "none": (128, 128, 128),
    "Person": (0, 255, 255),
    "no_helmet": (0, 0, 255),
    "no_goggle": (0, 0, 200),
    "no_gloves": (0, 0, 150),
    "no_vest": (0, 0, 100),
}

REQUIRED_PPE = ["helmet", "gloves", "vest", "goggles"]


def draw_detection(frame, x1, y1, x2, y2, label, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_missing_summary(frame, detected_classes):
    missing_items = [item for item in REQUIRED_PPE if item not in detected_classes]
    y_offset = 30
    for item in missing_items:
        text = f"{item.upper()} MISSING"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30


def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        detected_classes = set()

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detected_classes.add(cls_name)
                color = CLASS_COLORS.get(cls_name, (255, 255, 255))
                label = f"{cls_name} {conf:.2f}"

                draw_detection(frame, x1, y1, x2, y2, label, color)

        draw_missing_summary(frame, detected_classes)

        cv2.imshow("PPE Detection - Live", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()