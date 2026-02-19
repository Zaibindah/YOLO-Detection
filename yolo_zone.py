from ultralytics import YOLO
import cv2
import numpy as np

# -----------------------------
# VIDEO PATH
# -----------------------------
video_path = "videos\Walking_in_the_same_direction_timely.mp4"
output_path = "output_videos\output_tracked_video.mp4"

# -----------------------------
# LOAD YOLO MODEL
# -----------------------------
model = YOLO("yolo11n.pt")

# -----------------------------
# CREATE WINDOW (800x600)
# -----------------------------
cv2.namedWindow("YOLO Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Tracking", 800, 600)

# -----------------------------
# VIDEO CAPTURE (for FPS)
# -----------------------------
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

# -----------------------------
# VIDEO WRITER (MP4)
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (800, 600))

# -----------------------------
# GREEN ZONE
# -----------------------------
green_points = np.array([
    (247, 262), (227, 279), (211, 279), (117, 367),
    (143, 390), (4, 506), (6, 591), (192, 598),
    (318, 595), (326, 485), (334, 400), (341, 341),
    (335, 339), (339, 295), (342, 269), (258, 271)
], np.int32)

# -----------------------------
# RED ZONE
# -----------------------------
red_points = np.array([
    (394, 263), (395, 297), (398, 286), (516, 283),
    (624, 283), (735, 290), (794, 289), (793, 431),
    (795, 475), (794, 593), (627, 596), (412, 593),
    (410, 491), (404, 423), (399, 347)
], np.int32)

# -----------------------------
# YELLOW ZONE
# -----------------------------
yellow_points = np.array([
    (342, 268), (336, 321), (336, 401), (335, 505),
    (333, 598), (325, 599), (395, 598), (395, 426),
    (394, 326), (379, 281), (344, 263)
], np.int32)

# -----------------------------
# YOLO TRACKING
# -----------------------------
results = model.track(
    source=video_path,
    tracker="bytetrack.yaml",
    persist=True,
    stream=True
)

for r in results:

    original_frame = r.orig_img
    h_orig, w_orig = original_frame.shape[:2]

    frame = cv2.resize(original_frame, (800, 600), interpolation=cv2.INTER_LINEAR)

    scale_x = 800 / w_orig
    scale_y = 600 / h_orig

    if r.boxes.id is not None:
        for box, track_id, cls_id in zip(r.boxes.xyxy, r.boxes.id, r.boxes.cls):

            if int(cls_id) != 0:  # Only PERSON
                continue

            x1, y1, x2, y2 = map(int, box)

            x1 = int(x1 * scale_x)
            x2 = int(x2 * scale_x)
            y1 = int(y1 * scale_y)
            y2 = int(y2 * scale_y)

            tid = int(track_id)

            # Bottom center point
            cx = (x1 + x2) // 2
            cy = y2
            point = (cx, cy)

            dist_green = cv2.pointPolygonTest(green_points, point, False)
            dist_red = cv2.pointPolygonTest(red_points, point, False)
            dist_yellow = cv2.pointPolygonTest(yellow_points, point, False)

            # Zone classification (boundaries invisible)
            if dist_red >= 0:
                color = (0, 0, 255)
                label = f"ID {tid} - RESTRICTED ZONE"

            elif dist_yellow >= 0:
                color = (0, 255, 255)
                label = f"ID {tid} - WARNING ZONE"

            elif dist_green >= 0:
                color = (0, 255, 0)
                label = f"ID {tid} - SAFE ZONE"

            else:
                color = (0, 255, 0)
                label = f"ID {tid} - NORMAL"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2)

    cv2.imshow("YOLO Tracking", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

out.release()
cv2.destroyAllWindows()
