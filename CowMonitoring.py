import cv2
from ultralytics import YOLO


# -----------------------------
# Track Class with disappearance counter
# -----------------------------
class Track:
    def __init__(self, track_id, box, cls):
        self.id = track_id
        self.box = box
        self.cls = cls
        self.centers = []
        self.cross_state = None
        self.missed_frames = 0  # NEW: for removing stale tracks


class Tracker:
    def __init__(self, iou_thresh=0.3, max_missed=10):
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self.tracks = {}
        self.next_id = 0

    @staticmethod
    def iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter <= 0:
            return 0

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union

    # -----------------------------------------
    # Updated Stable Tracker
    # -----------------------------------------
    def update(self, detections):
        matched_ids = set()

        # Match existing tracks to detections
        for det_box, det_cls in detections:
            best_iou = 0
            best_id = None

            for tid, tr in self.tracks.items():
                iou_val = self.iou(det_box, tr.box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_id = tid

            if best_iou > self.iou_thresh:
                track = self.tracks[best_id]
                track.box = det_box
                track.centers.append(((det_box[0] + det_box[2]) // 2,
                                      (det_box[1] + det_box[3]) // 2))
                track.missed_frames = 0
                matched_ids.add(best_id)
            else:
                track = Track(self.next_id, det_box, det_cls)
                track.centers.append(((det_box[0] + det_box[2]) // 2,
                                      (det_box[1] + det_box[3]) // 2))
                self.tracks[self.next_id] = track
                matched_ids.add(self.next_id)
                self.next_id += 1

        # Increase missed frames for unmatched tracks
        for tid, tr in list(self.tracks.items()):
            if tid not in matched_ids:
                tr.missed_frames += 1

        # Remove tracks not seen for a while
        self.tracks = {tid: tr for tid, tr in self.tracks.items()
                       if tr.missed_frames <= self.max_missed}

        return self.tracks


# -----------------------------
# Main Counting System
# -----------------------------
def main():
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    tracker = Tracker()

    human_count = 0
    cow_count = 0

    LINE_X = 400

    print("Running... press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        results = model(frame, conf=0.5, verbose=False)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id].lower()

            if cls_name not in ["person", "cow"]:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append(([x1, y1, x2, y2], cls_name))

        tracks = tracker.update(detections)

        # Draw the vertical line
        cv2.line(frame, (LINE_X, 0), (LINE_X, h), (0, 0, 255), 3)

        for tid, tr in tracks.items():

            # Skip tracks without enough data
            if len(tr.centers) == 0:
                continue

            box = tr.box
            x1, y1, x2, y2 = map(int, box)
            cx, cy = tr.centers[-1]

            current_side = "left" if cx < LINE_X else "right"

            # Initialize state
            if tr.cross_state is None:
                tr.cross_state = current_side

            else:
                # CROSS LEFT → RIGHT
                if tr.cross_state == "left" and current_side == "right":
                    if tr.cls == "person":
                        human_count += 1
                    else:
                        cow_count += 1
                    tr.cross_state = current_side

                # CROSS RIGHT → LEFT
                elif tr.cross_state == "right" and current_side == "left":
                    if tr.cls == "person":
                        human_count -= 1
                    else:
                        cow_count -= 1
                    tr.cross_state = current_side

            # Draw box + ID
            color = (0, 255, 0) if tr.cls == "person" else (255, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{tr.cls} ID {tid}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display counters
        cv2.putText(frame, f"Humans: {human_count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"Cows: {cow_count}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Human & Cow Counter (Clean & Working)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
