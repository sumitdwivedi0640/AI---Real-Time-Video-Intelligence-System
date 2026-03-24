import cv2
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load models
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

# Restricted area (x1, y1, x2, y2)
restricted_area = (100, 100, 400, 400)

# Loitering tracking
loitering_time = {}
LOITER_THRESHOLD = 5  # seconds


def run_system():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot access camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for performance
        frame = cv2.resize(frame, (640, 480))

        # Run YOLO detection
        results = model(frame)[0]

        detections = []

        # Extract detections (only persons)
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = r

            if int(class_id) == 0:  # person class
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, class_id))

        # Tracking
        tracks = tracker.update_tracks(detections, frame=frame)

        # Draw restricted area
        cv2.rectangle(frame,
                      (restricted_area[0], restricted_area[1]),
                      (restricted_area[2], restricted_area[3]),
                      (0, 0, 255), 2)

        cv2.putText(frame, "Restricted Area",
                    (restricted_area[0], restricted_area[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        current_time = time.time()

        # Crowd detection
        person_count = len(tracks)
        if person_count > 3:
            cv2.putText(frame, "CROWD ALERT!",
                        (50, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Process each track
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, w, h = track.to_ltrb()
            l, t, w, h = int(l), int(t), int(w), int(h)

            # Center point
            cx = int(l + w / 2)
            cy = int(t + h / 2)

            # Initialize loitering timer
            if track_id not in loitering_time:
                loitering_time[track_id] = current_time

            time_spent = current_time - loitering_time[track_id]

            # Default color
            color = (0, 255, 0)

            # 🚨 Intrusion detection
            if (restricted_area[0] < cx < restricted_area[2] and
                    restricted_area[1] < cy < restricted_area[3]):

                cv2.putText(frame, "ALERT: INTRUSION!",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3)

                color = (0, 0, 255)

            # ⏱️ Loitering detection
            if time_spent > LOITER_THRESHOLD:
                cv2.putText(frame, f"LOITERING ALERT ID: {track_id}",
                            (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2)

                color = (0, 0, 255)

            # Draw bounding box
            cv2.rectangle(frame, (l, t), (l + w, t + h), color, 2)

            # ID + timer
            cv2.putText(frame,
                        f"ID: {track_id} ({int(time_spent)}s)",
                        (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2)

            # Draw center point
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        # Show frame
        cv2.imshow("AI Video Intelligence System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_system()
