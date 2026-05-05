import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import argparse
import sys


class Track:
    """Pojedynczy śledzony obiekt z filtrem Kalmana."""

    def __init__(self, detection, track_id):
        self.id = track_id
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.side = None

        x1, y1, x2, y2 = detection
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0

        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.array([
            [1,0,0,0,1,0,0,0],
            [0,1,0,0,0,1,0,0],
            [0,0,1,0,0,0,1,0],
            [0,0,0,1,0,0,0,1],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1]
        ], np.float32)
        self.kf.measurementMatrix = np.array([
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0]
        ], np.float32)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 10.0
        self.kf.statePost = np.array([[cx], [cy], [w], [h], [0], [0], [0], [0]], np.float32)

    def predict(self):
        pred = self.kf.predict()
        cx, cy, w, h = pred[0,0], pred[1,0], pred[2,0], pred[3,0]
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        self.age += 1
        self.time_since_update += 1
        return np.array([x1, y1, x2, y2])

    def update(self, detection):
        x1, y1, x2, y2 = detection
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w/2.0
        cy = y1 + h/2.0
        measurement = np.array([[cx], [cy], [w], [h]], np.float32)
        self.kf.correct(measurement)
        self.hits += 1
        self.age = 0
        self.time_since_update = 0

    def get_state(self):
        """Zwraca wygładzony bounding box (używane tylko do linii i historii)."""
        s = self.kf.statePost
        cx, cy, w, h = s[0,0], s[1,0], s[2,0], s[3,0]
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
    


class ObjectTracking:
    """Object Tracking + zliczanie linii – wyświetla detekcje YOLO, tylko osoby."""

    def __init__(self, model="yolo26n.pt", source="path/to/video.mp4",
                 conf_thres=0.5, iou_match_thres=0.3,
                 max_age=30, min_hits=3,
                 line_start=None, line_end=None,
                 target_class=0):                    # NOWE: domyślnie klasa 0 = person
        self.model = YOLO(model)
        self.names = self.model.names
        self.target_class = target_class              # filtrowana klasa

        self.cap = cv2.VideoCapture(source)
        assert self.cap.isOpened(), "Error reading video file"

        w, h, fps = (int(self.cap.get(x)) for x in
                     (cv2.CAP_PROP_FRAME_WIDTH,
                      cv2.CAP_PROP_FRAME_HEIGHT,
                      cv2.CAP_PROP_FPS))
        self.writer = cv2.VideoWriter(
            "object-tracking.avi",
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (w, h)
        )

        self.conf_thres = conf_thres
        self.iou_match_thres = iou_match_thres
        self.max_age = max_age
        self.min_hits = min_hits

        self.tracks = []
        self.next_id = 0
        self.track_history = defaultdict(lambda: [])

        # Linia wirtualna
        if line_start is None or line_end is None:
            self.line_start = (0, h // 2)
            self.line_end = (w, h // 2)
        else:
            self.line_start = line_start
            self.line_end = line_end

        self.in_count = 0
        self.out_count = 0
        self.track_side = {}

        self.rect_width = 2
        self.font = 1.0
        self.text_width = 2
        self.padding = 12
        self.margin = 10
        self.circle_thickness = 5
        self.polyline_thickness = 2
        self.window_name = "YOLO Tracking"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def iou_batch(self, bboxes1, bboxes2):
        bboxes1 = np.expand_dims(bboxes1, 1)
        bboxes2 = np.expand_dims(bboxes2, 0)
        xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
        yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
        xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
        yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        inter = w * h
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        union = area1 + area2 - inter
        return inter / np.maximum(union, 1e-6)

    def draw_bbox(self, im0, box, track_id, cls):
        """Rysuje prostokąt z etykietą (bez zmian)."""
        x1, y1, x2, y2 = map(int, box)
        color = colors(int(cls), True)
        cv2.rectangle(im0, (x1, y1), (x2, y2), color, self.rect_width)

        label = f"{self.names[int(cls)]}:{int(track_id)}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font, self.text_width)
        bg_x1 = x1
        bg_x2 = bg_x1 + (tw + 2 * self.padding)
        bg_y2 = y1
        bg_y1 = bg_y2 - (th + 2 * self.margin)
        cv2.rectangle(im0, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        text_x = bg_x1 + ((bg_x2 - bg_x1) - tw) // 2
        text_y = bg_y1 + ((bg_y2 - bg_y1) + th) // 2 - 2
        cv2.putText(im0, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    self.font, (104, 31, 17) if cls == 2 else (255, 255, 255),
                    self.text_width, cv2.LINE_AA)

    def check_line_crossing(self, track_id, prev_center, curr_center):
        """Zwraca 1 (IN), -1 (OUT), 0 (brak). Kalman centers."""
        line_vec = np.array([self.line_end[0] - self.line_start[0],
                             self.line_end[1] - self.line_start[1]])
        prev_vec = np.array([prev_center[0] - self.line_start[0],
                             prev_center[1] - self.line_start[1]])
        curr_vec = np.array([curr_center[0] - self.line_start[0],
                             curr_center[1] - self.line_start[1]])
        prev_cross = np.cross(line_vec, prev_vec)
        curr_cross = np.cross(line_vec, curr_vec)
        prev_side = 1 if prev_cross > 0 else -1
        curr_side = 1 if curr_cross > 0 else -1
        if prev_side != curr_side:
            if prev_side == -1 and curr_side == 1:
                self.in_count += 1
                return 1
            else:
                self.out_count += 1
                return -1
        return 0
    
    def update_side_and_count(self, track, center):
        """
        Aktualizuje stronę linii dla śledzonego obiektu i zlicza przekroczenia.
        center: (cx, cy) – środek obiektu (np. ze stanu Kalmana)
        """
        # Wektor linii
        line_vec = np.array([self.line_end[0] - self.line_start[0],
                            self.line_end[1] - self.line_start[1]])
        # Wektor od początku linii do punktu
        pt_vec = np.array([center[0] - self.line_start[0],
                        center[1] - self.line_start[1]])
        # Obliczamy stronę: dodatni Z -> lewa strona (umownie)
        cross = np.cross(line_vec, pt_vec)
        current_side = 1 if cross > 0 else -1

        if track.side is None:
            # Pierwsza rejestracja – tylko zapisz stronę, nie zliczaj
            track.side = current_side
        elif track.side != current_side:
            # Wykryto zmianę strony
            if track.side == -1 and current_side == 1:
                self.in_count += 1
            else:
                self.out_count += 1
            # Aktualizujemy zapamiętaną stronę
            track.side = current_side

    def run(self):
        """Główna pętla."""
        while self.cap.isOpened():
            success, im0 = self.cap.read()
            if not success:
                print("End of video or failed to read image.")
                break

            # --- Detekcja YOLO ---
            results = self.model(im0, conf=self.conf_thres, verbose=False)
            detections = []   # tylko osoby
            classes = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    clss = result.boxes.cls.cpu().numpy()
                    for box, conf, cls in zip(boxes, confs, clss):
                        if int(cls) == self.target_class:   # filtr klasy
                            detections.append(box)
                            classes.append(cls)

            # --- Predykcje Kalmana (potrzebne do skojarzenia i historii) ---
            predicted_bboxes = []
            prev_centers = []
            for track in self.tracks:
                prev_bbox = track.get_state()
                prev_centers.append(((prev_bbox[0]+prev_bbox[2])/2.0,
                                     (prev_bbox[1]+prev_bbox[3])/2.0))
                pred = track.predict()
                predicted_bboxes.append(pred)

            # --- Skojarzenie węgierskie + IOU ---
            matches = []
            unmatched_detections = list(range(len(detections)))
            unmatched_tracks = list(range(len(self.tracks)))

            if len(predicted_bboxes) > 0 and len(detections) > 0:
                iou_matrix = self.iou_batch(np.array(predicted_bboxes), np.array(detections))
                cost = 1.0 - iou_matrix
                cost[iou_matrix < self.iou_match_thres] = 1e5
                row_ind, col_ind = linear_sum_assignment(cost)
                for r, c in zip(row_ind, col_ind):
                    if iou_matrix[r, c] >= self.iou_match_thres:
                        matches.append((r, c))
                        unmatched_detections.remove(c)
                        unmatched_tracks.remove(r)

            # --- Aktualizacja ścieżek + zapamiętanie par (track, detection_box) ---
            track_detection_pairs = []    # (track_id, detection_box, cls)
            for track_idx, det_idx in matches:
                track = self.tracks[track_idx]
                track.update(detections[det_idx])
                if not hasattr(track, 'cls'):
                    track.cls = classes[det_idx]
                else:
                    track.cls = classes[det_idx]
                # Zapamiętaj do rysowania detekcji YOLO
                track_detection_pairs.append((track.id, detections[det_idx], track.cls))

            # --- Nowe ścieżki z nieprzypisanych detekcji ---
            for det_idx in unmatched_detections:
                new_track = Track(detections[det_idx], self.next_id)
                new_track.cls = classes[det_idx]
                self.tracks.append(new_track)
                track_detection_pairs.append((self.next_id, detections[det_idx], classes[det_idx]))
                self.next_id += 1

            # --- Sprawdzenie przekroczeń linii (używamy wygładzonych środków Kalmana) ---
            for track_idx, det_idx in matches:
                track = self.tracks[track_idx]
                if track.hits >= self.min_hits:
                    prev_center = prev_centers[track_idx]
                    curr_center = ((track.get_state()[0] + track.get_state()[2]) / 2.0,
                                   (track.get_state()[1] + track.get_state()[3]) / 2.0)
                    # self.check_line_crossing(track.id, prev_center, curr_center)

            # --- Usuwanie starych ścieżek ---
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

            # --- RYSOWANIE ---
            # Linia i liczniki
            cv2.line(im0, self.line_start, self.line_end, (0, 255, 255), 2)
            cv2.putText(im0, f"IN: {self.in_count}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(im0, f"OUT: {self.out_count}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # 1. Rysowanie bounding boxów – tylko detekcje YOLO
            for track_id, bbox, cls in track_detection_pairs:
                # Możemy chcieć rysować tylko te, które mają odpowiednią liczbę potwierdzeń?
                # Aby zachować spójność, rysujemy wszystkie detekcje, które są śledzone.
                self.draw_bbox(im0, bbox, track_id, cls)

            # 2. Aktualizacja historii i rysowanie śladów (centra z wygładzonego stanu)
            for track in self.tracks:
                if track.hits >= self.min_hits:
                    bbox_kalman = track.get_state()
                    cx = (bbox_kalman[0] + bbox_kalman[2]) / 2.0
                    cy = (bbox_kalman[1] + bbox_kalman[3]) / 2.0
                    self.track_history[track.id].append((cx, cy))

                    self.update_side_and_count(track, (cx, cy))

                    if len(self.track_history[track.id]) > 50:
                        self.track_history[track.id].pop(0)

                    points = np.array(self.track_history[track.id], dtype=np.float32).reshape((-1, 1, 2))
                    if len(points) > 1:
                        cv2.polylines(im0, [points.astype(np.int32)], False,
                                      colors(int(track.cls) if hasattr(track, 'cls') else 0, True),
                                      self.polyline_thickness)
                    cv2.circle(im0, (int(cx), int(cy)), 5,
                               colors(int(track.cls) if hasattr(track, 'cls') else 0, True), -1)

            self.writer.write(im0)
            cv2.imshow(self.window_name, im0)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                print("Selection cleared")

        self.cap.release()
        cv2.destroyAllWindows()

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=False, help="YOLO model to use, defaults to yolo26s.pt", default="yolo26s.pt")
    parser.add_argument('source', type=str, help="source video file")
    args = parser.parse_args(argv[1:])
    tracker = ObjectTracking(
        model=args.model,
        source=args.source,
        target_class=0,  # 0 = person; możesz zmienić na inną klasę
        # line_start=(200,300),
        # line_end=(800,300)
    )
    tracker.run()

if __name__ == "__main__":
    main(sys.argv)