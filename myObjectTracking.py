import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


class Track:
    """Pojedynczy śledzony obiekt z filtrem Kalmana."""

    def __init__(self, detection, track_id):
        """
        detection: [x1, y1, x2, y2]
        """
        self.id = track_id
        self.hits = 1               # liczba kolejnych dopasowań
        self.age = 0                # liczba klatek od ostatniego dopasowania
        self.time_since_update = 0

        # Konwersja na stan [cx, cy, w, h]
        x1, y1, x2, y2 = detection
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0

        # Kalman: 8 stanów [cx, cy, w, h, vx, vy, vw, vh], 4 pomiary [cx, cy, w, h]
        self.kf = cv2.KalmanFilter(8, 4)

        # Macierz przejścia (stała prędkość)
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

        # Macierz pomiarowa
        self.kf.measurementMatrix = np.array([
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0]
        ], np.float32)

        # Szum procesu i pomiaru
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 10.0

        # Stan początkowy
        self.kf.statePost = np.array([[cx], [cy], [w], [h], [0], [0], [0], [0]], np.float32)

    def predict(self):
        """Przewidywanie stanu."""
        pred = self.kf.predict()
        # pred to (8,1) wektor
        cx, cy, w, h = pred[0,0], pred[1,0], pred[2,0], pred[3,0]
        # Zwróć bounding box jako [x1, y1, x2, y2]
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        self.age += 1
        self.time_since_update += 1
        return np.array([x1, y1, x2, y2])

    def update(self, detection):
        """Korekcja filtru nową detekcją."""
        x1, y1, x2, y2 = detection
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        measurement = np.array([[cx], [cy], [w], [h]], np.float32)
        self.kf.correct(measurement)
        self.hits += 1
        self.age = 0
        self.time_since_update = 0

    def get_state(self):
        """Zwraca aktualny bounding box (po korekcji) jako [x1,y1,x2,y2]."""
        s = self.kf.statePost
        cx, cy, w, h = s[0,0], s[1,0], s[2,0], s[3,0]
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])


class ObjectTracking:
    """Object Tracking – własna implementacja z Kalmanem, Węgierskim i IOU."""

    def __init__(self, model="yolo26n.pt", source="path/to/video.mp4",
                 conf_thres=0.5, iou_match_thres=0.3,
                 max_age=30, min_hits=3):
        self.model = YOLO(model)
        self.names = self.model.names

        self.cap = cv2.VideoCapture(source)
        assert self.cap.isOpened(), "Error reading video file"

        w, h, fps = (
            int(self.cap.get(x))
            for x in (cv2.CAP_PROP_FRAME_WIDTH,
                      cv2.CAP_PROP_FRAME_HEIGHT,
                      cv2.CAP_PROP_FPS)
        )
        self.writer = cv2.VideoWriter(
            "object-tracking.avi",
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (w, h)
        )

        # Parametry trackera
        self.conf_thres = conf_thres
        self.iou_match_thres = iou_match_thres
        self.max_age = max_age
        self.min_hits = min_hits

        # Lista aktywnych ścieżek i generator ID
        self.tracks = []
        self.next_id = 0

        # Historia ścieżek do rysowania linii
        self.track_history = defaultdict(lambda: [])

        # Ustawienia wyświetlania
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
        """
        Oblicza macierz IOU między dwoma zestawami bounding boxów.
        bboxes1: (N,4), bboxes2: (M,4) – format [x1,y1,x2,y2]
        """
        bboxes1 = np.expand_dims(bboxes1, 1)  # (N,1,4)
        bboxes2 = np.expand_dims(bboxes2, 0)  # (1,M,4)

        # Współrzędne przecięcia
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
        """Rysowanie bounding boxa z etykietą (oryginalna funkcja)."""
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

    def run(self):
        """Główna pętla śledzenia."""
        while self.cap.isOpened():
            success, im0 = self.cap.read()
            if not success:
                print("End of video or failed to read image.")
                break

            # --- Detekcja YOLO (bez wbudowanego trackingu) ---
            results = self.model(im0, conf=self.conf_thres, verbose=False)
            detections = []    # bounding boxy [x1,y1,x2,y2]
            classes = []       # klasy
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    clss = result.boxes.cls.cpu().numpy()
                    # Filtrowanie po pewności – już zrobione przez conf=self.conf_thres
                    for box, conf, cls in zip(boxes, confs, clss):
                        detections.append(box)
                        classes.append(cls)

            # --- Predykcja filtrów Kalmana dla aktywnych ścieżek ---
            predicted_bboxes = []
            for track in self.tracks:
                pred = track.predict()
                predicted_bboxes.append(pred)

            # --- Skojarzenie detekcji ze ścieżkami (algorytm węgierski + IOU) ---
            matches = []
            unmatched_detections = list(range(len(detections)))
            unmatched_tracks = list(range(len(self.tracks)))

            if len(predicted_bboxes) > 0 and len(detections) > 0:
                iou_matrix = self.iou_batch(np.array(predicted_bboxes), np.array(detections))
                # Koszt = 1 - IOU (minimalizacja)
                cost = 1.0 - iou_matrix
                # Ustawiamy koszt na duży dla par poniżej progu IOU
                cost[iou_matrix < self.iou_match_thres] = 1e5

                row_ind, col_ind = linear_sum_assignment(cost)

                # Tylko pary z IOU powyżej progu
                for r, c in zip(row_ind, col_ind):
                    if iou_matrix[r, c] >= self.iou_match_thres:
                        matches.append((r, c))
                        unmatched_detections.remove(c)
                        unmatched_tracks.remove(r)

            # --- Aktualizacja dopasowanych ścieżek ---
            for track_idx, det_idx in matches:
                self.tracks[track_idx].update(detections[det_idx])
                # Zapisujemy klasę – potrzebna do rysowania; można przechować w tracku
                # Prosty sposób: zapamiętamy klasę w tracku jako atrybut
                # Jeśli track nie ma jeszcze klasy, przypisujemy ją teraz
                if not hasattr(self.tracks[track_idx], 'cls'):
                    self.tracks[track_idx].cls = classes[det_idx]
                else:
                    # Opcjonalna aktualizacja klasy (np. głosowanie)
                    self.tracks[track_idx].cls = classes[det_idx]

            # --- Nowe ścieżki dla niedopasowanych detekcji ---
            for det_idx in unmatched_detections:
                new_track = Track(detections[det_idx], self.next_id)
                new_track.cls = classes[det_idx]  # zapamiętaj klasę
                self.tracks.append(new_track)
                self.next_id += 1

            # --- Usuwanie starych ścieżek (max_age) ---
            # Najpierw zwiększ age dla nieuaktualnionych
            # for track_idx in unmatched_tracks:
            #     # age został już zwiększony w predict()
            #     pass

            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

            # --- Rysowanie tylko potwierdzonych (min_hits) ścieżek ---
            for track in self.tracks:
                if track.hits >= self.min_hits:
                    bbox = track.get_state()
                    cls = track.cls if hasattr(track, 'cls') else 0
                    self.draw_bbox(im0, bbox, track.id, cls)

                    # Historia do rysowania linii
                    cx = (bbox[0] + bbox[2]) / 2.0
                    cy = (bbox[1] + bbox[3]) / 2.0
                    self.track_history[track.id].append((cx, cy))
                    if len(self.track_history[track.id]) > 50:
                        self.track_history[track.id].pop(0)

                    # Rysowanie śladu i kółka
                    points = np.array(self.track_history[track.id], dtype=np.float32).reshape((-1, 1, 2))
                    if len(points) > 1:
                        cv2.polylines(im0, [points.astype(np.int32)], False,
                                      colors(int(cls), True), self.polyline_thickness)
                    cv2.circle(im0, (int(cx), int(cy)), 5, colors(int(cls), True), -1)

            self.writer.write(im0)
            cv2.imshow(self.window_name, im0)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Czyszczenie zaznaczenia (oryginalna funkcja, można rozbudować)
                print("Selection cleared")

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = ObjectTracking(
        model="yolo26s.pt",
        source="./output.mp4",
        conf_thres=0.5,
        iou_match_thres=0.2,
        max_age=30,
        min_hits=3
    )
    tracker.run()