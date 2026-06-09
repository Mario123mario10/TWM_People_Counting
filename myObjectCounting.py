import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import argparse
import sys


class Track:

    def __init__(self, detection, track_id):
        self.id = track_id
        self.hits = 1
        self.time_since_update = 0
        self.side = None
        self.count_cooldown = 0

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
        self.time_since_update = 0

    def get_state(self):
        s = self.kf.statePost
        cx, cy, w, h = s[0,0], s[1,0], s[2,0], s[3,0]
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])



class ObjectTracking:

    def __init__(self, model="yolo26n.pt", source=None,
                 conf_thres=0.3, iou_match_thres=0.3,
                 max_age=30, min_hits=4,
                 line_start=None, line_end=None,
                 target_class=0, imgsz=None,
                 output="object-tracking.avi", display=True,
                 device=None, augment=False,
                 nms_iou=None, max_det=None,
                 line_margin=0.0, count_cooldown=0):
        self.model = YOLO(model)
        self.names = self.model.names
        self.target_class = target_class
        self.imgsz = imgsz
        self.output = output
        self.display = display
        self.device = device
        self.augment = augment
        self.nms_iou = nms_iou
        self.max_det = max_det
        self.line_margin = line_margin
        self.count_cooldown = count_cooldown

        self.cap = cv2.VideoCapture(source if source else 0)
        assert self.cap.isOpened(), "Error reading video file"

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
        self.writer = cv2.VideoWriter(
            self.output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (w, h)
        )

        self.conf_thres = conf_thres
        self.iou_match_thres = iou_match_thres
        self.max_age = max_age
        self.min_hits = min_hits
        self.yolo_kwargs = {
            "conf": self.conf_thres,
            "classes": [self.target_class],
            "verbose": False,
        }
        if self.imgsz is not None:
            self.yolo_kwargs["imgsz"] = self.imgsz
        if self.device is not None:
            self.yolo_kwargs["device"] = self.device
        if self.augment:
            self.yolo_kwargs["augment"] = True
        if self.nms_iou is not None:
            self.yolo_kwargs["iou"] = self.nms_iou
        if self.max_det is not None:
            self.yolo_kwargs["max_det"] = self.max_det

        self.tracks = []
        self.next_id = 0
        self.track_history = defaultdict(lambda: [])

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
        if self.display:
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

    def update_side_and_count(self, track, center):
        line_vec = np.array([self.line_end[0] - self.line_start[0],
                            self.line_end[1] - self.line_start[1]])
        pt_vec = np.array([center[0] - self.line_start[0],
                        center[1] - self.line_start[1]])
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return

        cross = np.cross(line_vec, pt_vec)
        signed_distance = cross / line_len
        if abs(signed_distance) < self.line_margin:
            return

        current_side = 1 if cross > 0 else -1
        if track.count_cooldown > 0:
            track.side = current_side
            return

        if track.side is None:
            track.side = current_side
        elif track.side != current_side:
            if track.side == -1 and current_side == 1:
                self.in_count += 1
            else:
                self.out_count += 1
            track.side = current_side
            track.count_cooldown = self.count_cooldown

    def run(self):
        while self.cap.isOpened():
            success, im0 = self.cap.read()
            if not success:
                print("End of video or failed to read image.")
                break

            results = self.model(im0, **self.yolo_kwargs)
            detections = []
            classes = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    clss = result.boxes.cls.cpu().numpy()
                    for box, conf, cls in zip(boxes, confs, clss):
                        if int(cls) == self.target_class:
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
                if track.count_cooldown > 0:
                    track.count_cooldown -= 1
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
                track_detection_pairs.append((track.id, detections[det_idx], track.cls))

            # --- Nowe ścieżki z nieprzypisanych detekcji ---
            for det_idx in unmatched_detections:
                new_track = Track(detections[det_idx], self.next_id)
                new_track.cls = classes[det_idx]
                self.tracks.append(new_track)
                track_detection_pairs.append((self.next_id, detections[det_idx], classes[det_idx]))
                self.next_id += 1

            # --- Usuwanie starych ścieżek ---
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

            # --- RYSOWANIE ---
            cv2.line(im0, self.line_start, self.line_end, (0, 255, 255), 2)
            cv2.putText(im0, f"IN: {self.in_count}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(im0, f"OUT: {self.out_count}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)


            for track_id, bbox, cls in track_detection_pairs:
                self.draw_bbox(im0, bbox, track_id, cls)

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

            if self.display:
                cv2.imshow(self.window_name, im0)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    print("Selection cleared")

        self.cap.release()
        self.writer.release()
        if self.display:
            cv2.destroyAllWindows()
        print(f"Final counts: IN={self.in_count}, OUT={self.out_count}")

def main(argv):
    parser = argparse.ArgumentParser(description="YOLO person tracking and IN/OUT counting.")
    parser.add_argument('--model', type=str, required=False, help="YOLO model to use, defaults to yolo26s.pt", default="yolo26s.pt")
    parser.add_argument('--source', type=str, required=False, help="source video file, runs with default camera input if not provided", default=None)
    parser.add_argument('--output', type=str, required=False, help="output video path", default="object-tracking.avi")
    parser.add_argument('--conf', type=float, required=False, help="YOLO confidence threshold", default=0.5)
    parser.add_argument('--imgsz', type=int, required=False, help="YOLO inference image size, e.g. 640 or 960", default=None)
    parser.add_argument('--iou', type=float, required=False, help="IOU threshold for matching detections to tracks", default=0.3)
    parser.add_argument('--max-age', type=int, required=False, help="frames to keep a track without detection", default=30)
    parser.add_argument('--min-hits', type=int, required=False, help="detections needed before a track is considered valid", default=3)
    parser.add_argument('--line', type=int, nargs=4, metavar=('X1', 'Y1', 'X2', 'Y2'),
                        help="counting line coordinates, e.g. --line 0 120 320 120", default=None)
    parser.add_argument('--target-class', type=int, required=False, help="YOLO class id to count, defaults to 0/person", default=0)
    parser.add_argument('--device', type=str, required=False, help="inference device, e.g. cpu or cuda:0", default=None)
    parser.add_argument('--augment', action='store_true', help="enable YOLO test-time augmentation, slower but sometimes more robust")
    parser.add_argument('--nms-iou', type=float, required=False, help="YOLO NMS IOU threshold, separate from tracker --iou", default=None)
    parser.add_argument('--max-det', type=int, required=False, help="maximum YOLO detections per frame", default=None)
    parser.add_argument('--line-margin', type=float, required=False,
                        help="dead zone around counting line in pixels; crossings inside it are ignored", default=0.0)
    parser.add_argument('--count-cooldown', type=int, required=False,
                        help="frames to wait before the same track can be counted again", default=0)
    parser.add_argument('--no-display', action='store_true', help="run without showing an OpenCV window")
    args = parser.parse_args(argv[1:])

    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1")
    if not 0 <= args.iou <= 1:
        parser.error("--iou must be between 0 and 1")
    if args.nms_iou is not None and not 0 <= args.nms_iou <= 1:
        parser.error("--nms-iou must be between 0 and 1")
    if args.max_age < 0:
        parser.error("--max-age must be 0 or greater")
    if args.min_hits < 1:
        parser.error("--min-hits must be 1 or greater")
    if args.max_det is not None and args.max_det < 1:
        parser.error("--max-det must be 1 or greater")
    if args.line_margin < 0:
        parser.error("--line-margin must be 0 or greater")
    if args.count_cooldown < 0:
        parser.error("--count-cooldown must be 0 or greater")

    line_start = None
    line_end = None
    if args.line is not None:
        line_start = (args.line[0], args.line[1])
        line_end = (args.line[2], args.line[3])

    tracker = ObjectTracking(
        model=args.model,
        source=args.source,
        conf_thres=args.conf,
        iou_match_thres=args.iou,
        max_age=args.max_age,
        min_hits=args.min_hits,
        line_start=line_start,
        line_end=line_end,
        target_class=args.target_class,
        imgsz=args.imgsz,
        output=args.output,
        display=not args.no_display,
        device=args.device,
        augment=args.augment,
        nms_iou=args.nms_iou,
        max_det=args.max_det,
        line_margin=args.line_margin,
        count_cooldown=args.count_cooldown
    )
    tracker.run()

if __name__ == "__main__":
    main(sys.argv)
