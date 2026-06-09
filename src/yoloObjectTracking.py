import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors
from collections import defaultdict
import argparse
import sys


class ObjectTracking:

    def __init__(self, model="yolo26n.pt", source: str = None,
                 conf_thres=0.3, iou_match_thres=0.3,
                 max_age=30, min_hits=4,
                 line_start=None, line_end=None,
                 target_class=0, imgsz=None,
                 output="object-tracking.avi", display=True,
                 device=None, augment=False,
                 nms_iou=None, max_det=None,
                 line_margin=0.0, count_cooldown=0,
                 tracker="bytetrack.yaml"):
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
        self.tracker = tracker

        self.cap = cv2.VideoCapture((int(source) if source.isnumeric() else source) if source else 0)
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
        self.iou_match_thres = iou_match_thres      # ignored with ByteTrack
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

        # Tracking state
        self.track_info = {}
        self.track_history = defaultdict(list)

        if line_start is None or line_end is None:
            self.line_start = (0, h // 2)
            self.line_end = (w, h // 2)
        else:
            self.line_start = line_start
            self.line_end = line_end

        self.in_count = 0
        self.out_count = 0

        self.rect_width = 2
        self.font = 1.0
        self.text_width = 2
        self.padding = 12
        self.margin = 10
        self.circle_thickness = 5
        self.polyline_thickness = 2
        self.window_name = "YOLO Tracking + ByteTrack"
        if self.display:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def draw_bbox(self, im0, box, track_id, cls):
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

    def update_side_and_count(self, track_data, center):
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
        if track_data["count_cooldown"] > 0:
            track_data["side"] = current_side
            return

        if track_data["side"] is None:
            track_data["side"] = current_side
        elif track_data["side"] != current_side:
            if track_data["side"] == -1 and current_side == 1:
                self.in_count += 1
            else:
                self.out_count += 1
            track_data["side"] = current_side
            track_data["count_cooldown"] = self.count_cooldown

    def run(self):
        while self.cap.isOpened():
            success, im0 = self.cap.read()
            if not success:
                print("End of video or failed to read image.")
                break

            results = self.model.track(im0, persist=True, tracker=self.tracker, **self.yolo_kwargs)

            # Gather detections with track IDs
            active_detections = {}   # track_id -> (bbox, cls)
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                    clss = result.boxes.cls.cpu().numpy()
                    for box, tid, cls in zip(boxes, track_ids, clss):
                        active_detections[int(tid)] = (box, cls)

            # Update track_info
            for tid in list(self.track_info.keys()):
                if tid not in active_detections:
                    self.track_info[tid]["time_since_update"] += 1

            for tid, (box, cls) in active_detections.items():
                info = self.track_info.get(tid)
                if info is None:
                    info = {
                        "hits": 1,
                        "time_since_update": 0,
                        "side": None,
                        "count_cooldown": 0,
                        "cls": cls,
                    }
                    self.track_info[tid] = info
                else:
                    info["hits"] += 1
                    info["time_since_update"] = 0
                    info["cls"] = cls

            # Remove tracks that exceed max_age
            expired = [tid for tid, info in self.track_info.items() if info["time_since_update"] > self.max_age]
            for tid in expired:
                del self.track_info[tid]
                if tid in self.track_history:
                    del self.track_history[tid]

            # Drawing
            cv2.line(im0, self.line_start, self.line_end, (0, 255, 255), 2)
            cv2.putText(im0, f"IN: {self.in_count}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(im0, f"OUT: {self.out_count}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            for tid, info in self.track_info.items():
                if info["hits"] < self.min_hits:
                    continue
                if tid not in active_detections:
                    continue   # no bbox to draw this frame

                box, cls = active_detections[tid]
                self.draw_bbox(im0, box, tid, cls)

                # Update history and crossing
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                self.track_history[tid].append((cx, cy))
                if len(self.track_history[tid]) > 50:
                    self.track_history[tid].pop(0)

                self.update_side_and_count(info, (cx, cy))

                # Draw trail
                points = np.array(self.track_history[tid], dtype=np.float32).reshape((-1, 1, 2))
                if len(points) > 1:
                    cv2.polylines(im0, [points.astype(np.int32)], False,
                                  colors(int(cls), True), self.polyline_thickness)
                cv2.circle(im0, (int(cx), int(cy)), 5,
                           colors(int(cls), True), -1)

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
    parser = argparse.ArgumentParser(description="YOLO tracking and IN/OUT counting with ByteTrack.")
    parser.add_argument('--model', type=str, required=False, help="YOLO model to use", default="yolo26s.pt")
    parser.add_argument('--source', type=str, required=False, help="source video file or camera index", default=None)
    parser.add_argument('--output', type=str, required=False, help="output video path", default="object-tracking.avi")
    parser.add_argument('--conf', type=float, required=False, help="YOLO confidence threshold", default=0.5)
    parser.add_argument('--imgsz', type=int, required=False, help="YOLO inference image size", default=None)
    parser.add_argument('--iou', type=float, required=False, help="NMS IoU threshold (not used for tracker)", default=0.3)
    parser.add_argument('--max-age', type=int, required=False, help="frames to keep a track without detection", default=30)
    parser.add_argument('--min-hits', type=int, required=False, help="detections needed before track is considered valid", default=3)
    parser.add_argument('--line', type=int, nargs=4, metavar=('X1', 'Y1', 'X2', 'Y2'),
                        help="counting line coordinates", default=None)
    parser.add_argument('--target-class', type=int, required=False, help="YOLO class id to count", default=0)
    parser.add_argument('--device', type=str, required=False, help="inference device", default=None)
    parser.add_argument('--augment', action='store_true', help="enable test-time augmentation")
    parser.add_argument('--nms-iou', type=float, required=False, help="YOLO NMS IoU threshold", default=None)
    parser.add_argument('--max-det', type=int, required=False, help="maximum YOLO detections per frame", default=None)
    parser.add_argument('--line-margin', type=float, required=False, help="dead zone around counting line", default=0.0)
    parser.add_argument('--count-cooldown', type=int, required=False, help="frames before same track can be counted again", default=0)
    parser.add_argument('--no-display', action='store_true', help="run without showing OpenCV window")
    parser.add_argument('--tracker', type=str, required=False, default="bytetrack.yaml",
                        help="tracker config file (e.g., bytetrack.yaml or path to custom yaml)")
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
        iou_match_thres=args.iou,         # not used, kept for compatibility
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
        count_cooldown=args.count_cooldown,
        tracker=args.tracker
    )
    tracker.run()


if __name__ == "__main__":
    main(sys.argv)