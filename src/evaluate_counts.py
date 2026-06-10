import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path


COUNT_RE = re.compile(r"Final counts:\s*IN=(\d+),\s*OUT=(\d+)")

DEFAULT_SOURCE_DIRS = [
    "data/new_processed",
    "data/processed",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate IN/OUT counting results against ground_truth.csv."
    )
    parser.add_argument(
        "--ground-truth",
        default="data/annotations/ground_truth_baidu.csv",
        help="CSV with filename, gt_in, gt_out and optional line coordinates.",
    )
    parser.add_argument(
        "--source-dir",
        action="append",
        default=None,
        help="Directory with videos. Can be used multiple times. Defaults to common data directories.",
    )
    parser.add_argument(
        "--script",
        default="src/myObjectCounting.py",
        help="Counting script to run.",
    )
    parser.add_argument(
        "--output",
        default="results/evaluation_results.csv",
        help="Detailed evaluation CSV output.",
    )
    parser.add_argument(
        "--summary",
        default="results/evaluation_summary.md",
        help="Markdown summary output.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/videos",
        help="Directory for rendered tracking videos produced during evaluation.",
    )
    parser.add_argument("--model", default="models/best7.pt", help="YOLO model weights.")
    parser.add_argument("--conf", type=float, default=0.3, help="YOLO confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size.")
    parser.add_argument("--nms-iou", type=float, default=0.8, help="YOLO NMS IoU threshold.")
    parser.add_argument("--iou", type=float, default=0.3, help="Tracker IoU matching threshold.")
    parser.add_argument("--max-age", type=int, default=12, help="Frames to keep a track without detection.")
    parser.add_argument("--min-hits", type=int, default=4, help="Detections needed to confirm a track.")
    parser.add_argument("--target-class", type=int, default=0, help="Class id to count.")
    parser.add_argument("--device", default=None, help="Inference device, e.g. cpu or cuda:0.")
    parser.add_argument("--line-margin", type=float, default=8.0, help="Dead zone around counting line.")
    parser.add_argument("--count-cooldown", type=int, default=0, help="Frames before a track can be counted again.")
    parser.add_argument(
        "--line",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=None,
        help="Fallback counting line when a CSV row has no line coordinates.",
    )
    parser.add_argument("--augment", action="store_true", help="Use YOLO test-time augmentation.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N rows.")
    parser.add_argument("--timeout", type=int, default=None, help="Per-video timeout in seconds.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after first failed video.")
    return parser.parse_args()


def resolve_path(path_value, base_dir):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return base_dir / path


def read_ground_truth(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"filename", "gt_in", "gt_out"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {path}: {', '.join(sorted(missing))}")
        return list(reader)


def find_video(filename, source_dirs, base_dir):
    direct = Path(filename)
    if direct.is_absolute() and direct.exists():
        return direct

    for source_dir in source_dirs:
        candidate = resolve_path(source_dir, base_dir) / filename
        if candidate.exists():
            return candidate

    searched = ", ".join(str(resolve_path(d, base_dir)) for d in source_dirs)
    raise FileNotFoundError(f"Could not find {filename}. Searched: {searched}")


def get_line(row, fallback_line):
    keys = ("line_x1", "line_y1", "line_x2", "line_y2")
    values = [row.get(key, "").strip() for key in keys]
    if all(values):
        return [int(value) for value in values]
    if fallback_line is not None:
        return list(fallback_line)
    return None


def get_video_metadata(video_path):
    try:
        import cv2
    except ImportError:
        return {"video_frames": "", "video_fps": "", "video_duration_sec": ""}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video_frames": "", "video_fps": "", "video_duration_sec": ""}

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    duration = frames / fps if fps > 0 else ""
    return {
        "video_frames": frames if frames > 0 else "",
        "video_fps": round(fps, 3) if fps > 0 else "",
        "video_duration_sec": round(duration, 3) if duration != "" else "",
    }


def build_command(args, script_path, video_path, output_video, line):
    cmd = [
        sys.executable,
        str(script_path),
        "--model",
        args.model,
        "--source",
        str(video_path),
        "--output",
        str(output_video),
        "--conf",
        str(args.conf),
        "--iou",
        str(args.iou),
        "--max-age",
        str(args.max_age),
        "--min-hits",
        str(args.min_hits),
        "--target-class",
        str(args.target_class),
        "--line-margin",
        str(args.line_margin),
        "--count-cooldown",
        str(args.count_cooldown),
        "--no-display",
    ]

    if args.imgsz is not None:
        cmd.extend(["--imgsz", str(args.imgsz)])
    if args.nms_iou is not None:
        cmd.extend(["--nms-iou", str(args.nms_iou)])
    if args.device:
        cmd.extend(["--device", args.device])
    if line is not None:
        cmd.extend(["--line", *[str(value) for value in line]])
    if args.augment:
        cmd.append("--augment")

    return cmd


def parse_counts(stdout, stderr):
    text = f"{stdout}\n{stderr}"
    match = COUNT_RE.search(text)
    if not match:
        raise ValueError("Could not parse final counts from myObjectCounting.py output.")
    return int(match.group(1)), int(match.group(2))


def accuracy(gt_in, gt_out, pred_in, pred_out):
    total_gt = gt_in + gt_out
    total_abs_error = abs(gt_in - pred_in) + abs(gt_out - pred_out)
    if total_gt == 0:
        return 1.0 if total_abs_error == 0 else 0.0
    return max(0.0, 1.0 - (total_abs_error / total_gt))


def evaluate_row(args, row, index, source_dirs, base_dir, script_path, output_dir):
    filename = row["filename"].strip()
    video_path = find_video(filename, source_dirs, base_dir)
    line = get_line(row, args.line)
    output_video = output_dir / f"{index:03d}_{Path(filename).stem}_tracked.mp4"
    cmd = build_command(args, script_path, video_path, output_video, line)

    started = time.perf_counter()
    completed = subprocess.run(
        cmd,
        cwd=str(base_dir),
        text=True,
        capture_output=True,
        timeout=args.timeout,
    )
    elapsed = time.perf_counter() - started

    if completed.returncode != 0:
        raise RuntimeError(
            "Counting script failed with exit code "
            f"{completed.returncode}.\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    pred_in, pred_out = parse_counts(completed.stdout, completed.stderr)
    gt_in = int(row["gt_in"])
    gt_out = int(row["gt_out"])
    error_in = abs(gt_in - pred_in)
    error_out = abs(gt_out - pred_out)
    total_abs_error = error_in + error_out
    net_gt = gt_in - gt_out
    net_pred = pred_in - pred_out
    occupancy_error = abs(net_gt - net_pred)
    video_meta = get_video_metadata(video_path)

    video_frames = video_meta["video_frames"]
    processing_fps = ""
    if video_frames != "" and elapsed > 0:
        processing_fps = round(video_frames / elapsed, 3)

    return {
        "filename": filename,
        "dataset": row.get("dataset", "").strip(),
        "gt_in": gt_in,
        "gt_out": gt_out,
        "pred_in": pred_in,
        "pred_out": pred_out,
        "error_in": error_in,
        "error_out": error_out,
        "total_abs_error": total_abs_error,
        "counting_accuracy": round(accuracy(gt_in, gt_out, pred_in, pred_out), 4),
        "counting_accuracy_percent": round(accuracy(gt_in, gt_out, pred_in, pred_out) * 100, 2),
        "occupancy_gt": net_gt,
        "occupancy_pred": net_pred,
        "occupancy_error": occupancy_error,
        "line_x1": line[0] if line else "",
        "line_y1": line[1] if line else "",
        "line_x2": line[2] if line else "",
        "line_y2": line[3] if line else "",
        "video_frames": video_meta["video_frames"],
        "video_fps": video_meta["video_fps"],
        "video_duration_sec": video_meta["video_duration_sec"],
        "processing_seconds": round(elapsed, 3),
        "processing_fps": processing_fps,
        "model": args.model,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "nms_iou": args.nms_iou,
        "tracker_iou": args.iou,
        "max_age": args.max_age,
        "min_hits": args.min_hits,
        "line_margin": args.line_margin,
        "count_cooldown": args.count_cooldown,
        "device": args.device,
        "output_video": str(output_video),
        "notes": row.get("notes", "").strip(),
        "status": "ok",
        "error": "",
    }


def failed_row(row, error):
    return {
        "filename": row.get("filename", "").strip(),
        "dataset": row.get("dataset", "").strip(),
        "gt_in": row.get("gt_in", ""),
        "gt_out": row.get("gt_out", ""),
        "status": "failed",
        "error": str(error).replace("\n", " "),
    }


def write_csv(path, rows):
    if not rows:
        return

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, rows):
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    failed_rows = [row for row in rows if row.get("status") != "ok"]

    total_gt = sum(int(row["gt_in"]) + int(row["gt_out"]) for row in ok_rows)
    total_error = sum(int(row["total_abs_error"]) for row in ok_rows)
    total_error_in = sum(int(row["error_in"]) for row in ok_rows)
    total_error_out = sum(int(row["error_out"]) for row in ok_rows)
    total_occupancy_error = sum(int(row["occupancy_error"]) for row in ok_rows)
    overall_accuracy = max(0.0, 1.0 - total_error / total_gt) if total_gt else 0.0
    mean_accuracy = (
        sum(float(row["counting_accuracy"]) for row in ok_rows) / len(ok_rows)
        if ok_rows
        else 0.0
    )

    lines = [
        "# Evaluation summary",
        "",
        f"- Evaluated videos: {len(ok_rows)}",
        f"- Failed videos: {len(failed_rows)}",
        f"- Total counting accuracy: {overall_accuracy * 100:.2f}%",
        f"- Mean per-video accuracy: {mean_accuracy * 100:.2f}%",
        f"- Total Error_IN: {total_error_in}",
        f"- Total Error_OUT: {total_error_out}",
        f"- Total Occupancy Error: {total_occupancy_error}",
        "",
        "| filename | GT IN | GT OUT | Pred IN | Pred OUT | Error IN | Error OUT | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in ok_rows:
        lines.append(
            "| {filename} | {gt_in} | {gt_out} | {pred_in} | {pred_out} | "
            "{error_in} | {error_out} | {counting_accuracy_percent:.2f}% |".format(**row)
        )

    if failed_rows:
        lines.extend(["", "## Failed videos", ""])
        for row in failed_rows:
            lines.append(f"- {row.get('filename', '')}: {row.get('error', '')}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent.parent
    ground_truth_path = resolve_path(args.ground_truth, base_dir)
    script_path = resolve_path(args.script, base_dir)
    output_path = resolve_path(args.output, base_dir)
    summary_path = resolve_path(args.summary, base_dir)
    output_dir = resolve_path(args.output_dir, base_dir)
    source_dirs = args.source_dir or DEFAULT_SOURCE_DIRS

    rows = read_ground_truth(ground_truth_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for index, row in enumerate(rows, start=1):
        filename = row.get("filename", "").strip()
        print(f"[{index}/{len(rows)}] Evaluating {filename}...")
        try:
            result = evaluate_row(
                args=args,
                row=row,
                index=index,
                source_dirs=source_dirs,
                base_dir=base_dir,
                script_path=script_path,
                output_dir=output_dir,
            )
            print(
                "  GT: IN={gt_in}, OUT={gt_out} | Pred: IN={pred_in}, OUT={pred_out} | "
                "Accuracy={counting_accuracy_percent:.2f}%".format(**result)
            )
            results.append(result)
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            results.append(failed_row(row, exc))
            if args.fail_fast:
                break

    write_csv(output_path, results)
    write_summary(summary_path, results)
    print(f"\nSaved detailed results to: {output_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
