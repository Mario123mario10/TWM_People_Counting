import argparse
import csv
import html
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a markdown report and SVG charts from evaluation_results.csv."
    )
    parser.add_argument(
        "--input",
        default="evaluation_results.csv",
        help="CSV produced by evaluate_counts.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_plots",
        help="Directory for generated SVG charts.",
    )
    parser.add_argument(
        "--report",
        default="evaluation_report.md",
        help="Markdown report path.",
    )
    return parser.parse_args()


def to_int(value, default=0):
    try:
        if value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value, default=0.0):
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    failed_rows = [row for row in rows if row.get("status") != "ok"]
    return ok_rows, failed_rows


def label_rows(rows):
    return [f"V{i}" for i in range(1, len(rows) + 1)]


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def svg_header(width, height):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#202124}",
        ".title{font-size:20px;font-weight:700}",
        ".axis{font-size:12px;fill:#5f6368}",
        ".label{font-size:11px;fill:#3c4043}",
        ".grid{stroke:#e0e3e7;stroke-width:1}",
        ".axis-line{stroke:#5f6368;stroke-width:1}",
        ".legend{font-size:12px}",
        "</style>",
    ]


def svg_footer():
    return ["</svg>"]


def nice_max(values):
    max_value = max(values) if values else 1
    if max_value <= 0:
        return 1
    if max_value <= 1:
        return 1
    if max_value <= 5:
        return 5
    if max_value <= 10:
        return 10
    step = 10
    return ((int(max_value) + step - 1) // step) * step


def draw_legend(parts, series, x, y):
    for idx, item in enumerate(series):
        name, color, _values = item
        lx = x + idx * 130
        parts.append(f'<rect x="{lx}" y="{y - 10}" width="12" height="12" fill="{color}" rx="2"/>')
        parts.append(f'<text class="legend" x="{lx + 18}" y="{y}">{html.escape(name)}</text>')


def grouped_bar_chart(path, title, labels, series, y_max=None, value_suffix=""):
    width = 980
    height = 440
    margin_left = 70
    margin_right = 30
    margin_top = 70
    margin_bottom = 70
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    y_max = y_max or nice_max([value for _name, _color, values in series for value in values])
    y_max = max(y_max, 1)

    parts = svg_header(width, height)
    parts.append(f'<text class="title" x="{margin_left}" y="32">{html.escape(title)}</text>')
    draw_legend(parts, series, margin_left, 55)

    for i in range(6):
        value = y_max * i / 5
        y = margin_top + chart_h - (value / y_max) * chart_h
        parts.append(f'<line class="grid" x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}"/>')
        parts.append(f'<text class="axis" x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end">{value:.0f}{value_suffix}</text>')

    parts.append(f'<line class="axis-line" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_h}"/>')
    parts.append(f'<line class="axis-line" x1="{margin_left}" y1="{margin_top + chart_h}" x2="{width - margin_right}" y2="{margin_top + chart_h}"/>')

    groups = max(len(labels), 1)
    group_w = chart_w / groups
    series_count = max(len(series), 1)
    bar_w = min(24, group_w * 0.7 / series_count)

    for i, label in enumerate(labels):
        group_center = margin_left + group_w * i + group_w / 2
        first_x = group_center - (series_count * bar_w) / 2
        for j, (_name, color, values) in enumerate(series):
            value = values[i] if i < len(values) else 0
            bar_h = (value / y_max) * chart_h
            x = first_x + j * bar_w
            y = margin_top + chart_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 2:.1f}" height="{bar_h:.1f}" fill="{color}" rx="2"/>')
        parts.append(f'<text class="axis" x="{group_center:.1f}" y="{margin_top + chart_h + 22}" text-anchor="middle">{html.escape(label)}</text>')

    parts.extend(svg_footer())
    write_text(path, "\n".join(parts))


def stacked_bar_chart(path, title, labels, series, y_max=None):
    width = 980
    height = 440
    margin_left = 70
    margin_right = 30
    margin_top = 70
    margin_bottom = 70
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    totals = [sum(values[i] for _name, _color, values in series) for i in range(len(labels))]
    y_max = y_max or nice_max(totals)
    y_max = max(y_max, 1)

    parts = svg_header(width, height)
    parts.append(f'<text class="title" x="{margin_left}" y="32">{html.escape(title)}</text>')
    draw_legend(parts, series, margin_left, 55)

    for i in range(6):
        value = y_max * i / 5
        y = margin_top + chart_h - (value / y_max) * chart_h
        parts.append(f'<line class="grid" x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}"/>')
        parts.append(f'<text class="axis" x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end">{value:.0f}</text>')

    groups = max(len(labels), 1)
    group_w = chart_w / groups
    bar_w = min(34, group_w * 0.55)

    for i, label in enumerate(labels):
        group_center = margin_left + group_w * i + group_w / 2
        x = group_center - bar_w / 2
        cumulative = 0
        for _name, color, values in series:
            value = values[i] if i < len(values) else 0
            bar_h = (value / y_max) * chart_h
            y = margin_top + chart_h - ((cumulative + value) / y_max) * chart_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="2"/>')
            cumulative += value
        parts.append(f'<text class="axis" x="{group_center:.1f}" y="{margin_top + chart_h + 22}" text-anchor="middle">{html.escape(label)}</text>')

    parts.append(f'<line class="axis-line" x1="{margin_left}" y1="{margin_top + chart_h}" x2="{width - margin_right}" y2="{margin_top + chart_h}"/>')
    parts.extend(svg_footer())
    write_text(path, "\n".join(parts))


def totals_chart(path, title, totals):
    labels = ["IN", "OUT", "Total"]
    gt_values = [totals["gt_in"], totals["gt_out"], totals["gt_in"] + totals["gt_out"]]
    pred_values = [totals["pred_in"], totals["pred_out"], totals["pred_in"] + totals["pred_out"]]
    grouped_bar_chart(
        path,
        title,
        labels,
        [
            ("GT", "#1a73e8", gt_values),
            ("Pred", "#fbbc04", pred_values),
        ],
    )


def build_summary(rows):
    total_gt_in = sum(to_int(row.get("gt_in")) for row in rows)
    total_gt_out = sum(to_int(row.get("gt_out")) for row in rows)
    total_pred_in = sum(to_int(row.get("pred_in")) for row in rows)
    total_pred_out = sum(to_int(row.get("pred_out")) for row in rows)
    total_error_in = sum(to_int(row.get("error_in")) for row in rows)
    total_error_out = sum(to_int(row.get("error_out")) for row in rows)
    total_abs_error = total_error_in + total_error_out
    total_gt = total_gt_in + total_gt_out
    total_accuracy = max(0.0, 1.0 - total_abs_error / total_gt) if total_gt else 0.0

    accuracies = [to_float(row.get("counting_accuracy_percent")) for row in rows]
    fps_values = [to_float(row.get("processing_fps")) for row in rows if row.get("processing_fps") not in ("", None)]
    occupancy_errors = [to_int(row.get("occupancy_error")) for row in rows]

    return {
        "videos": len(rows),
        "gt_in": total_gt_in,
        "gt_out": total_gt_out,
        "pred_in": total_pred_in,
        "pred_out": total_pred_out,
        "error_in": total_error_in,
        "error_out": total_error_out,
        "total_abs_error": total_abs_error,
        "total_accuracy_percent": total_accuracy * 100,
        "mean_accuracy_percent": statistics.mean(accuracies) if accuracies else 0.0,
        "median_accuracy_percent": statistics.median(accuracies) if accuracies else 0.0,
        "mean_processing_fps": statistics.mean(fps_values) if fps_values else 0.0,
        "total_occupancy_error": sum(occupancy_errors),
    }


def format_percent(value):
    return f"{value:.2f}%"


def generate_report(path, chart_dir, rows, failed_rows, summary):
    sorted_by_accuracy = sorted(rows, key=lambda row: to_float(row.get("counting_accuracy_percent")))
    worst = sorted_by_accuracy[:3]
    best = list(reversed(sorted_by_accuracy[-3:]))

    lines = [
        "# Evaluation report",
        "",
        "## Overall summary",
        "",
        f"- Evaluated videos: {summary['videos']}",
        f"- Failed videos: {len(failed_rows)}",
        f"- Total GT IN/OUT: {summary['gt_in']} / {summary['gt_out']}",
        f"- Total predicted IN/OUT: {summary['pred_in']} / {summary['pred_out']}",
        f"- Total Error_IN / Error_OUT: {summary['error_in']} / {summary['error_out']}",
        f"- Total counting accuracy: {format_percent(summary['total_accuracy_percent'])}",
        f"- Mean per-video accuracy: {format_percent(summary['mean_accuracy_percent'])}",
        f"- Median per-video accuracy: {format_percent(summary['median_accuracy_percent'])}",
        f"- Mean processing FPS: {summary['mean_processing_fps']:.2f}",
        f"- Total Occupancy Error: {summary['total_occupancy_error']}",
        "",
        "## Charts",
        "",
        f"![Accuracy by video]({chart_dir.name}/accuracy_by_video.svg)",
        "",
        f"![Errors by video]({chart_dir.name}/errors_by_video.svg)",
        "",
        f"![GT vs predicted totals]({chart_dir.name}/gt_vs_pred_totals.svg)",
        "",
        f"![Processing FPS]({chart_dir.name}/processing_fps_by_video.svg)",
        "",
        "## Video key",
        "",
        "| ID | filename | GT IN | GT OUT | Pred IN | Pred OUT | Accuracy | Notes |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]

    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| V{idx} | {filename} | {gt_in} | {gt_out} | {pred_in} | {pred_out} | {accuracy:.2f}% | {notes} |".format(
                idx=idx,
                filename=row.get("filename", ""),
                gt_in=row.get("gt_in", ""),
                gt_out=row.get("gt_out", ""),
                pred_in=row.get("pred_in", ""),
                pred_out=row.get("pred_out", ""),
                accuracy=to_float(row.get("counting_accuracy_percent")),
                notes=row.get("notes", "").replace("|", "/"),
            )
        )

    lines.extend(["", "## Worst cases", ""])
    for row in worst:
        lines.append(
            f"- {row.get('filename')}: {to_float(row.get('counting_accuracy_percent')):.2f}% "
            f"(Error_IN={row.get('error_in')}, Error_OUT={row.get('error_out')})"
        )

    lines.extend(["", "## Best cases", ""])
    for row in best:
        lines.append(
            f"- {row.get('filename')}: {to_float(row.get('counting_accuracy_percent')):.2f}% "
            f"(GT={row.get('gt_in')}/{row.get('gt_out')}, Pred={row.get('pred_in')}/{row.get('pred_out')})"
        )

    if failed_rows:
        lines.extend(["", "## Failed rows", ""])
        for row in failed_rows:
            lines.append(f"- {row.get('filename', '')}: {row.get('error', '')}")

    write_text(path, "\n".join(lines) + "\n")


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / args.input
    chart_dir = base_dir / args.output_dir
    report_path = base_dir / args.report

    rows, failed_rows = read_rows(input_path)
    if not rows:
        raise SystemExit(f"No successful rows found in {input_path}")

    chart_dir.mkdir(parents=True, exist_ok=True)
    labels = label_rows(rows)

    accuracies = [to_float(row.get("counting_accuracy_percent")) for row in rows]
    error_in = [to_int(row.get("error_in")) for row in rows]
    error_out = [to_int(row.get("error_out")) for row in rows]
    fps = [to_float(row.get("processing_fps")) for row in rows]
    summary = build_summary(rows)

    grouped_bar_chart(
        chart_dir / "accuracy_by_video.svg",
        "Counting accuracy by video",
        labels,
        [("Accuracy", "#34a853", accuracies)],
        y_max=100,
        value_suffix="%",
    )
    stacked_bar_chart(
        chart_dir / "errors_by_video.svg",
        "Absolute counting errors by video",
        labels,
        [
            ("Error IN", "#ea4335", error_in),
            ("Error OUT", "#fbbc04", error_out),
        ],
    )
    totals_chart(
        chart_dir / "gt_vs_pred_totals.svg",
        "Ground truth vs predicted totals",
        summary,
    )
    grouped_bar_chart(
        chart_dir / "processing_fps_by_video.svg",
        "Processing FPS by video",
        labels,
        [("FPS", "#1a73e8", fps)],
    )
    generate_report(report_path, chart_dir, rows, failed_rows, summary)

    print(f"Saved report to: {report_path}")
    print(f"Saved charts to: {chart_dir}")


if __name__ == "__main__":
    main()
