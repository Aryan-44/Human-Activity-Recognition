"""Record labelled pose samples for training your own activity model.

Webcam:   python collect.py
          python collect.py --labels "Standing,Waving,Push-up"
  Press 1-9 to pick the activity, SPACE to start/stop recording (after a
  3 second countdown), Q to quit.

Video:    python collect.py --source walking.mp4 --label Walking
  Records every frame of the video under one label, no keys needed.

Samples are appended to data/samples.csv, so you can record over several runs.
"""

import argparse
import csv
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp

from features import FEATURE_NAMES, FeatureExtractor
from main import draw_text, hud_scale, open_source, open_window, toggle_fullscreen


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

DATA_PATH = Path("data") / "samples.csv"
WINDOW_NAME = "Collect training data"
COUNTDOWN_SECONDS = 3
DEFAULT_LABELS = [
    "Standing",
    "Walking",
    "Sitting",
    "Squatting",
    "Bending",
    "Lying down",
    "One hand raised",
    "Both hands raised",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Record labelled pose samples.")
    parser.add_argument("--source", default="0", help="webcam number or path to a video file")
    parser.add_argument("--labels", help="comma separated activity names (up to 9)")
    parser.add_argument("--label", help="record a whole video file under this one label")
    parser.add_argument("--output", type=Path, default=DATA_PATH, help="CSV file to append to")
    args = parser.parse_args()
    args.labels = [name.strip() for name in args.labels.split(",")] if args.labels else DEFAULT_LABELS
    if len(args.labels) > 9:
        parser.error("use at most 9 labels (one per number key)")
    return args


def count_existing(path):
    if not path.exists():
        return Counter()
    with open(path, newline="") as file:
        return Counter(row["label"] for row in csv.DictReader(file))


def open_output(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    file = open(path, "a", newline="")
    writer = csv.writer(file)
    if is_new:
        writer.writerow(["clip", "label"] + FEATURE_NAMES)
    return file, writer


def draw_panel(frame, labels, current, counts, status, status_color, body_visible):
    height = frame.shape[0]
    scale = hud_scale(frame)

    def text(content, x, y, size, color, thickness=1):
        draw_text(
            frame, content, (int(x * scale), int(y * scale)), size * scale, color,
            max(1, round(thickness * scale)),
        )

    overlay = frame.copy()
    cv2.rectangle(overlay, (int(10 * scale), int(10 * scale)), (int(330 * scale), int((70 + 24 * len(labels)) * scale)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    text(status, 22, 42, 0.75, status_color, 2)
    for i, label in enumerate(labels):
        y = 72 + 24 * i
        color = (0, 255, 255) if label == current else (255, 255, 255)
        text(f"{i + 1}. {label}", 22, y, 0.55, color)
        text(str(counts[label]), 270, y, 0.55, color)

    bottom = height / scale
    if not body_visible:
        text("Shoulders and hips must be visible to record", 12, bottom - 45, 0.6, (0, 165, 255), 2)
    text("1-9: activity   SPACE: record   F: fullscreen   Q: quit", 12, bottom - 15, 0.55, (255, 255, 255))


def main():
    args = parse_args()
    capture, is_file = open_source(args.source)
    auto_record = args.label is not None
    if auto_record and not is_file:
        raise SystemExit("--label is only for video files. On a webcam use the number keys.")

    labels = [args.label] if auto_record else args.labels
    counts = count_existing(args.output)
    extractor = FeatureExtractor()
    file, writer = open_output(args.output)

    current = labels[0]
    recording = auto_record
    record_from = 0.0
    clip_id = f"{datetime.now():%Y%m%d_%H%M%S}_{current}" if auto_record else None
    clip_frames = 0
    window_open = False

    try:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_detector:
            while True:
                success, frame = capture.read()
                if not success:
                    if is_file:
                        break
                    raise RuntimeError("Could not read a frame from the webcam.")
                if not is_file:
                    frame = cv2.flip(frame, 1)

                result = pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                features = None
                if result.pose_landmarks:
                    features = extractor.update(result.pose_landmarks.landmark)
                    mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                else:
                    extractor.reset()

                now = time.time()
                if recording and now < record_from:
                    status, color = f"Get ready... {int(record_from - now) + 1}", (0, 255, 255)
                elif recording:
                    if features is not None:
                        writer.writerow([clip_id, current] + [f"{value:.5f}" for value in features])
                        counts[current] += 1
                        clip_frames += 1
                    status, color = f"RECORDING  {clip_frames} frames", (60, 60, 255)
                else:
                    status, color = "Ready - press SPACE", (80, 220, 80)

                draw_panel(frame, labels, current, counts, status, color, features is not None)
                if not window_open:
                    open_window(WINDOW_NAME, frame)
                    window_open = True
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("f"):
                    toggle_fullscreen(WINDOW_NAME)
                if auto_record:
                    continue
                if ord("1") <= key < ord("1") + len(labels):
                    current = labels[key - ord("1")]
                    recording = False
                elif key == ord(" "):
                    recording = not recording
                    if recording:
                        record_from = now + COUNTDOWN_SECONDS
                        clip_id = f"{datetime.now():%Y%m%d_%H%M%S}_{current}"
                        clip_frames = 0
                    else:
                        file.flush()
    finally:
        file.close()
        capture.release()
        cv2.destroyAllWindows()

    print(f"\nSamples saved to {args.output}")
    for label, count in sorted(counts.items()):
        print(f"  {label:<20}{count:>6} frames")
    print("\nWhen every activity has a few hundred frames, run:  python train.py")


if __name__ == "__main__":
    main()
