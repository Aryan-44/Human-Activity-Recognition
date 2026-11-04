"""Human Activity Recognition demo using a webcam or a video file."""

import argparse
import csv
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from activity import (
    CLAPPING,
    JUMPING_JACKS,
    NO_PERSON,
    PARTLY_VISIBLE,
    TOUCHING_FACE,
    WAVING,
    ActivityRecognizer,
)
from model import DEFAULT_MODEL_PATH, ActivityModel

try:
    import winsound
except ImportError:  # not on Windows
    winsound = None


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

WINDOW_NAME = "Human Activity Recognition"
OUTPUT_DIR = Path("output")
FALL_EVENT = "FALL DETECTED"

# Text colors in BGR order.
ACTIVITY_COLORS = {
    "Standing": (80, 220, 80),
    "Walking": (255, 200, 0),
    "Sitting": (255, 140, 60),
    "Squatting": (60, 200, 255),
    "Bending": (200, 120, 255),
    "Lying down": (180, 105, 255),
    "One hand raised": (0, 255, 255),
    "Both hands raised": (0, 255, 255),
    JUMPING_JACKS: (255, 255, 0),
    WAVING: (0, 230, 255),
    CLAPPING: (255, 120, 255),
    TOUCHING_FACE: (200, 200, 255),
    PARTLY_VISIBLE: (0, 165, 255),
    NO_PERSON: (90, 90, 255),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="0",
        help="webcam number (0, 1, ...) or path to a video file (default: 0)",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="also save the annotated video to the output folder",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"trained model from train.py (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--rules-only",
        action="store_true",
        help="ignore the trained model and use only the hand-written rules",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="start in fullscreen (press F to switch at any time)",
    )
    return parser.parse_args()


def load_model(args):
    if args.rules_only:
        return None
    if not args.model.exists():
        print(f"No trained model at {args.model}, using rules. See README to train one.")
        return None
    model = ActivityModel(args.model)
    print(f"Loaded {model.name} model with activities: {', '.join(model.labels)}")
    return model


def open_source(source):
    """Open a webcam (if source is a number) or a video file."""
    is_file = not source.isdigit()
    capture = cv2.VideoCapture(source if is_file else int(source))
    if not capture.isOpened():
        what = f"video file '{source}'" if is_file else f"webcam {source}"
        raise RuntimeError(f"Could not open {what}. Check the path, connection and permissions.")
    if not is_file:
        # Ask for HD. Cameras that can't do it simply keep their own resolution.
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return capture, is_file


def screen_size():
    try:
        import ctypes

        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except (AttributeError, OSError):  # not on Windows
        return 1280, 720


def open_window(name, frame, fullscreen=False):
    """Create a big, resizable window that fills most of the screen."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        return
    screen_width, screen_height = screen_size()
    height, width = frame.shape[:2]
    scale = min(screen_width * 0.9 / width, screen_height * 0.85 / height)
    window_width, window_height = int(width * scale), int(height * scale)
    cv2.resizeWindow(name, window_width, window_height)
    cv2.moveWindow(name, (screen_width - window_width) // 2, (screen_height - window_height) // 3)


def toggle_fullscreen(name):
    is_full = cv2.getWindowProperty(name, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
    cv2.setWindowProperty(
        name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL if is_full else cv2.WINDOW_FULLSCREEN
    )


def draw_text(frame, text, position, scale, color, thickness=2):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def hud_scale(frame):
    """Size factor for text and panels, so they look the same at any resolution.

    The layouts are written in 640 x 480 coordinates; 0.7 keeps the panel
    small enough not to cover the person.
    """
    height, width = frame.shape[:2]
    return 0.7 * min(width / 640, height / 480)


def draw_hud(frame, activity, confidence, recognizer, seconds_in_activity, fps):
    """Draw the information panel on top of the frame."""
    height, width = frame.shape[:2]
    scale = hud_scale(frame)

    def px(value):
        return int(value * scale)

    def text(content, x, y, size, color, thickness=1):
        draw_text(frame, content, (px(x), px(y)), size * scale, color, max(1, round(thickness * scale)))

    overlay = frame.copy()
    cv2.rectangle(overlay, (px(10), px(10)), (px(470), px(234)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    white = (255, 255, 255)
    color = ACTIVITY_COLORS.get(activity, white)
    knee = f"{recognizer.knee_angle:.0f} deg" if recognizer.knee_angle else "-"
    rows = [
        (f"Confidence: {confidence:.0%}", f"FPS: {fps:.0f}"),
        (f"Time in activity: {seconds_in_activity:.1f}s", f"Decided by: {recognizer.decided_by}"),
        (f"View: {recognizer.view}", f"Falls: {recognizer.fall_count}"),
        (f"Squats: {recognizer.squat_count}", f"Jumping jacks: {recognizer.jumping_jack_count}"),
        (f"Knee angle: {knee}", ""),
    ]
    text(activity, 22, 48, 0.9, color, 2)
    for i, (left, right) in enumerate(rows):
        y = 80 + 26 * i
        text(left, 22, y, 0.55, white)
        text(right, 262, y, 0.55, white)

    # Confidence bar
    cv2.rectangle(frame, (px(22), px(214)), (px(452), px(222)), (80, 80, 80), -1)
    cv2.rectangle(frame, (px(22), px(214)), (px(22 + 430 * confidence), px(222)), color, -1)

    if activity == WAVING and recognizer.waving_hand:
        draw_speech_bubble(frame, "Hi!", recognizer.waving_hand, scale)

    if recognizer.fall_alert:
        red = (40, 40, 230)
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), red, px(12))
        cv2.rectangle(frame, (0, height - px(130)), (width, height - px(60)), red, -1)
        label = "FALL DETECTED!"
        label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale, px(3))[0][0]
        draw_text(frame, label, ((width - label_width) // 2, height - px(82)), 1.2 * scale, white, px(3))

    draw_text(
        frame,
        "Q: quit   R: reset   S: screenshot   F: fullscreen",
        (px(12), height - px(15)),
        0.55 * scale,
        white,
        max(1, round(scale)),
    )


def draw_speech_bubble(frame, message, hand, scale):
    """Draw a speech bubble next to the waving hand (hand is in 0..1 coordinates)."""
    height, width = frame.shape[:2]
    font_scale, thickness = 1.4 * scale, max(2, round(3 * scale))
    (text_width, text_height), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    pad = int(14 * scale)
    box_width, box_height = text_width + 2 * pad, text_height + 2 * pad

    hand_x, hand_y = int(hand[0] * width), int(hand[1] * height)
    left = min(max(hand_x + int(30 * scale), 0), width - box_width)
    top = min(max(hand_y - box_height - int(40 * scale), 0), height - box_height)

    tail = np.array(
        [[left + pad, top + box_height], [left + 3 * pad, top + box_height], [hand_x, hand_y]], np.int32
    )
    cv2.fillPoly(frame, [tail], (255, 255, 255))
    cv2.rectangle(frame, (left, top), (left + box_width, top + box_height), (255, 255, 255), -1)
    cv2.rectangle(frame, (left, top), (left + box_width, top + box_height), (0, 160, 230), max(2, int(2 * scale)))
    draw_text(frame, message, (left + pad, top + pad + text_height), font_scale, (0, 120, 200), thickness)


def beep():
    if winsound is not None:
        winsound.MessageBeep(winsound.MB_ICONHAND)  # returns at once, does not block


def save_session(segments, started_at, recognizer):
    """Save the activity timeline as CSV and a time-per-activity chart."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    stamp = started_at.strftime("%Y%m%d_%H%M%S")

    csv_path = OUTPUT_DIR / f"session_{stamp}.csv"
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["start_seconds", "end_seconds", "duration_seconds", "activity"])
        for start, end, activity in sorted(segments):
            writer.writerow([f"{start:.2f}", f"{end:.2f}", f"{end - start:.2f}", activity])

    totals = defaultdict(float)
    for start, end, activity in segments:
        if activity not in (NO_PERSON, PARTLY_VISIBLE, FALL_EVENT):
            totals[activity] += end - start

    print("\nSession summary")
    print("-" * 36)
    if not totals:
        print("No activities were recognized.")
    for activity, seconds in sorted(totals.items(), key=lambda item: -item[1]):
        print(f"{activity:<24}{seconds:>8.1f} s")
    print(f"\nSquats: {recognizer.squat_count}")
    print(f"Jumping jacks: {recognizer.jumping_jack_count}")
    print(f"Falls detected: {recognizer.fall_count}")
    print(f"\nTimeline saved to {csv_path}")

    if totals:
        chart_path = OUTPUT_DIR / f"session_{stamp}.png"
        save_chart(totals, chart_path)
        print(f"Chart saved to {chart_path}")


def save_chart(totals, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = sorted(totals.items(), key=lambda item: item[1])
    names = [name for name, _ in items]
    seconds = [value for _, value in items]

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(items) + 1.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    bars = ax.barh(names, seconds, color="#2a78d6", height=0.6)
    ax.bar_label(bars, labels=[f"{s:.1f}s" for s in seconds], padding=4, color="#52514e")
    ax.set_title("Time spent in each activity", loc="left", color="#0b0b0b", fontsize=13)
    ax.set_xlabel("Seconds", color="#52514e")
    ax.tick_params(colors="#52514e", length=0)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    capture, is_file = open_source(args.source)
    recognizer = ActivityRecognizer(model=load_model(args))

    started_at = datetime.now()
    start_time = time.time()
    segments = []  # (start_seconds, end_seconds, activity)
    current_activity, segment_start = None, 0.0
    previous_frame_time, fps = time.time(), 0.0
    writer = None
    timestamp = 0.0
    window_open = False

    try:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_detector:
            while True:
                success, frame = capture.read()
                if not success:
                    if is_file:
                        break  # end of the video
                    raise RuntimeError("Could not read a frame from the webcam.")

                if not is_file:
                    frame = cv2.flip(frame, 1)  # mirror view feels natural on a webcam
                result = pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Video files use their own clock so the timings match the video.
                timestamp = (
                    capture.get(cv2.CAP_PROP_POS_MSEC) / 1000 if is_file else time.time() - start_time
                )
                landmarks = result.pose_landmarks.landmark if result.pose_landmarks else None
                falls_before = recognizer.fall_count
                activity, confidence = recognizer.update(landmarks, timestamp)
                if recognizer.fall_count > falls_before:
                    segments.append((timestamp, timestamp, FALL_EVENT))
                    print(f"Fall detected at {timestamp:.1f}s")
                    beep()
                if result.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                    )

                if activity != current_activity:
                    if current_activity is not None:
                        segments.append((segment_start, timestamp, current_activity))
                    current_activity, segment_start = activity, timestamp

                now = time.time()
                fps = 0.9 * fps + 0.1 / max(now - previous_frame_time, 1e-6)
                previous_frame_time = now

                draw_hud(frame, activity, confidence, recognizer, timestamp - segment_start, fps)
                if not window_open:
                    open_window(WINDOW_NAME, frame, args.fullscreen)
                    window_open = True
                cv2.imshow(WINDOW_NAME, frame)

                if args.save_video:
                    if writer is None:
                        OUTPUT_DIR.mkdir(exist_ok=True)
                        video_path = OUTPUT_DIR / f"video_{started_at:%Y%m%d_%H%M%S}.mp4"
                        video_fps = capture.get(cv2.CAP_PROP_FPS) or 30
                        height, width = frame.shape[:2]
                        writer = cv2.VideoWriter(
                            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (width, height)
                        )
                    writer.write(frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("r"):
                    recognizer.reset()
                if key == ord("f"):
                    toggle_fullscreen(WINDOW_NAME)
                if key == ord("s"):
                    OUTPUT_DIR.mkdir(exist_ok=True)
                    shot = OUTPUT_DIR / f"screenshot_{datetime.now():%Y%m%d_%H%M%S}.png"
                    cv2.imwrite(str(shot), frame)
                    print(f"Screenshot saved to {shot}")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if current_activity is not None:
        segments.append((segment_start, timestamp, current_activity))
    save_session(segments, started_at, recognizer)


if __name__ == "__main__":
    main()
