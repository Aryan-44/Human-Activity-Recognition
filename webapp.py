"""Web interface for activity recognition on uploaded videos."""

import csv
import math
import os
import secrets
import shutil
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import mediapipe as mp
from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for

from activity import NO_PERSON, PARTLY_VISIBLE, ActivityRecognizer
from model import DEFAULT_MODEL_PATH, ActivityModel


matplotlib.use("Agg")
import matplotlib.pyplot as plt


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
app.config["JOBS_DIR"] = Path("output") / "web_jobs"

ALLOWED_EXTENSIONS = {"avi", "mkv", "mov", "mp4", "webm"}
MAX_VIDEO_SECONDS = 120
MAX_VIDEO_FRAMES = 14400
JOB_RETENTION_SECONDS = 6 * 60 * 60
POSE = mp.solutions.pose
FALL_EVENT = "FALL DETECTED"


class VideoProcessingError(ValueError):
    """An uploaded file could not be processed as a supported video."""


def _cleanup_old_jobs():
    jobs_dir = Path(app.config["JOBS_DIR"])
    jobs_dir.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - JOB_RETENTION_SECONDS
    for job_dir in jobs_dir.iterdir():
        if job_dir.is_dir() and job_dir.stat().st_mtime < cutoff:
            shutil.rmtree(job_dir)


def _save_chart(totals, path):
    items = sorted(totals.items(), key=lambda item: item[1])
    names = [name for name, _ in items]
    seconds = [value for _, value in items]

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(items) + 1.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    bars = ax.barh(names, seconds, color="#2a78d6", height=0.6)
    ax.bar_label(bars, labels=[f"{value:.1f}s" for value in seconds], padding=4)
    ax.set_title("Time spent in each activity", loc="left")
    ax.set_xlabel("Seconds")
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def analyze_video(video_path, job_dir):
    """Analyze an uploaded video and save its timeline and activity chart."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise VideoProcessingError("We couldn't read that video. Try another supported video file.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps > 0 and frame_count > 0 and frame_count / fps > MAX_VIDEO_SECONDS:
        capture.release()
        raise VideoProcessingError("Videos must be 2 minutes or shorter.")

    fps = fps if math.isfinite(fps) and fps > 0 else 30.0
    recognizer = ActivityRecognizer(
        model=ActivityModel(DEFAULT_MODEL_PATH) if DEFAULT_MODEL_PATH.exists() else None
    )
    segments = []
    current_activity = None
    segment_start = 0.0
    timestamp = 0.0
    processed_frames = 0

    try:
        with POSE.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as detector:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                if (
                    processed_frames >= MAX_VIDEO_FRAMES
                    or processed_frames / fps >= MAX_VIDEO_SECONDS
                ):
                    raise VideoProcessingError("Videos must be 2 minutes or shorter.")

                timestamp = processed_frames / fps
                result = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                landmarks = result.pose_landmarks.landmark if result.pose_landmarks else None
                falls_before = recognizer.fall_count
                activity, _ = recognizer.update(landmarks, timestamp)

                if recognizer.fall_count > falls_before:
                    segments.append((timestamp, timestamp, FALL_EVENT))
                if activity != current_activity:
                    if current_activity is not None:
                        segments.append((segment_start, timestamp, current_activity))
                    current_activity, segment_start = activity, timestamp

                processed_frames += 1

    except VideoProcessingError:
        raise
    except Exception as error:
        raise VideoProcessingError("An error occurred while analyzing this video.") from error
    finally:
        capture.release()

    if processed_frames == 0:
        raise VideoProcessingError("That file didn't contain any readable video frames.")
    if current_activity is not None:
        segments.append((segment_start, timestamp + 1 / fps, current_activity))

    totals = defaultdict(float)
    for start, end, activity in segments:
        if activity not in (NO_PERSON, PARTLY_VISIBLE, FALL_EVENT):
            totals[activity] += end - start

    timeline_path = Path(job_dir) / "timeline.csv"
    with timeline_path.open("w", newline="", encoding="utf-8") as timeline_file:
        writer = csv.writer(timeline_file)
        writer.writerow(["start_seconds", "end_seconds", "duration_seconds", "activity"])
        for start, end, activity in sorted(segments):
            writer.writerow(
                [f"{start:.2f}", f"{end:.2f}", f"{end - start:.2f}", activity]
            )

    chart_name = None
    if totals:
        chart_name = "activity_chart.png"
        _save_chart(totals, Path(job_dir) / chart_name)

    return {
        "duration": processed_frames / fps,
        "activities": [
            {"name": name, "seconds": seconds}
            for name, seconds in sorted(totals.items(), key=lambda item: -item[1])
        ],
        "squats": recognizer.squat_count,
        "jumping_jacks": recognizer.jumping_jack_count,
        "falls": recognizer.fall_count,
        "chart": chart_name,
    }


@app.errorhandler(413)
def upload_too_large(_error):
    flash("The video is too large. Maximum upload size is 100 MB.")
    return render_template("index.html"), 413


@app.get("/")
def index():
    _cleanup_old_jobs()
    return render_template("index.html")


@app.post("/")
def process_upload():
    _cleanup_old_jobs()
    video = request.files.get("video")
    if video is None or not video.filename:
        flash("Choose a video to upload.")
        return redirect(url_for("index"))

    suffix = Path(video.filename).suffix.lower().lstrip(".")
    if suffix not in ALLOWED_EXTENSIONS:
        flash("Choose an MP4, MOV, AVI, MKV, or WebM video.")
        return redirect(url_for("index"))

    job_id = uuid.uuid4().hex
    job_dir = Path(app.config["JOBS_DIR"]) / job_id
    job_dir.mkdir(parents=True)
    video_path = job_dir / f"upload.{suffix}"
    try:
        video.save(video_path)
        result = analyze_video(video_path, job_dir)
    except VideoProcessingError as error:
        app.logger.info("Video processing rejected for job %s: %s", job_id, error)
        shutil.rmtree(job_dir)
        flash(str(error))
        return redirect(url_for("index"))
    except Exception:
        app.logger.exception("Unexpected video processing failure for job %s", job_id)
        shutil.rmtree(job_dir)
        flash("The video could not be processed. Please try a different file.")
        return redirect(url_for("index"))
    finally:
        video_path.unlink(missing_ok=True)

    return render_template(
        "result.html",
        result=result,
        job_id=job_id,
        filename=Path(video.filename).name,
        analyzed_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


@app.get("/results/<job_id>/<artifact>")
def download_result(job_id, artifact):
    if artifact not in {"timeline.csv", "activity_chart.png"}:
        return "Not found", 404
    try:
        normalized_id = uuid.UUID(hex=job_id).hex
    except ValueError:
        return "Not found", 404
    job_dir = Path(app.config["JOBS_DIR"]) / normalized_id
    if not (job_dir / artifact).is_file():
        return "Not found", 404
    return send_from_directory(job_dir, artifact, as_attachment=artifact == "timeline.csv")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
