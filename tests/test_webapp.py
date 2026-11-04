"""Tests for the upload-based web interface."""

import io
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from webapp import app


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        app.config.update(
            TESTING=True,
            JOBS_DIR=Path(self.temp_dir.name) / "jobs",
            MAX_CONTENT_LENGTH=100 * 1024 * 1024,
        )
        self.client = app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_homepage_shows_upload_form(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'name="video"', response.data)

    def test_upload_requires_a_file(self):
        response = self.client.post("/", data={}, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Choose a video to upload.", response.data)

    def test_rejects_unsupported_file_types(self):
        response = self.client.post(
            "/",
            data={"video": (io.BytesIO(b"not a video"), "notes.txt")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Choose an MP4, MOV, AVI, MKV, or WebM video.", response.data)

    def test_successful_upload_returns_results_and_csv_download(self):
        def fake_analyze(_video_path, job_dir):
            (job_dir / "timeline.csv").write_text("activity\nStanding\n", encoding="utf-8")
            return {
                "duration": 3.0,
                "activities": [{"name": "Standing", "seconds": 2.5}],
                "squats": 1,
                "jumping_jacks": 0,
                "falls": 0,
                "chart": None,
            }

        with patch("webapp.analyze_video", side_effect=fake_analyze):
            response = self.client.post(
                "/",
                data={"video": (io.BytesIO(b"sample"), "clip.mp4")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Standing", response.data)
        self.assertIn(b"2.5 s", response.data)

        jobs = list((Path(self.temp_dir.name) / "jobs").iterdir())
        self.assertEqual(len(jobs), 1)
        self.assertFalse(any(path.name.startswith("upload.") for path in jobs[0].iterdir()))
        self.assertRegex(
            response.get_data(as_text=True),
            rf'href="/results/{re.escape(jobs[0].name)}/timeline\.csv"',
        )
        download = self.client.get(f"/results/{jobs[0].name}/timeline.csv")
        self.assertEqual(download.status_code, 200)
        self.assertIn(b"Standing", download.data)
        download.close()

    def test_result_download_rejects_unknown_artifacts(self):
        response = self.client.get("/results/not-a-job/server.py")
        self.assertEqual(response.status_code, 404)

    def test_analysis_writes_timeline_and_chart(self):
        class FakeCapture:
            def __init__(self, _path):
                self.frames = iter([np.zeros((2, 2, 3), dtype=np.uint8)] * 2)

            def isOpened(self):
                return True

            def get(self, prop):
                import cv2

                if prop == cv2.CAP_PROP_FPS:
                    return 2.0
                if prop == cv2.CAP_PROP_FRAME_COUNT:
                    return 2.0
                return 0.0

            def read(self):
                frame = next(self.frames, None)
                return (frame is not None, frame)

            def release(self):
                pass

        class FakeDetector:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def process(self, _frame):
                return SimpleNamespace(pose_landmarks=None)

        class FakeRecognizer:
            fall_count = 0
            squat_count = 2
            jumping_jack_count = 1

            def update(self, _landmarks, _timestamp):
                return "Standing", 1.0

        jobs_dir = Path(self.temp_dir.name)
        with patch("webapp.cv2.VideoCapture", FakeCapture), patch(
            "webapp.POSE.Pose", return_value=FakeDetector()
        ), patch("webapp.ActivityRecognizer", return_value=FakeRecognizer()):
            from webapp import analyze_video

            result = analyze_video(Path("sample.mp4"), jobs_dir)

        self.assertEqual(result["duration"], 1.0)
        self.assertEqual(result["activities"], [{"name": "Standing", "seconds": 1.0}])
        self.assertTrue((jobs_dir / "timeline.csv").is_file())
        self.assertTrue((jobs_dir / "activity_chart.png").is_file())


if __name__ == "__main__":
    unittest.main()
