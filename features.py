"""Turns MediaPipe Pose landmarks into a fixed-length feature vector for ML."""

import math
from collections import deque

import numpy as np

from pose_utils import POSE, angle, is_visible


KEY_POINTS = (
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
)

ANGLES = {
    "left_knee_angle": ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
    "right_knee_angle": ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
    "left_hip_angle": ("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"),
    "right_hip_angle": ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"),
    "left_elbow_angle": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    "right_elbow_angle": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
}

# Points whose speed over the last few frames is used as a motion feature.
MOVING_POINTS = ("LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_WRIST", "RIGHT_WRIST")

FEATURE_NAMES = (
    [f"{name.lower()}_{axis}" for name in KEY_POINTS for axis in ("x", "y")]
    + list(ANGLES)
    + ["torso_angle"]
    + [f"{name.lower()}_speed" for name in MOVING_POINTS]
    + ["hip_vertical_speed"]
)


class FeatureExtractor:
    """Builds features from the current frame plus motion over recent frames.

    Positions are measured from the hip center and divided by the torso
    length, so the features do not depend on where the person stands or how
    far they are from the camera.
    """

    def __init__(self, window_frames=8):
        self.history = deque(maxlen=window_frames)

    def reset(self):
        self.history.clear()

    def update(self, landmarks):
        """Return a numpy feature vector, or None if the body is not visible."""
        points = {name: landmarks[POSE[name].value] for name in KEY_POINTS}
        core = ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP")
        if not all(is_visible(points[name]) for name in core):
            self.reset()
            return None

        hip_x = (points["LEFT_HIP"].x + points["RIGHT_HIP"].x) / 2
        hip_y = (points["LEFT_HIP"].y + points["RIGHT_HIP"].y) / 2
        shoulder_x = (points["LEFT_SHOULDER"].x + points["RIGHT_SHOULDER"].x) / 2
        shoulder_y = (points["LEFT_SHOULDER"].y + points["RIGHT_SHOULDER"].y) / 2
        torso_length = math.dist((hip_x, hip_y), (shoulder_x, shoulder_y))
        if torso_length == 0:
            self.reset()
            return None

        positions = []
        for name in KEY_POINTS:
            positions += [
                (points[name].x - hip_x) / torso_length,
                (points[name].y - hip_y) / torso_length,
            ]

        # Angles are scaled to 0..1 so they sit on a similar range as positions.
        angles = [angle(*(points[name] for name in names)) / 180 for names in ANGLES.values()]
        torso_angle = math.degrees(math.atan2(abs(shoulder_x - hip_x), abs(shoulder_y - hip_y))) / 90

        # Motion is measured in image space (not hip-relative), so walking
        # across the room counts as movement too.
        self.history.append(
            {
                "points": {name: (points[name].x, points[name].y) for name in MOVING_POINTS},
                "hip_y": hip_y,
                "torso": torso_length,
            }
        )
        motion = self._motion_features()
        return np.array(positions + angles + [torso_angle] + motion, dtype=np.float32)

    def _motion_features(self):
        frames = list(self.history)
        if len(frames) < 2:
            return [0.0] * (len(MOVING_POINTS) + 1)

        scale = sum(frame["torso"] for frame in frames) / len(frames)
        steps = len(frames) - 1
        speeds = [
            sum(
                math.dist(frames[i]["points"][name], frames[i - 1]["points"][name])
                for i in range(1, len(frames))
            )
            / steps
            / scale
            for name in MOVING_POINTS
        ]
        hip_speed = (frames[-1]["hip_y"] - frames[0]["hip_y"]) / steps / scale
        return speeds + [hip_speed]
