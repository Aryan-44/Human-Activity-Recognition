"""Activity recognition: rules, an optional ML model, gestures, rep counters and fall alerts."""

import math
from collections import Counter, deque

from features import FeatureExtractor
from gestures import ClapDetector, WaveDetector, shoulder_width
from pose_utils import POSE, angle, center, is_visible


NO_PERSON = "No person detected"
PARTLY_VISIBLE = "Person partly visible"
JUMPING_JACKS = "Jumping jacks"
WAVING = "Waving - Hi!"
CLAPPING = "Clapping"
TOUCHING_FACE = "Touching face"

BODY_PARTS = (
    "NOSE",
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


def measure_body(points):
    """Return hip height, torso length and torso tilt, or None if not visible."""
    shoulders = [points[n] for n in ("LEFT_SHOULDER", "RIGHT_SHOULDER") if is_visible(points[n])]
    hips = [points[n] for n in ("LEFT_HIP", "RIGHT_HIP") if is_visible(points[n])]
    if not shoulders or not hips:
        return None

    shoulder_center = center(shoulders)
    hip_center = center(hips)
    torso_length = math.dist(shoulder_center, hip_center)
    if torso_length == 0:
        return None

    # 0 degrees = upright torso, 90 degrees = horizontal torso.
    torso_angle = math.degrees(
        math.atan2(
            abs(shoulder_center[0] - hip_center[0]),
            abs(shoulder_center[1] - hip_center[1]),
        )
    )
    return {"hip_y": hip_center[1], "torso_length": torso_length, "torso_angle": torso_angle}


def describe_view(points):
    """How much of the person the camera can see."""
    if any(is_visible(points[n]) for n in ("LEFT_ANKLE", "RIGHT_ANKLE")):
        return "Full body"
    if any(is_visible(points[n]) for n in ("LEFT_KNEE", "RIGHT_KNEE")):
        return "Down to knees"
    if any(is_visible(points[n]) for n in ("LEFT_SHOULDER", "RIGHT_SHOULDER")):
        return "Upper body"
    if is_visible(points["NOSE"]):
        return "Face only"
    return "Partial"


class FallDetector:
    """Flags a fall: the hips drop fast and the body ends up horizontal.

    Lying down slowly on purpose takes longer than `window_seconds`, so it
    does not trigger the alert.
    """

    def __init__(self, window_seconds=1.0, drop_ratio=0.6, alert_seconds=3.0):
        self.window_seconds = window_seconds
        self.drop_ratio = drop_ratio
        self.alert_seconds = alert_seconds
        self.history = deque()  # (timestamp, hip_y, torso_length)
        self.fall_count = 0
        self.alert_until = -1.0

    def reset(self):
        self.history.clear()
        self.fall_count = 0
        self.alert_until = -1.0

    def update(self, timestamp, body):
        """Return True while a fall alert is active."""
        if body is not None:
            self.history.append((timestamp, body["hip_y"], body["torso_length"]))
            while timestamp - self.history[0][0] > self.window_seconds:
                self.history.popleft()

            highest_hip = min(hip_y for _, hip_y, _ in self.history)  # smaller y = higher
            torso_length = max(length for _, _, length in self.history)
            if (
                body["torso_angle"] >= 60
                and body["hip_y"] - highest_hip > self.drop_ratio * torso_length
            ):
                self.fall_count += 1
                self.alert_until = timestamp + self.alert_seconds
                self.history.clear()
        return timestamp < self.alert_until


class JumpingJackCounter:
    """Counts jumping jacks: arms up with feet apart, then arms down with feet together."""

    def __init__(self, active_seconds=2.0):
        self.active_seconds = active_seconds
        self.count = 0
        self.last_rep_time = None
        self._open = False

    def reset(self):
        self.count = 0
        self.last_rep_time = None
        self._open = False

    def update(self, points, timestamp):
        names = ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_ANKLE", "RIGHT_ANKLE")
        if not all(is_visible(points[name]) for name in names):
            return
        shoulder_width = abs(points["LEFT_SHOULDER"].x - points["RIGHT_SHOULDER"].x)
        if shoulder_width == 0:
            return

        feet_gap = abs(points["LEFT_ANKLE"].x - points["RIGHT_ANKLE"].x) / shoulder_width
        wrists_above = [
            points[wrist].y < points[shoulder].y
            for wrist, shoulder in (("LEFT_WRIST", "LEFT_SHOULDER"), ("RIGHT_WRIST", "RIGHT_SHOULDER"))
        ]
        if all(wrists_above) and feet_gap > 1.6:
            self._open = True
        elif self._open and not any(wrists_above) and feet_gap < 1.2:
            self._open = False
            self.count += 1
            self.last_rep_time = timestamp

    def is_active(self, timestamp):
        """True if a rep finished recently, i.e. the person is doing jumping jacks."""
        return self.last_rep_time is not None and timestamp - self.last_rep_time < self.active_seconds


class ActivityRecognizer:
    """Classifies each frame, smooths the result, spots gestures, counts reps and detects falls.

    It works with whatever part of the body is visible: the full body gives
    the most activities, but sitting at a desk with only the upper body in
    view is recognized too. If a trained model is given, its prediction is
    used whenever it is confident enough. Otherwise the rules decide.
    """

    def __init__(self, model=None, min_model_confidence=0.6, smoothing_frames=8, walking_frames=8):
        self.model = model
        self.min_model_confidence = min_model_confidence
        self.features = FeatureExtractor()
        self.fall_detector = FallDetector()
        self.jumping_jacks = JumpingJackCounter()
        self.waves = WaveDetector()
        self.claps = ClapDetector()
        self.recent_labels = deque(maxlen=smoothing_frames)
        self.ankle_history = deque(maxlen=walking_frames)
        self.squat_count = 0
        self.knee_angle = None
        self.fall_alert = False
        self.decided_by = "Rules"
        self.view = "-"
        self._in_squat = False

    def reset(self):
        self.features.reset()
        self.fall_detector.reset()
        self.jumping_jacks.reset()
        self.waves.reset()
        self.claps.reset()
        self.recent_labels.clear()
        self.ankle_history.clear()
        self.squat_count = 0
        self.knee_angle = None
        self.fall_alert = False
        self._in_squat = False

    @property
    def fall_count(self):
        return self.fall_detector.fall_count

    @property
    def jumping_jack_count(self):
        return self.jumping_jacks.count

    @property
    def waving_hand(self):
        """(x, y) of the waving hand in 0..1 image coordinates."""
        return self.waves.wrist

    def update(self, landmarks, timestamp):
        """Return (activity, confidence) for one frame.

        The activity is the most common label of the last few frames, which
        stops the text from flickering. Confidence is the share of those
        frames that agree (0.0 - 1.0). Moving gestures (waving, clapping,
        jumping jacks) are checked over time and take priority.
        """
        self.knee_angle = None
        self.decided_by = "Rules"
        body = None
        waving = clapping = False

        if landmarks is None:
            raw_label = NO_PERSON
            self.view = "-"
            self.features.reset()
            self.ankle_history.clear()
        else:
            points = {name: landmarks[POSE[name].value] for name in BODY_PARTS}
            self.view = describe_view(points)
            body = measure_body(points)
            raw_label = self._classify_with_rules(points, body)
            if raw_label not in ("Standing", "Walking"):
                self.ankle_history.clear()

            features = self.features.update(landmarks)
            if self.model is not None and features is not None:
                model_label, probability = self.model.predict(features)
                if probability >= self.min_model_confidence:
                    raw_label = model_label
                    self.decided_by = "Model"

            width = shoulder_width(points)
            self.jumping_jacks.update(points, timestamp)
            waving = self.waves.update(points, timestamp, width)
            clapping = self.claps.update(points, timestamp, width)

        self.fall_alert = self.fall_detector.update(timestamp, body)

        self.recent_labels.append(raw_label)
        label, votes = Counter(self.recent_labels).most_common(1)[0]
        confidence = votes / len(self.recent_labels)
        if self.jumping_jacks.is_active(timestamp):
            label, confidence = JUMPING_JACKS, 1.0
        elif clapping:
            label, confidence = CLAPPING, 1.0
        elif waving:
            label, confidence = WAVING, 1.0
        self._count_squats(label)
        return label, confidence

    def _count_squats(self, label):
        # One rep = going down into a squat and standing back up.
        if label == "Squatting":
            self._in_squat = True
        elif label == "Standing" and self._in_squat:
            self._in_squat = False
            self.squat_count += 1

    def _classify_with_rules(self, points, body):
        def visible(*names):
            return all(is_visible(points[name]) for name in names)

        # --- Hands (work with just the upper body in view) ---
        width = shoulder_width(points)
        nose = points["NOSE"]
        # Hand close to the nose and not above it (a hand above the head is "raised").
        if width is not None and visible("NOSE") and any(
            visible(wrist)
            and points[wrist].y > nose.y - 0.1 * width
            and math.dist((points[wrist].x, points[wrist].y), (nose.x, nose.y)) < 0.6 * width
            for wrist in ("LEFT_WRIST", "RIGHT_WRIST")
        ):
            return TOUCHING_FACE

        hands_up = sum(
            visible(wrist, shoulder) and points[wrist].y < points[shoulder].y
            for wrist, shoulder in (
                ("LEFT_WRIST", "LEFT_SHOULDER"),
                ("RIGHT_WRIST", "RIGHT_SHOULDER"),
            )
        )
        if hands_up == 2:
            return "Both hands raised"
        if hands_up == 1:
            return "One hand raised"

        # --- Only the head and shoulders are in view, e.g. at a desk ---
        if body is None:
            if visible("NOSE") or visible("LEFT_SHOULDER") or visible("RIGHT_SHOULDER"):
                return "Sitting"
            return PARTLY_VISIBLE

        torso_angle = body["torso_angle"]
        torso_length = body["torso_length"]
        if torso_angle >= 60:
            return "Lying down"

        legs = [
            (points[hip], points[knee], points[ankle])
            for hip, knee, ankle in (
                ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
                ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
            )
            if visible(hip, knee, ankle)
        ]

        # --- Hips visible but feet are not: decide from the thighs if we can ---
        if not legs:
            thighs = [
                (points[hip], points[knee])
                for hip, knee in (("LEFT_HIP", "LEFT_KNEE"), ("RIGHT_HIP", "RIGHT_KNEE"))
                if visible(hip, knee)
            ]
            if not thighs:
                return "Sitting"  # legs hidden, e.g. under a desk
            # A vertical thigh means standing, a level (or foreshortened) thigh means sitting.
            thigh_drop = max(knee.y - hip.y for hip, knee in thighs)
            return "Standing" if thigh_drop > 0.5 * torso_length else "Sitting"

        # --- Full body in view ---
        knee_angles = [angle(hip, knee, ankle) for hip, knee, ankle in legs]
        self.knee_angle = min(knee_angles)

        if torso_angle < 35 and any(
            90 < knee_angle < 140 and abs(knee.y - hip.y) < torso_length * 0.35
            for (hip, knee, _), knee_angle in zip(legs, knee_angles)
        ):
            return "Sitting"
        if len(legs) == 2 and all(knee_angle < 120 for knee_angle in knee_angles):
            return "Squatting"
        if torso_angle >= 35 and all(knee_angle > 150 for knee_angle in knee_angles):
            return "Bending"

        if visible("LEFT_ANKLE", "RIGHT_ANKLE"):
            self.ankle_history.append(
                (
                    (points["LEFT_ANKLE"].x, points["LEFT_ANKLE"].y),
                    (points["RIGHT_ANKLE"].x, points["RIGHT_ANKLE"].y),
                )
            )
        else:
            self.ankle_history.clear()

        if len(self.ankle_history) == self.ankle_history.maxlen:
            movement = sum(
                math.dist(self.ankle_history[i][side], self.ankle_history[i - 1][side])
                for i in range(1, len(self.ankle_history))
                for side in range(2)
            )
            if movement / torso_length > 0.45:
                return "Walking"
        return "Standing"
