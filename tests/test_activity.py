"""Tests using hand-made poses, so no camera is needed.

Run from the project folder:  python -m unittest discover tests
"""

import unittest
from types import SimpleNamespace

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from activity import (
    CLAPPING,
    JUMPING_JACKS,
    NO_PERSON,
    PARTLY_VISIBLE,
    TOUCHING_FACE,
    WAVING,
    ActivityRecognizer,
)
from features import FEATURE_NAMES, FeatureExtractor
from pose_utils import POSE
from train import split_by_clip


def make_landmarks(points):
    """33 MediaPipe-like landmarks; the named ones are visible, the rest hidden."""
    landmarks = [SimpleNamespace(x=0.5, y=0.5, visibility=0.0) for _ in range(33)]
    for name, (x, y) in points.items():
        landmarks[POSE[name].value] = SimpleNamespace(x=x, y=y, visibility=0.9)
    return landmarks


STANDING = {
    "LEFT_SHOULDER": (0.45, 0.30), "RIGHT_SHOULDER": (0.55, 0.30),
    "LEFT_ELBOW": (0.43, 0.42), "RIGHT_ELBOW": (0.57, 0.42),
    "LEFT_WRIST": (0.43, 0.52), "RIGHT_WRIST": (0.57, 0.52),
    "LEFT_HIP": (0.47, 0.55), "RIGHT_HIP": (0.53, 0.55),
    "LEFT_KNEE": (0.47, 0.72), "RIGHT_KNEE": (0.53, 0.72),
    "LEFT_ANKLE": (0.47, 0.90), "RIGHT_ANKLE": (0.53, 0.90),
}
SQUATTING = dict(
    STANDING,
    LEFT_SHOULDER=(0.45, 0.42), RIGHT_SHOULDER=(0.55, 0.42),
    LEFT_HIP=(0.47, 0.65), RIGHT_HIP=(0.53, 0.65),
    LEFT_KNEE=(0.40, 0.75), RIGHT_KNEE=(0.60, 0.75),
)
SITTING = dict(
    STANDING,
    LEFT_HIP=(0.47, 0.60), RIGHT_HIP=(0.53, 0.60),
    LEFT_KNEE=(0.40, 0.64), RIGHT_KNEE=(0.60, 0.64),
    LEFT_ANKLE=(0.42, 0.90), RIGHT_ANKLE=(0.58, 0.90),
)
BENDING = dict(
    STANDING,
    LEFT_SHOULDER=(0.64, 0.38), RIGHT_SHOULDER=(0.66, 0.38),
    LEFT_WRIST=(0.70, 0.70), RIGHT_WRIST=(0.72, 0.70),
)
LYING = {
    "LEFT_SHOULDER": (0.20, 0.85), "RIGHT_SHOULDER": (0.20, 0.87),
    "LEFT_WRIST": (0.30, 0.90), "RIGHT_WRIST": (0.30, 0.92),
    "LEFT_HIP": (0.50, 0.86), "RIGHT_HIP": (0.50, 0.88),
    "LEFT_KNEE": (0.65, 0.86), "RIGHT_KNEE": (0.65, 0.88),
    "LEFT_ANKLE": (0.80, 0.86), "RIGHT_ANKLE": (0.80, 0.88),
}
BOTH_HANDS_UP = dict(STANDING, LEFT_WRIST=(0.40, 0.10), RIGHT_WRIST=(0.60, 0.10))
ONE_HAND_UP = dict(STANDING, LEFT_WRIST=(0.40, 0.10))
JACK_OPEN = dict(
    BOTH_HANDS_UP, LEFT_ANKLE=(0.35, 0.90), RIGHT_ANKLE=(0.65, 0.90)
)
LEGS_ONLY = {k: v for k, v in STANDING.items() if "KNEE" in k or "ANKLE" in k}

# Close to a laptop camera: head, shoulders and arms only.
AT_DESK = {
    "NOSE": (0.50, 0.30),
    "LEFT_SHOULDER": (0.35, 0.60), "RIGHT_SHOULDER": (0.65, 0.60),
    "LEFT_ELBOW": (0.30, 0.90), "RIGHT_ELBOW": (0.70, 0.90),
    "LEFT_WRIST": (0.38, 0.95), "RIGHT_WRIST": (0.62, 0.95),
}
TOUCHING_FACE_POSE = dict(AT_DESK, RIGHT_WRIST=(0.55, 0.38), RIGHT_ELBOW=(0.66, 0.70))
# Hips and knees in view, feet cut off by the bottom of the frame.
STANDING_TO_KNEES = {k: v for k, v in STANDING.items() if "ANKLE" not in k}
SITTING_TO_KNEES = {k: v for k, v in SITTING.items() if "ANKLE" not in k}
HIPS_NO_LEGS = {k: v for k, v in STANDING.items() if "KNEE" not in k and "ANKLE" not in k}


def run(recognizer, pose, frames=8, start=0.0, fps=30):
    """Feed the same pose for several frames and return the last result."""
    for i in range(frames):
        result = recognizer.update(make_landmarks(pose) if pose else None, start + i / fps)
    return result


class RuleTests(unittest.TestCase):
    def test_static_poses(self):
        cases = {
            "Standing": STANDING,
            "Squatting": SQUATTING,
            "Sitting": SITTING,
            "Bending": BENDING,
            "Lying down": LYING,
            "Both hands raised": BOTH_HANDS_UP,
            "One hand raised": ONE_HAND_UP,
            PARTLY_VISIBLE: LEGS_ONLY,
            NO_PERSON: None,
        }
        for expected, pose in cases.items():
            with self.subTest(expected=expected):
                label, confidence = run(ActivityRecognizer(), pose)
                self.assertEqual(label, expected)
                self.assertEqual(confidence, 1.0)

    def test_partly_visible_body(self):
        cases = [
            ("at desk", AT_DESK, "Sitting", "Upper body"),
            ("hips but no legs", HIPS_NO_LEGS, "Sitting", "Upper body"),
            ("standing, feet cut off", STANDING_TO_KNEES, "Standing", "Down to knees"),
            ("sitting, feet cut off", SITTING_TO_KNEES, "Sitting", "Down to knees"),
            ("hand on face", TOUCHING_FACE_POSE, TOUCHING_FACE, "Upper body"),
            ("full body", STANDING, "Standing", "Full body"),
        ]
        for name, pose, expected, view in cases:
            with self.subTest(name):
                recognizer = ActivityRecognizer()
                label, _ = run(recognizer, pose)
                self.assertEqual(label, expected)
                self.assertEqual(recognizer.view, view)

    def test_walking(self):
        recognizer = ActivityRecognizer()
        for i in range(20):
            step = 0.04 if i % 2 else -0.04
            pose = dict(STANDING, LEFT_ANKLE=(0.47 + step, 0.90), RIGHT_ANKLE=(0.53 - step, 0.88))
            label, _ = recognizer.update(make_landmarks(pose), i / 30)
        self.assertEqual(label, "Walking")

    def test_smoothing_ignores_a_single_odd_frame(self):
        recognizer = ActivityRecognizer()
        run(recognizer, STANDING)
        label, confidence = recognizer.update(make_landmarks(SQUATTING), 1.0)
        self.assertEqual(label, "Standing")
        self.assertLess(confidence, 1.0)


class CounterTests(unittest.TestCase):
    def test_squat_reps(self):
        recognizer = ActivityRecognizer()
        t = 0.0
        for _ in range(3):
            for pose in (STANDING, SQUATTING):
                run(recognizer, pose, start=t)
                t += 1
        run(recognizer, STANDING, start=t)
        self.assertEqual(recognizer.squat_count, 3)

    def test_jumping_jacks(self):
        recognizer = ActivityRecognizer()
        t = 0.0
        for _ in range(4):
            for pose in (JACK_OPEN, STANDING):
                label, _ = run(recognizer, pose, frames=4, start=t)
                t += 4 / 30
        self.assertEqual(recognizer.jumping_jack_count, 4)
        self.assertEqual(label, JUMPING_JACKS)

    def test_reset(self):
        recognizer = ActivityRecognizer()
        run(recognizer, SQUATTING)
        run(recognizer, STANDING, start=1)
        recognizer.reset()
        self.assertEqual(recognizer.squat_count, 0)


class GestureTests(unittest.TestCase):
    def wave_frames(self, base, frames=40, amplitude=0.08, frames_per_swing=6):
        """Raised right hand moving left and right."""
        for i in range(frames):
            phase = (i // frames_per_swing) % 2
            x = 0.75 + (amplitude if phase else -amplitude)
            yield dict(base, RIGHT_WRIST=(x, 0.30), RIGHT_ELBOW=(0.75, 0.60))

    def test_waving_at_desk(self):
        recognizer = ActivityRecognizer()
        for i, pose in enumerate(self.wave_frames(AT_DESK)):
            label, _ = recognizer.update(make_landmarks(pose), i / 30)
        self.assertEqual(label, WAVING)
        self.assertIsNotNone(recognizer.waving_hand)

    def test_waving_stops_after_hand_goes_down(self):
        recognizer = ActivityRecognizer()
        for i, pose in enumerate(self.wave_frames(AT_DESK)):
            recognizer.update(make_landmarks(pose), i / 30)
        label, _ = run(recognizer, AT_DESK, frames=30, start=2.0)
        self.assertEqual(label, "Sitting")

    def test_still_raised_hand_is_not_a_wave(self):
        still = dict(AT_DESK, RIGHT_WRIST=(0.75, 0.22), RIGHT_ELBOW=(0.75, 0.55))
        label, _ = run(ActivityRecognizer(), still, frames=40)
        self.assertEqual(label, "One hand raised")

    def test_clapping(self):
        apart = dict(AT_DESK, LEFT_WRIST=(0.30, 0.75), RIGHT_WRIST=(0.70, 0.75))
        together = dict(AT_DESK, LEFT_WRIST=(0.49, 0.75), RIGHT_WRIST=(0.51, 0.75))
        recognizer = ActivityRecognizer()
        t = 0.0
        for _ in range(3):
            for pose in (apart, together):
                label, _ = run(recognizer, pose, frames=5, start=t)
                t += 5 / 30
        self.assertEqual(label, CLAPPING)


class FallTests(unittest.TestCase):
    def test_sudden_fall_is_detected(self):
        recognizer = ActivityRecognizer()
        run(recognizer, STANDING, frames=15)
        run(recognizer, LYING, frames=3, start=0.6)
        self.assertTrue(recognizer.fall_alert)
        self.assertEqual(recognizer.fall_count, 1)

        # The alert stays on for a while, then switches off by itself.
        run(recognizer, LYING, frames=1, start=2.0)
        self.assertTrue(recognizer.fall_alert)
        run(recognizer, LYING, frames=1, start=5.0)
        self.assertFalse(recognizer.fall_alert)
        self.assertEqual(recognizer.fall_count, 1)

    def test_lying_down_slowly_is_not_a_fall(self):
        recognizer = ActivityRecognizer()
        run(recognizer, STANDING, frames=15)
        run(recognizer, SITTING, frames=15, start=1.5)
        run(recognizer, LYING, frames=15, start=3.0)
        self.assertFalse(recognizer.fall_alert)
        self.assertEqual(recognizer.fall_count, 0)


class FeatureTests(unittest.TestCase):
    def test_length_matches_names(self):
        features = FeatureExtractor().update(make_landmarks(STANDING))
        self.assertEqual(len(features), len(FEATURE_NAMES))

    def test_same_pose_elsewhere_in_frame_gives_same_features(self):
        moved = {name: (0.2 + x * 0.6, 0.1 + y * 0.6) for name, (x, y) in STANDING.items()}
        original = FeatureExtractor().update(make_landmarks(STANDING))
        shifted = FeatureExtractor().update(make_landmarks(moved))
        np.testing.assert_allclose(original, shifted, atol=1e-5)

    def test_hidden_body_gives_no_features(self):
        self.assertIsNone(FeatureExtractor().update(make_landmarks({})))


class ModelTests(unittest.TestCase):
    def test_confident_model_overrides_rules(self):
        extractor = FeatureExtractor()
        rng = np.random.default_rng(0)
        base = extractor.update(make_landmarks(STANDING))
        samples = np.vstack([base + rng.normal(0, 0.01, base.shape) for _ in range(30)])
        other = samples + 1.0
        classifier = RandomForestClassifier(n_estimators=20, random_state=0).fit(
            np.vstack([samples, other]), ["Waving"] * 30 + ["Other"] * 30
        )

        class FakeModel:
            def predict(self, features):
                probabilities = classifier.predict_proba(features.reshape(1, -1))[0]
                return classifier.classes_[probabilities.argmax()], probabilities.max()

        recognizer = ActivityRecognizer(model=FakeModel())
        label, _ = run(recognizer, STANDING)
        self.assertEqual(label, "Waving")
        self.assertEqual(recognizer.decided_by, "Model")


class SplitTests(unittest.TestCase):
    def test_holds_out_last_clip_or_last_part(self):
        labels = np.array(["A"] * 10 + ["A"] * 5 + ["B"] * 10)
        clips = np.array(["a1"] * 10 + ["a2"] * 5 + ["b1"] * 10)
        train, test = split_by_clip(labels, clips)
        self.assertTrue(test[10:15].all())  # whole second clip of A
        self.assertFalse(test[:10].any())
        self.assertTrue(test[23:].all())  # last 20% of B's only clip
        self.assertFalse(test[15:23].any())
        self.assertTrue((train ^ test).all())


if __name__ == "__main__":
    unittest.main()
