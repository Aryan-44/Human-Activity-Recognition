"""Small geometry helpers shared by the rules and the ML features."""

import math

import mediapipe as mp


POSE = mp.solutions.pose.PoseLandmark
MIN_VISIBILITY = 0.5


def angle(a, b, c):
    """Return the angle ABC in degrees."""
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)
    lengths = math.hypot(*ba) * math.hypot(*bc)
    if lengths == 0:
        return 180
    cosine = max(-1, min(1, (ba[0] * bc[0] + ba[1] * bc[1]) / lengths))
    return math.degrees(math.acos(cosine))


def is_visible(landmark):
    return (
        landmark.visibility >= MIN_VISIBILITY
        and 0 <= landmark.x <= 1
        and 0 <= landmark.y <= 1
    )


def center(landmarks):
    """Average (x, y) of a list of landmarks."""
    return (
        sum(p.x for p in landmarks) / len(landmarks),
        sum(p.y for p in landmarks) / len(landmarks),
    )
