"""Upper-body gestures that need several frames: waving and clapping.

Distances are divided by the shoulder width, so the thresholds work at any
distance from the camera.
"""

import math
from collections import deque

from pose_utils import is_visible


def shoulder_width(points):
    """Distance between the shoulders, or None if they are not both visible."""
    left, right = points["LEFT_SHOULDER"], points["RIGHT_SHOULDER"]
    if not (is_visible(left) and is_visible(right)):
        return None
    width = math.dist((left.x, left.y), (right.x, right.y))
    return width if width > 0.02 else None


class WaveDetector:
    """A hand at shoulder height or above that swings side to side is a wave."""

    def __init__(self, window_seconds=1.5, min_swing=0.25, min_reversals=2, hold_seconds=1.0):
        self.window_seconds = window_seconds
        self.min_swing = min_swing
        self.min_reversals = min_reversals
        self.hold_seconds = hold_seconds
        self.history = {"LEFT": deque(), "RIGHT": deque()}  # (timestamp, wrist x)
        self.active_until = -1.0
        self.wrist = None  # (x, y) of the waving hand, for drawing "Hi!"

    def reset(self):
        for history in self.history.values():
            history.clear()
        self.active_until = -1.0
        self.wrist = None

    def update(self, points, timestamp, width):
        """Return True while the person is waving."""
        if width is not None:
            for side, history in self.history.items():
                wrist = points[f"{side}_WRIST"]
                shoulder = points[f"{side}_SHOULDER"]
                # Hands at chest height moving sideways are usually clapping, not waving.
                hand_up = is_visible(wrist) and wrist.y < shoulder.y + 0.25 * width
                if not hand_up:
                    history.clear()
                    continue

                history.append((timestamp, wrist.x / width))
                while timestamp - history[0][0] > self.window_seconds:
                    history.popleft()
                if self._count_reversals([x for _, x in history]) >= self.min_reversals:
                    self.active_until = timestamp + self.hold_seconds
                    self.wrist = (wrist.x, wrist.y)
        return timestamp < self.active_until

    def _count_reversals(self, xs):
        """Count how often the hand changed direction by at least min_swing."""
        extreme, direction, reversals = xs[0], 0, 0
        for x in xs[1:]:
            if direction == 0:
                if abs(x - extreme) >= self.min_swing:
                    direction = 1 if x > extreme else -1
                    extreme = x
            elif (x - extreme) * direction > 0:
                extreme = x  # still moving the same way
            elif abs(x - extreme) >= self.min_swing:
                reversals += 1
                direction = -direction
                extreme = x
        return reversals


class ClapDetector:
    """Hands coming together and apart again, at least twice in two seconds."""

    def __init__(self, window_seconds=2.0, min_claps=2, hold_seconds=1.0):
        self.window_seconds = window_seconds
        self.min_claps = min_claps
        self.hold_seconds = hold_seconds
        self.clap_times = deque()
        self.active_until = -1.0
        self._hands_apart = False

    def reset(self):
        self.clap_times.clear()
        self.active_until = -1.0
        self._hands_apart = False

    def update(self, points, timestamp, width):
        """Return True while the person is clapping."""
        left, right, nose = points["LEFT_WRIST"], points["RIGHT_WRIST"], points["NOSE"]
        if width is not None and is_visible(left) and is_visible(right):
            gap = math.dist((left.x, left.y), (right.x, right.y)) / width
            # Hands meeting above the head (e.g. jumping jacks) is not a clap.
            below_face = not is_visible(nose) or min(left.y, right.y) > nose.y
            if gap > 0.8:
                self._hands_apart = True
            elif gap < 0.35 and self._hands_apart and below_face:
                self._hands_apart = False
                self.clap_times.append(timestamp)
                while timestamp - self.clap_times[0] > self.window_seconds:
                    self.clap_times.popleft()
                if len(self.clap_times) >= self.min_claps:
                    self.active_until = timestamp + self.hold_seconds
        return timestamp < self.active_until
