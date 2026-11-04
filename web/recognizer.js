// Activity recognition rules, gestures, rep counters and fall detection.
// A JavaScript port of activity.py, gestures.py and pose_utils.py, so the
// browser version behaves like the Python app.

export const NO_PERSON = "No person detected";
export const PARTLY_VISIBLE = "Person partly visible";
export const JUMPING_JACKS = "Jumping jacks";
export const WAVING = "Waving - Hi!";
export const CLAPPING = "Clapping";
export const TOUCHING_FACE = "Touching face";

// MediaPipe Pose landmark numbers (the same in Python and JavaScript).
export const POSE = {
  NOSE: 0,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
};

const MIN_VISIBILITY = 0.5;

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const xy = (p) => [p.x, p.y];

export function isVisible(landmark) {
  return (
    landmark !== undefined &&
    landmark.visibility >= MIN_VISIBILITY &&
    landmark.x >= 0 && landmark.x <= 1 &&
    landmark.y >= 0 && landmark.y <= 1
  );
}

/** Angle ABC in degrees. */
export function angle(a, b, c) {
  const ba = [a.x - b.x, a.y - b.y];
  const bc = [c.x - b.x, c.y - b.y];
  const lengths = Math.hypot(...ba) * Math.hypot(...bc);
  if (lengths === 0) return 180;
  const cosine = Math.max(-1, Math.min(1, (ba[0] * bc[0] + ba[1] * bc[1]) / lengths));
  return (Math.acos(cosine) * 180) / Math.PI;
}

function center(landmarks) {
  return [
    landmarks.reduce((sum, p) => sum + p.x, 0) / landmarks.length,
    landmarks.reduce((sum, p) => sum + p.y, 0) / landmarks.length,
  ];
}

/** Pick the named landmarks out of MediaPipe's list of 33. */
export function namedPoints(landmarks) {
  const points = {};
  for (const [name, index] of Object.entries(POSE)) points[name] = landmarks[index];
  return points;
}

export function shoulderWidth(points) {
  const left = points.LEFT_SHOULDER;
  const right = points.RIGHT_SHOULDER;
  if (!isVisible(left) || !isVisible(right)) return null;
  const width = dist(xy(left), xy(right));
  return width > 0.02 ? width : null;
}

/** Hip height, torso length and torso tilt, or null if not visible. */
export function measureBody(points) {
  const shoulders = ["LEFT_SHOULDER", "RIGHT_SHOULDER"].map((n) => points[n]).filter(isVisible);
  const hips = ["LEFT_HIP", "RIGHT_HIP"].map((n) => points[n]).filter(isVisible);
  if (!shoulders.length || !hips.length) return null;

  const shoulderCenter = center(shoulders);
  const hipCenter = center(hips);
  const torsoLength = dist(shoulderCenter, hipCenter);
  if (torsoLength === 0) return null;

  // 0 degrees = upright torso, 90 degrees = horizontal torso.
  const torsoAngle =
    (Math.atan2(
      Math.abs(shoulderCenter[0] - hipCenter[0]),
      Math.abs(shoulderCenter[1] - hipCenter[1]),
    ) * 180) / Math.PI;
  return { hipY: hipCenter[1], torsoLength, torsoAngle };
}

/** How much of the person the camera can see. */
export function describeView(points) {
  const any = (...names) => names.some((n) => isVisible(points[n]));
  if (any("LEFT_ANKLE", "RIGHT_ANKLE")) return "Full body";
  if (any("LEFT_KNEE", "RIGHT_KNEE")) return "Down to knees";
  if (any("LEFT_SHOULDER", "RIGHT_SHOULDER")) return "Upper body";
  if (isVisible(points.NOSE)) return "Face only";
  return "Partial";
}

/** A hand at shoulder height or above that swings side to side is a wave. */
export class WaveDetector {
  constructor({ windowSeconds = 1.5, minSwing = 0.25, minReversals = 2, holdSeconds = 1.0 } = {}) {
    Object.assign(this, { windowSeconds, minSwing, minReversals, holdSeconds });
    this.reset();
  }

  reset() {
    this.history = { LEFT: [], RIGHT: [] }; // [timestamp, wrist x]
    this.activeUntil = -1;
    this.wrist = null; // [x, y] of the waving hand, for drawing "Hi!"
  }

  update(points, timestamp, width) {
    if (width !== null) {
      for (const side of ["LEFT", "RIGHT"]) {
        const history = this.history[side];
        const wrist = points[`${side}_WRIST`];
        const shoulder = points[`${side}_SHOULDER`];
        // Hands at chest height moving sideways are usually clapping, not waving.
        const handUp = isVisible(wrist) && wrist.y < shoulder.y + 0.25 * width;
        if (!handUp) {
          history.length = 0;
          continue;
        }
        history.push([timestamp, wrist.x / width]);
        while (timestamp - history[0][0] > this.windowSeconds) history.shift();
        if (this.countReversals(history.map(([, x]) => x)) >= this.minReversals) {
          this.activeUntil = timestamp + this.holdSeconds;
          this.wrist = [wrist.x, wrist.y];
        }
      }
    }
    return timestamp < this.activeUntil;
  }

  /** How often the hand changed direction by at least minSwing. */
  countReversals(xs) {
    let extreme = xs[0];
    let direction = 0;
    let reversals = 0;
    for (const x of xs.slice(1)) {
      if (direction === 0) {
        if (Math.abs(x - extreme) >= this.minSwing) {
          direction = x > extreme ? 1 : -1;
          extreme = x;
        }
      } else if ((x - extreme) * direction > 0) {
        extreme = x; // still moving the same way
      } else if (Math.abs(x - extreme) >= this.minSwing) {
        reversals += 1;
        direction = -direction;
        extreme = x;
      }
    }
    return reversals;
  }
}

/** Hands coming together and apart again, at least twice in two seconds. */
export class ClapDetector {
  constructor({ windowSeconds = 2.0, minClaps = 2, holdSeconds = 1.0 } = {}) {
    Object.assign(this, { windowSeconds, minClaps, holdSeconds });
    this.reset();
  }

  reset() {
    this.clapTimes = [];
    this.activeUntil = -1;
    this.handsApart = false;
  }

  update(points, timestamp, width) {
    const { LEFT_WRIST: left, RIGHT_WRIST: right, NOSE: nose } = points;
    if (width !== null && isVisible(left) && isVisible(right)) {
      const gap = dist(xy(left), xy(right)) / width;
      // Hands meeting above the head (e.g. jumping jacks) is not a clap.
      const belowFace = !isVisible(nose) || Math.min(left.y, right.y) > nose.y;
      if (gap > 0.8) {
        this.handsApart = true;
      } else if (gap < 0.35 && this.handsApart && belowFace) {
        this.handsApart = false;
        this.clapTimes.push(timestamp);
        while (timestamp - this.clapTimes[0] > this.windowSeconds) this.clapTimes.shift();
        if (this.clapTimes.length >= this.minClaps) this.activeUntil = timestamp + this.holdSeconds;
      }
    }
    return timestamp < this.activeUntil;
  }
}

/** Flags a fall: the hips drop fast and the body ends up horizontal. */
export class FallDetector {
  constructor({ windowSeconds = 1.0, dropRatio = 0.6, alertSeconds = 3.0 } = {}) {
    Object.assign(this, { windowSeconds, dropRatio, alertSeconds });
    this.reset();
  }

  reset() {
    this.history = []; // [timestamp, hipY, torsoLength]
    this.fallCount = 0;
    this.alertUntil = -1;
  }

  update(timestamp, body) {
    if (body !== null) {
      this.history.push([timestamp, body.hipY, body.torsoLength]);
      while (timestamp - this.history[0][0] > this.windowSeconds) this.history.shift();

      const highestHip = Math.min(...this.history.map(([, y]) => y)); // smaller y = higher
      const torsoLength = Math.max(...this.history.map(([, , length]) => length));
      if (body.torsoAngle >= 60 && body.hipY - highestHip > this.dropRatio * torsoLength) {
        this.fallCount += 1;
        this.alertUntil = timestamp + this.alertSeconds;
        this.history = [];
      }
    }
    return timestamp < this.alertUntil;
  }
}

/** Counts jumping jacks: arms up with feet apart, then arms down with feet together. */
export class JumpingJackCounter {
  constructor({ activeSeconds = 2.0 } = {}) {
    this.activeSeconds = activeSeconds;
    this.reset();
  }

  reset() {
    this.count = 0;
    this.lastRepTime = null;
    this.open = false;
  }

  update(points, timestamp) {
    const names = ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_ANKLE", "RIGHT_ANKLE"];
    if (!names.every((n) => isVisible(points[n]))) return;
    const width = Math.abs(points.LEFT_SHOULDER.x - points.RIGHT_SHOULDER.x);
    if (width === 0) return;

    const feetGap = Math.abs(points.LEFT_ANKLE.x - points.RIGHT_ANKLE.x) / width;
    const wristsAbove = [
      points.LEFT_WRIST.y < points.LEFT_SHOULDER.y,
      points.RIGHT_WRIST.y < points.RIGHT_SHOULDER.y,
    ];
    if (wristsAbove.every(Boolean) && feetGap > 1.6) {
      this.open = true;
    } else if (this.open && !wristsAbove.some(Boolean) && feetGap < 1.2) {
      this.open = false;
      this.count += 1;
      this.lastRepTime = timestamp;
    }
  }

  isActive(timestamp) {
    return this.lastRepTime !== null && timestamp - this.lastRepTime < this.activeSeconds;
  }
}

/** Classifies each frame, smooths the result, spots gestures, counts reps and detects falls. */
export class ActivityRecognizer {
  constructor({ smoothingFrames = 8, walkingFrames = 8 } = {}) {
    this.smoothingFrames = smoothingFrames;
    this.walkingFrames = walkingFrames;
    this.fallDetector = new FallDetector();
    this.jumpingJacks = new JumpingJackCounter();
    this.waves = new WaveDetector();
    this.claps = new ClapDetector();
    this.reset();
  }

  reset() {
    this.fallDetector.reset();
    this.jumpingJacks.reset();
    this.waves.reset();
    this.claps.reset();
    this.recentLabels = [];
    this.ankleHistory = [];
    this.squatCount = 0;
    this.kneeAngle = null;
    this.fallAlert = false;
    this.view = "-";
    this.inSquat = false;
  }

  get fallCount() { return this.fallDetector.fallCount; }
  get jumpingJackCount() { return this.jumpingJacks.count; }
  get wavingHand() { return this.waves.wrist; }

  /**
   * Returns {activity, confidence} for one frame. `landmarks` is MediaPipe's
   * list of 33 points (or null when nobody is in view), `timestamp` in seconds.
   */
  update(landmarks, timestamp) {
    this.kneeAngle = null;
    let body = null;
    let rawLabel;
    let waving = false;
    let clapping = false;

    if (!landmarks) {
      rawLabel = NO_PERSON;
      this.view = "-";
      this.ankleHistory = [];
    } else {
      const points = namedPoints(landmarks);
      this.view = describeView(points);
      body = measureBody(points);
      rawLabel = this.classifyWithRules(points, body);
      if (rawLabel !== "Standing" && rawLabel !== "Walking") this.ankleHistory = [];

      const width = shoulderWidth(points);
      this.jumpingJacks.update(points, timestamp);
      waving = this.waves.update(points, timestamp, width);
      clapping = this.claps.update(points, timestamp, width);
    }

    this.fallAlert = this.fallDetector.update(timestamp, body);

    // Majority vote over the last few frames stops the label from flickering.
    this.recentLabels.push(rawLabel);
    if (this.recentLabels.length > this.smoothingFrames) this.recentLabels.shift();
    const votes = new Map();
    for (const label of this.recentLabels) votes.set(label, (votes.get(label) || 0) + 1);
    let activity = rawLabel;
    let best = 0;
    for (const [label, count] of votes) {
      if (count > best) [activity, best] = [label, count];
    }
    let confidence = best / this.recentLabels.length;

    if (this.jumpingJacks.isActive(timestamp)) [activity, confidence] = [JUMPING_JACKS, 1];
    else if (clapping) [activity, confidence] = [CLAPPING, 1];
    else if (waving) [activity, confidence] = [WAVING, 1];

    this.countSquats(activity);
    return { activity, confidence };
  }

  countSquats(label) {
    // One rep = going down into a squat and standing back up.
    if (label === "Squatting") {
      this.inSquat = true;
    } else if (label === "Standing" && this.inSquat) {
      this.inSquat = false;
      this.squatCount += 1;
    }
  }

  classifyWithRules(points, body) {
    const visible = (...names) => names.every((n) => isVisible(points[n]));

    // --- Hands (work with just the upper body in view) ---
    const width = shoulderWidth(points);
    const nose = points.NOSE;
    // Hand close to the nose and not above it (a hand above the head is "raised").
    if (
      width !== null && visible("NOSE") &&
      ["LEFT_WRIST", "RIGHT_WRIST"].some(
        (w) =>
          visible(w) &&
          points[w].y > nose.y - 0.1 * width &&
          dist(xy(points[w]), xy(nose)) < 0.6 * width,
      )
    ) {
      return TOUCHING_FACE;
    }

    const handsUp = [
      ["LEFT_WRIST", "LEFT_SHOULDER"],
      ["RIGHT_WRIST", "RIGHT_SHOULDER"],
    ].filter(([w, s]) => visible(w, s) && points[w].y < points[s].y).length;
    if (handsUp === 2) return "Both hands raised";
    if (handsUp === 1) return "One hand raised";

    // --- Only the head and shoulders are in view, e.g. at a desk ---
    if (body === null) {
      if (visible("NOSE") || visible("LEFT_SHOULDER") || visible("RIGHT_SHOULDER")) return "Sitting";
      return PARTLY_VISIBLE;
    }

    const { torsoAngle, torsoLength } = body;
    if (torsoAngle >= 60) return "Lying down";

    const legs = [
      ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
      ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"],
    ]
      .filter((names) => visible(...names))
      .map((names) => names.map((n) => points[n]));

    // --- Hips visible but feet are not: decide from the thighs if we can ---
    if (!legs.length) {
      const thighs = [
        ["LEFT_HIP", "LEFT_KNEE"],
        ["RIGHT_HIP", "RIGHT_KNEE"],
      ]
        .filter((names) => visible(...names))
        .map(([hip, knee]) => [points[hip], points[knee]]);
      if (!thighs.length) return "Sitting"; // legs hidden, e.g. under a desk
      // A vertical thigh means standing, a level (or foreshortened) thigh means sitting.
      const thighDrop = Math.max(...thighs.map(([hip, knee]) => knee.y - hip.y));
      return thighDrop > 0.5 * torsoLength ? "Standing" : "Sitting";
    }

    // --- Full body in view ---
    const kneeAngles = legs.map(([hip, knee, ankle]) => angle(hip, knee, ankle));
    this.kneeAngle = Math.min(...kneeAngles);

    if (
      torsoAngle < 35 &&
      legs.some(
        ([hip, knee], i) =>
          kneeAngles[i] > 90 && kneeAngles[i] < 140 && Math.abs(knee.y - hip.y) < torsoLength * 0.35,
      )
    ) {
      return "Sitting";
    }
    if (legs.length === 2 && kneeAngles.every((a) => a < 120)) return "Squatting";
    if (torsoAngle >= 35 && kneeAngles.every((a) => a > 150)) return "Bending";

    if (visible("LEFT_ANKLE", "RIGHT_ANKLE")) {
      this.ankleHistory.push([xy(points.LEFT_ANKLE), xy(points.RIGHT_ANKLE)]);
      if (this.ankleHistory.length > this.walkingFrames) this.ankleHistory.shift();
    } else {
      this.ankleHistory = [];
    }

    if (this.ankleHistory.length === this.walkingFrames) {
      let movement = 0;
      for (let i = 1; i < this.ankleHistory.length; i++) {
        for (const side of [0, 1]) movement += dist(this.ankleHistory[i][side], this.ankleHistory[i - 1][side]);
      }
      if (movement / torsoLength > 0.45) return "Walking";
    }
    return "Standing";
  }
}
