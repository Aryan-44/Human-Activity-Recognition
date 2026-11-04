// Same hand-made poses as tests/test_activity.py, so both versions are checked
// against the same expectations. Run from the project folder:
//   node --test web/tests/

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import {
  ActivityRecognizer,
  CLAPPING,
  JUMPING_JACKS,
  NO_PERSON,
  PARTLY_VISIBLE,
  POSE,
  TOUCHING_FACE,
  WAVING,
} from "../recognizer.js";

/** 33 MediaPipe-like landmarks; the named ones are visible, the rest hidden. */
function makeLandmarks(points) {
  const landmarks = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 0 }));
  for (const [name, [x, y]] of Object.entries(points)) landmarks[POSE[name]] = { x, y, visibility: 0.9 };
  return landmarks;
}

const STANDING = {
  LEFT_SHOULDER: [0.45, 0.30], RIGHT_SHOULDER: [0.55, 0.30],
  LEFT_ELBOW: [0.43, 0.42], RIGHT_ELBOW: [0.57, 0.42],
  LEFT_WRIST: [0.43, 0.52], RIGHT_WRIST: [0.57, 0.52],
  LEFT_HIP: [0.47, 0.55], RIGHT_HIP: [0.53, 0.55],
  LEFT_KNEE: [0.47, 0.72], RIGHT_KNEE: [0.53, 0.72],
  LEFT_ANKLE: [0.47, 0.90], RIGHT_ANKLE: [0.53, 0.90],
};
const SQUATTING = {
  ...STANDING,
  LEFT_SHOULDER: [0.45, 0.42], RIGHT_SHOULDER: [0.55, 0.42],
  LEFT_HIP: [0.47, 0.65], RIGHT_HIP: [0.53, 0.65],
  LEFT_KNEE: [0.40, 0.75], RIGHT_KNEE: [0.60, 0.75],
};
const SITTING = {
  ...STANDING,
  LEFT_HIP: [0.47, 0.60], RIGHT_HIP: [0.53, 0.60],
  LEFT_KNEE: [0.40, 0.64], RIGHT_KNEE: [0.60, 0.64],
  LEFT_ANKLE: [0.42, 0.90], RIGHT_ANKLE: [0.58, 0.90],
};
const BENDING = {
  ...STANDING,
  LEFT_SHOULDER: [0.64, 0.38], RIGHT_SHOULDER: [0.66, 0.38],
  LEFT_WRIST: [0.70, 0.70], RIGHT_WRIST: [0.72, 0.70],
};
const LYING = {
  LEFT_SHOULDER: [0.20, 0.85], RIGHT_SHOULDER: [0.20, 0.87],
  LEFT_WRIST: [0.30, 0.90], RIGHT_WRIST: [0.30, 0.92],
  LEFT_HIP: [0.50, 0.86], RIGHT_HIP: [0.50, 0.88],
  LEFT_KNEE: [0.65, 0.86], RIGHT_KNEE: [0.65, 0.88],
  LEFT_ANKLE: [0.80, 0.86], RIGHT_ANKLE: [0.80, 0.88],
};
const BOTH_HANDS_UP = { ...STANDING, LEFT_WRIST: [0.40, 0.10], RIGHT_WRIST: [0.60, 0.10] };
const ONE_HAND_UP = { ...STANDING, LEFT_WRIST: [0.40, 0.10] };
const JACK_OPEN = { ...BOTH_HANDS_UP, LEFT_ANKLE: [0.35, 0.90], RIGHT_ANKLE: [0.65, 0.90] };
const pick = (pose, keep) => Object.fromEntries(Object.entries(pose).filter(([name]) => keep(name)));
const LEGS_ONLY = pick(STANDING, (n) => n.includes("KNEE") || n.includes("ANKLE"));
const AT_DESK = {
  NOSE: [0.50, 0.30],
  LEFT_SHOULDER: [0.35, 0.60], RIGHT_SHOULDER: [0.65, 0.60],
  LEFT_ELBOW: [0.30, 0.90], RIGHT_ELBOW: [0.70, 0.90],
  LEFT_WRIST: [0.38, 0.95], RIGHT_WRIST: [0.62, 0.95],
};
const TOUCHING_FACE_POSE = { ...AT_DESK, RIGHT_WRIST: [0.55, 0.38], RIGHT_ELBOW: [0.66, 0.70] };
const STANDING_TO_KNEES = pick(STANDING, (n) => !n.includes("ANKLE"));
const SITTING_TO_KNEES = pick(SITTING, (n) => !n.includes("ANKLE"));
const HIPS_NO_LEGS = pick(STANDING, (n) => !n.includes("KNEE") && !n.includes("ANKLE"));

/** Feed the same pose for several frames and return the last result. */
function run(recognizer, pose, { frames = 8, start = 0, fps = 30 } = {}) {
  let result;
  for (let i = 0; i < frames; i++) {
    result = recognizer.update(pose ? makeLandmarks(pose) : null, start + i / fps);
  }
  return result;
}

describe("rules", () => {
  const cases = [
    ["Standing", STANDING],
    ["Squatting", SQUATTING],
    ["Sitting", SITTING],
    ["Bending", BENDING],
    ["Lying down", LYING],
    ["Both hands raised", BOTH_HANDS_UP],
    ["One hand raised", ONE_HAND_UP],
    [PARTLY_VISIBLE, LEGS_ONLY],
    [NO_PERSON, null],
  ];
  for (const [expected, pose] of cases) {
    test(`static pose: ${expected}`, () => {
      const { activity, confidence } = run(new ActivityRecognizer(), pose);
      assert.equal(activity, expected);
      assert.equal(confidence, 1);
    });
  }

  const partial = [
    ["at desk", AT_DESK, "Sitting", "Upper body"],
    ["hips but no legs", HIPS_NO_LEGS, "Sitting", "Upper body"],
    ["standing, feet cut off", STANDING_TO_KNEES, "Standing", "Down to knees"],
    ["sitting, feet cut off", SITTING_TO_KNEES, "Sitting", "Down to knees"],
    ["hand on face", TOUCHING_FACE_POSE, TOUCHING_FACE, "Upper body"],
    ["full body", STANDING, "Standing", "Full body"],
  ];
  for (const [name, pose, expected, view] of partial) {
    test(`partly visible: ${name}`, () => {
      const recognizer = new ActivityRecognizer();
      assert.equal(run(recognizer, pose).activity, expected);
      assert.equal(recognizer.view, view);
    });
  }

  test("walking", () => {
    const recognizer = new ActivityRecognizer();
    let result;
    for (let i = 0; i < 20; i++) {
      const step = i % 2 ? 0.04 : -0.04;
      const pose = { ...STANDING, LEFT_ANKLE: [0.47 + step, 0.90], RIGHT_ANKLE: [0.53 - step, 0.88] };
      result = recognizer.update(makeLandmarks(pose), i / 30);
    }
    assert.equal(result.activity, "Walking");
  });

  test("smoothing ignores a single odd frame", () => {
    const recognizer = new ActivityRecognizer();
    run(recognizer, STANDING);
    const { activity, confidence } = recognizer.update(makeLandmarks(SQUATTING), 1);
    assert.equal(activity, "Standing");
    assert.ok(confidence < 1);
  });
});

describe("gestures", () => {
  function* waveFrames(base, frames = 40, amplitude = 0.08, framesPerSwing = 6) {
    for (let i = 0; i < frames; i++) {
      const x = 0.75 + (Math.floor(i / framesPerSwing) % 2 ? amplitude : -amplitude);
      yield { ...base, RIGHT_WRIST: [x, 0.30], RIGHT_ELBOW: [0.75, 0.60] };
    }
  }

  test("waving at desk says Hi", () => {
    const recognizer = new ActivityRecognizer();
    let result;
    let i = 0;
    for (const pose of waveFrames(AT_DESK)) result = recognizer.update(makeLandmarks(pose), i++ / 30);
    assert.equal(result.activity, WAVING);
    assert.ok(recognizer.wavingHand);
  });

  test("waving stops after the hand goes down", () => {
    const recognizer = new ActivityRecognizer();
    let i = 0;
    for (const pose of waveFrames(AT_DESK)) recognizer.update(makeLandmarks(pose), i++ / 30);
    assert.equal(run(recognizer, AT_DESK, { frames: 30, start: 2 }).activity, "Sitting");
  });

  test("a still raised hand is not a wave", () => {
    const still = { ...AT_DESK, RIGHT_WRIST: [0.75, 0.22], RIGHT_ELBOW: [0.75, 0.55] };
    assert.equal(run(new ActivityRecognizer(), still, { frames: 40 }).activity, "One hand raised");
  });

  test("clapping", () => {
    const apart = { ...AT_DESK, LEFT_WRIST: [0.30, 0.75], RIGHT_WRIST: [0.70, 0.75] };
    const together = { ...AT_DESK, LEFT_WRIST: [0.49, 0.75], RIGHT_WRIST: [0.51, 0.75] };
    const recognizer = new ActivityRecognizer();
    let t = 0;
    let result;
    for (let round = 0; round < 3; round++) {
      for (const pose of [apart, together]) {
        result = run(recognizer, pose, { frames: 5, start: t });
        t += 5 / 30;
      }
    }
    assert.equal(result.activity, CLAPPING);
  });
});

describe("counters", () => {
  test("squat reps", () => {
    const recognizer = new ActivityRecognizer();
    let t = 0;
    for (let rep = 0; rep < 3; rep++) {
      for (const pose of [STANDING, SQUATTING]) {
        run(recognizer, pose, { start: t });
        t += 1;
      }
    }
    run(recognizer, STANDING, { start: t });
    assert.equal(recognizer.squatCount, 3);
  });

  test("jumping jacks", () => {
    const recognizer = new ActivityRecognizer();
    let t = 0;
    let result;
    for (let rep = 0; rep < 4; rep++) {
      for (const pose of [JACK_OPEN, STANDING]) {
        result = run(recognizer, pose, { frames: 4, start: t });
        t += 4 / 30;
      }
    }
    assert.equal(recognizer.jumpingJackCount, 4);
    assert.equal(result.activity, JUMPING_JACKS);
  });

  test("reset", () => {
    const recognizer = new ActivityRecognizer();
    run(recognizer, SQUATTING);
    run(recognizer, STANDING, { start: 1 });
    recognizer.reset();
    assert.equal(recognizer.squatCount, 0);
  });
});

describe("falls", () => {
  test("a sudden fall is detected, and the alert switches off later", () => {
    const recognizer = new ActivityRecognizer();
    run(recognizer, STANDING, { frames: 15 });
    run(recognizer, LYING, { frames: 3, start: 0.6 });
    assert.ok(recognizer.fallAlert);
    assert.equal(recognizer.fallCount, 1);
    run(recognizer, LYING, { frames: 1, start: 2 });
    assert.ok(recognizer.fallAlert);
    run(recognizer, LYING, { frames: 1, start: 5 });
    assert.ok(!recognizer.fallAlert);
    assert.equal(recognizer.fallCount, 1);
  });

  test("lying down slowly is not a fall", () => {
    const recognizer = new ActivityRecognizer();
    run(recognizer, STANDING, { frames: 15 });
    run(recognizer, SITTING, { frames: 15, start: 1.5 });
    run(recognizer, LYING, { frames: 15, start: 3 });
    assert.ok(!recognizer.fallAlert);
    assert.equal(recognizer.fallCount, 0);
  });
});
