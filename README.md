# Human Activity Recognition

Real-time human activity recognition from a webcam or a video file, using
MediaPipe Pose. It recognizes what you are doing from **whatever part of your
body the camera can see**: sitting at a laptop showing only your head and
shoulders, or standing back with your full body in view.

It combines two approaches:

1. **Hand-written rules** based on joint angles and movement, which work
   straight away with no training.
2. **A machine-learning model** that you train on your own recordings. It
   compares three classifiers and keeps the best one. It can also learn new
   activities that the rules don't know, such as push-ups.

On top of that it recognizes gestures (waving says **"Hi!"**), counts exercise
reps, detects falls and writes a report for every session.

## What it recognizes

| Camera sees | Activities |
|-------------|------------|
| Head and shoulders (e.g. at a desk) | Sitting, Waving (shows "Hi!"), Clapping, Touching face, One/Both hands raised |
| Down to the knees | All of the above, plus Standing vs Sitting (from the thigh angle) and Lying down |
| Full body | All of the above, plus Walking, Squatting, Bending, Jumping jacks, and fall detection |

## Features

| Feature | Details |
|---------|---------|
| Works with any view | Uses whatever body parts are visible; the panel shows the current view |
| Gestures | Waving (with a "Hi!" speech bubble), clapping, touching face |
| Activities (ML) | Any activity you record with `collect.py` |
| Hybrid decision | Uses the ML model when it is at least 60% sure, otherwise falls back to the rules |
| Stable labels | Majority vote over the last 8 frames, so the label does not flicker |
| Rep counters | Squats and jumping jacks |
| Fall detection | Red on-screen alert and a warning sound when the hips drop fast and the body ends up horizontal |
| Big display | HD camera resolution, a window that fills most of the screen, F for fullscreen |
| Live info panel | Activity, confidence, time in activity, view, model or rules, reps, knee angle, falls, FPS |
| Session report | Timeline CSV and a time-per-activity chart, saved when you quit |
| Inputs and outputs | Webcam or video file; optional annotated video recording and screenshots |
| Unit tests | 18 tests using hand-made poses, so no camera is needed |

## Project structure

| File | What it does |
|------|--------------|
| `main.py` | Live app: camera/video loop, window, info panel, keys, fall alert, session report |
| `activity.py` | `ActivityRecognizer` (rules + model + smoothing), `FallDetector`, `JumpingJackCounter` |
| `gestures.py` | `WaveDetector` and `ClapDetector` (gestures that need several frames) |
| `features.py` | Turns pose landmarks into 36 features for ML |
| `pose_utils.py` | Shared geometry helpers (angles, visibility) |
| `collect.py` | Records labelled training data from the webcam or a video |
| `train.py` | Trains and compares 3 models, saves the best, writes a report and confusion matrix |
| `model.py` | Loads the trained model and makes predictions |
| `tests/` | Unit tests |

Folders created while running: `data/` (training samples), `models/` (trained
model and report), `output/` (session reports, screenshots, videos).

## Setup (Windows)

1. Install Python 3.9 - 3.12.
2. Open PowerShell in this folder:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

## 1. Run the app

```powershell
python main.py                            # webcam
python main.py --fullscreen               # start in fullscreen
python main.py --source my_video.mp4      # analyze a video file
python main.py --source 1                 # second webcam
python main.py --save-video               # also record the annotated video
python main.py --rules-only               # ignore the trained model
```

If `models/activity_model.joblib` exists it is loaded automatically.
Otherwise the rules are used.

| Key | Action |
|-----|--------|
| Q | Quit and save the session report |
| F | Switch fullscreen on/off |
| R | Reset counters |
| S | Save a screenshot |

**Try it at your desk:** wave at the camera (hand at shoulder height or
higher, moving side to side) and a "Hi!" bubble appears next to your hand.
Clap twice, or rest your chin on your hand.

## 2. Train your own model (optional)

**Record data.** Run `python collect.py`, press the number key of an
activity, then **SPACE**. You get a 3 second countdown, then every frame is
recorded until you press SPACE again. Record each activity in **2 or more
separate clips** of about 20-30 seconds, moving naturally. Recording needs
your shoulders and hips in view.

```powershell
python collect.py                                        # default activities
python collect.py --labels "Standing,Push-up,Stretching" # your own activities
python collect.py --source walk.mp4 --label Walking      # label a whole video
```

**Train.**

```powershell
python train.py
```

This trains a **Random Forest**, an **SVM** and a **neural network (MLP)**,
tests each one on recordings it has never seen, and prints a comparison:

```
Model                     Accuracy  Macro F1
Random Forest                93.2%     0.932
SVM (RBF kernel)             94.4%     0.945
Neural network (MLP)         95.3%     0.953
```

The best model is retrained on all data and saved to
`models/activity_model.joblib`, together with `models/report.txt` and
`models/confusion_matrix.png`. (The numbers above come from a synthetic test
dataset. Your results depend on your recordings.)

**Why test on separate clips?** Neighbouring video frames are almost
identical. If the test frames were picked at random, the model would be tested
on near-copies of its training frames and the score would look better than it
really is. So `train.py` holds out the last clip of each activity, or the last
20% of the clip if there is only one.

## How it works

**Pose estimation.** MediaPipe Pose finds 33 body landmarks in every frame,
each with a visibility score. A landmark only counts if it is inside the frame
and visible, so the app knows which body parts it can rely on.

**Rules, depending on what is visible.**

| Activity | Rule (simplified) |
|----------|-------------------|
| Touching face | A wrist close to the nose, at nose height or lower |
| Hands raised | Wrist above its shoulder |
| Sitting (upper body only) | Shoulders/face visible but hips or legs hidden, as at a desk |
| Standing / Sitting (down to knees) | Thigh pointing down = standing, thigh level = sitting |
| Lying down | Torso tilted 60 degrees or more from vertical |
| Sitting (full body) | Upright torso, knee bent 90-140 degrees, thigh roughly level |
| Squatting | Both knees bent below 120 degrees |
| Bending | Torso tilted 35-60 degrees, legs straight |
| Walking | Ankles moved a lot over the last 8 frames |
| Standing (full body) | None of the above |

**Gestures (checked over time, they take priority).**

| Gesture | How it is detected |
|---------|--------------------|
| Waving - Hi! | A hand at shoulder height or above changes direction sideways at least twice within 1.5 seconds |
| Clapping | The hands come together and apart twice within 2 seconds, below the face |
| Jumping jacks | Arms up with feet apart, then arms down with feet together |

All distances are divided by the shoulder width or torso length, so the
thresholds work whether you are close to the camera or far away.

**Features for ML (36 values).**
- Positions of 12 joints (shoulders, elbows, wrists, hips, knees, ankles),
  measured from the hip center and divided by torso length. The same pose
  therefore gives the same numbers wherever you stand.
- 6 joint angles (knees, hips, elbows) and the torso tilt.
- Motion: average speed of the ankles and wrists, and the vertical speed of the
  hips over the last 8 frames. This is what lets the model see walking.

**Fall detection.** If, within 1 second, the hips drop by more than 60% of the
torso length and the torso ends up tilted 60 degrees or more, it is flagged as a
fall. Lying down slowly on purpose takes longer than that, so it does not
trigger the alert.

## Run the tests

```powershell
python -m unittest discover tests
```

## Tips and limitations

- With only your upper body in view the app assumes you are **sitting**,
  since that is the usual situation at a laptop. Show your knees or full body
  if you want standing, walking, squats or fall detection.
- Use good lighting and face the camera.
- One person at a time (a MediaPipe Pose limitation).
- The thresholds are heuristics. They were checked with synthetic poses, so
  tune them for your camera if needed. They are the numbers in `activity.py`
  and `gestures.py`.
- A model trained on one person and one room works best for that person and
  room. Record more people and places to make it more general.
