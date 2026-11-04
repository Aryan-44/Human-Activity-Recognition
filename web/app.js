// Live activity recognition in the browser. The camera, the pose model and
// the recognition all run on the visitor's device; nothing is uploaded.

import {
  FilesetResolver,
  PoseLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs";

import {
  ActivityRecognizer,
  CLAPPING,
  JUMPING_JACKS,
  NO_PERSON,
  PARTLY_VISIBLE,
  TOUCHING_FACE,
  WAVING,
  isVisible,
} from "./recognizer.js";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task";
const FALL_EVENT = "FALL DETECTED";
const NOT_ACTIVITIES = new Set([NO_PERSON, PARTLY_VISIBLE, FALL_EVENT]);

const ACTIVITY_COLORS = {
  Standing: "#4fd17a",
  Walking: "#2fc6ff",
  Sitting: "#ff9a3d",
  Squatting: "#ffcc33",
  Bending: "#c47bff",
  "Lying down": "#ff6fb1",
  "One hand raised": "#ffe14d",
  "Both hands raised": "#ffe14d",
  [JUMPING_JACKS]: "#3dffd8",
  [WAVING]: "#ffd400",
  [CLAPPING]: "#ff7af2",
  [TOUCHING_FACE]: "#ffb3b3",
  [PARTLY_VISIBLE]: "#ffa500",
  [NO_PERSON]: "#ff6b6b",
};

const $ = (id) => document.getElementById(id);
const els = {
  video: $("video"),
  canvas: $("canvas"),
  stage: $("stage"),
  chip: $("chip"),
  fallBanner: $("fallBanner"),
  startScreen: $("startScreen"),
  startCamera: $("startCamera"),
  fileButton: $("fileButton"),
  fileInput: $("fileInput"),
  status: $("status"),
  stageControls: $("stageControls"),
  activity: $("activity"),
  confidenceBar: $("confidenceBar"),
  confidenceText: $("confidenceText"),
  statTime: $("statTime"),
  statView: $("statView"),
  statSquats: $("statSquats"),
  statJacks: $("statJacks"),
  statFalls: $("statFalls"),
  statKnee: $("statKnee"),
  statSession: $("statSession"),
  statFps: $("statFps"),
  sessionBars: $("sessionBars"),
  downloadCsv: $("downloadCsv"),
};
const ctx = els.canvas.getContext("2d");

let landmarker = null;
let recognizer = new ActivityRecognizer();
let stream = null;
let running = false;
let mirrored = true;
let audio = null;

// Session state
let sessionStartMs = 0;
let lastVideoTime = -1;
let lastFrameMs = 0;
let fps = 0;
let segments = []; // [start seconds, end seconds, activity]
let currentActivity = null;
let segmentStart = 0;
let timestamp = 0;
let lastPanelUpdate = 0;

// ---------- Setup ----------

function setStatus(message, isError = false) {
  els.status.textContent = message;
  els.status.classList.toggle("error", isError);
}

async function loadModel() {
  setStatus("Downloading the pose model (about 9 MB, only the first time)…");
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  try {
    landmarker = await PoseLandmarker.createFromOptions(vision, options("GPU"));
  } catch (error) {
    console.warn("GPU not available, using CPU", error);
    landmarker = await PoseLandmarker.createFromOptions(vision, options("CPU"));
  }
  els.startCamera.disabled = false;
  els.startCamera.textContent = "Start camera";
  els.fileInput.disabled = false;
  els.fileButton.removeAttribute("aria-disabled");
  setStatus("Ready. Your browser will ask for camera permission.");
}

function beep() {
  if (!audio) return;
  const oscillator = audio.createOscillator();
  const gain = audio.createGain();
  oscillator.frequency.value = 880;
  gain.gain.setValueAtTime(0.25, audio.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.001, audio.currentTime + 0.6);
  oscillator.connect(gain).connect(audio.destination);
  oscillator.start();
  oscillator.stop(audio.currentTime + 0.6);
}

// ---------- Starting and stopping ----------

async function startCamera() {
  audio ??= new AudioContext();
  setStatus("Waiting for camera permission…");
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
  } catch (error) {
    const messages = {
      NotAllowedError: "Camera permission was blocked. Allow the camera in your browser's address bar and try again.",
      NotFoundError: "No camera was found. You can analyze a video file instead.",
      NotReadableError: "The camera is being used by another app. Close it and try again.",
    };
    setStatus(messages[error.name] || `Could not start the camera: ${error.message}`, true);
    return;
  }
  els.video.srcObject = stream;
  els.video.removeAttribute("src");
  mirrored = true;
  await begin();
}

async function startFile(file) {
  audio ??= new AudioContext();
  els.video.srcObject = null;
  els.video.src = URL.createObjectURL(file);
  mirrored = false;
  try {
    await begin();
  } catch (error) {
    console.error(error);
    setStatus("Your browser can't play this video. Try an MP4 (H.264) or WebM file.", true);
  }
}

async function begin() {
  await els.video.play();
  els.canvas.width = els.video.videoWidth;
  els.canvas.height = els.video.videoHeight;

  recognizer = new ActivityRecognizer();
  segments = [];
  currentActivity = null;
  segmentStart = 0;
  lastVideoTime = -1;
  sessionStartMs = performance.now();
  lastFrameMs = sessionStartMs;
  fps = 0;

  els.startScreen.hidden = true;
  els.stageControls.hidden = false;
  els.chip.hidden = false;
  els.downloadCsv.disabled = true;
  running = true;
  requestAnimationFrame(loop);
}

function stop() {
  if (!running) return;
  running = false;
  if (currentActivity !== null) segments.push([segmentStart, timestamp, currentActivity]);
  currentActivity = null;
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  els.video.pause();
  if (document.fullscreenElement) document.exitFullscreen();

  els.chip.hidden = true;
  els.fallBanner.hidden = true;
  els.stageControls.hidden = true;
  els.startScreen.hidden = false;
  els.startCamera.textContent = "Start again";
  setStatus("Session finished. See your time per activity on the right, or download the CSV.");
  updateSessionBars();
  els.downloadCsv.disabled = segments.length === 0;
}

// ---------- Main loop ----------

function loop() {
  if (!running) return;
  if (els.video.ended) {
    stop();
    return;
  }

  const nowMs = performance.now();
  if (els.video.currentTime !== lastVideoTime && els.video.readyState >= 2) {
    lastVideoTime = els.video.currentTime;
    processFrame(nowMs);
  }
  requestAnimationFrame(loop);
}

function processFrame(nowMs) {
  const result = landmarker.detectForVideo(els.video, nowMs);
  let landmarks = result.landmarks?.[0] ?? null;
  if (landmarks && landmarks.every((p) => !p.visibility)) {
    // Some browser builds leave visibility empty; then trust every point inside the frame.
    landmarks = landmarks.map((p) => ({ ...p, visibility: 1 }));
  }
  if (landmarks && mirrored) {
    // Mirror like a selfie camera, the same as the Python app.
    landmarks = landmarks.map((p) => ({ ...p, x: 1 - p.x }));
  }

  // Video files use their own clock so the timings match the video.
  timestamp = mirrored ? (nowMs - sessionStartMs) / 1000 : els.video.currentTime;

  const fallsBefore = recognizer.fallCount;
  const { activity, confidence } = recognizer.update(landmarks, timestamp);
  if (recognizer.fallCount > fallsBefore) {
    segments.push([timestamp, timestamp, FALL_EVENT]);
    beep();
  }

  if (activity !== currentActivity) {
    if (currentActivity !== null) segments.push([segmentStart, timestamp, currentActivity]);
    currentActivity = activity;
    segmentStart = timestamp;
  }

  fps = 0.9 * fps + 0.1 * (1000 / Math.max(nowMs - lastFrameMs, 1));
  lastFrameMs = nowMs;

  draw(landmarks, activity);
  updatePanel(activity, confidence, nowMs);
}

// ---------- Drawing on the canvas ----------

function draw(landmarks, activity) {
  const { width, height } = els.canvas;
  ctx.save();
  if (mirrored) {
    ctx.translate(width, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(els.video, 0, 0, width, height);
  ctx.restore();

  if (landmarks) drawSkeleton(landmarks, ACTIVITY_COLORS[activity] || "#ffffff");
  if (activity === WAVING && recognizer.wavingHand) drawSpeechBubble("Hi! 👋", recognizer.wavingHand);
}

function drawSkeleton(landmarks, color) {
  const { width, height } = els.canvas;
  const size = Math.max(width, height) / 640;
  ctx.lineWidth = 3 * size;
  ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
  for (const { start, end } of PoseLandmarker.POSE_CONNECTIONS) {
    const a = landmarks[start];
    const b = landmarks[end];
    if (!isVisible(a) || !isVisible(b)) continue;
    ctx.beginPath();
    ctx.moveTo(a.x * width, a.y * height);
    ctx.lineTo(b.x * width, b.y * height);
    ctx.stroke();
  }
  ctx.fillStyle = color;
  for (const point of landmarks) {
    if (!isVisible(point)) continue;
    ctx.beginPath();
    ctx.arc(point.x * width, point.y * height, 4 * size, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawSpeechBubble(message, [handX, handY]) {
  const { width, height } = els.canvas;
  const size = Math.max(width, height) / 640;
  ctx.font = `700 ${Math.round(34 * size)}px Inter, system-ui, sans-serif`;
  const pad = 14 * size;
  const boxWidth = ctx.measureText(message).width + pad * 2;
  const boxHeight = 34 * size + pad * 2;

  const hx = handX * width;
  const hy = handY * height;
  const left = Math.min(Math.max(hx + 30 * size, 0), width - boxWidth);
  const top = Math.min(Math.max(hy - boxHeight - 40 * size, 0), height - boxHeight);

  ctx.fillStyle = "#ffffff";
  ctx.strokeStyle = "#ffb000";
  ctx.lineWidth = 3 * size;
  ctx.beginPath();
  ctx.moveTo(left + pad, top + boxHeight);
  ctx.lineTo(hx, hy);
  ctx.lineTo(left + pad * 3, top + boxHeight);
  ctx.closePath();
  ctx.fill();
  ctx.beginPath();
  ctx.roundRect(left, top, boxWidth, boxHeight, 12 * size);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = "#c26a00";
  ctx.textBaseline = "middle";
  ctx.fillText(message, left + pad, top + boxHeight / 2);
}

// ---------- Side panel ----------

const formatTime = (seconds) =>
  `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;

function updatePanel(activity, confidence, nowMs) {
  const color = ACTIVITY_COLORS[activity] || "#ffffff";
  els.chip.textContent = activity;
  els.chip.style.setProperty("--chip-color", color);
  els.fallBanner.hidden = !recognizer.fallAlert;

  // The text panel does not need 30 updates per second.
  if (nowMs - lastPanelUpdate < 150) return;
  lastPanelUpdate = nowMs;

  els.activity.textContent = activity;
  els.confidenceBar.style.width = `${Math.round(confidence * 100)}%`;
  els.confidenceText.textContent = `Confidence ${Math.round(confidence * 100)}%`;
  els.statTime.textContent = `${(timestamp - segmentStart).toFixed(1)} s`;
  els.statView.textContent = recognizer.view;
  els.statSquats.textContent = recognizer.squatCount;
  els.statJacks.textContent = recognizer.jumpingJackCount;
  els.statFalls.textContent = recognizer.fallCount;
  els.statKnee.textContent = recognizer.kneeAngle === null ? "—" : `${Math.round(recognizer.kneeAngle)}°`;
  els.statSession.textContent = formatTime(timestamp);
  els.statFps.textContent = Math.round(fps);
  updateSessionBars();
}

function activityTotals() {
  const totals = new Map();
  const add = (activity, seconds) => {
    if (!NOT_ACTIVITIES.has(activity)) totals.set(activity, (totals.get(activity) || 0) + seconds);
  };
  for (const [start, end, activity] of segments) add(activity, end - start);
  if (running && currentActivity !== null) add(currentActivity, timestamp - segmentStart);
  return [...totals].filter(([, seconds]) => seconds >= 0.5).sort((a, b) => b[1] - a[1]);
}

function updateSessionBars() {
  const totals = activityTotals();
  els.sessionBars.replaceChildren();
  if (!totals.length) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = running ? "Recognized activities will appear here." : "Start the camera to begin a session.";
    els.sessionBars.append(empty);
    return;
  }
  const longest = totals[0][1];
  for (const [activity, seconds] of totals) {
    const item = document.createElement("li");
    item.innerHTML = `<span class="name"></span><span class="value"></span>
      <span class="track"><span class="fill"></span></span>`;
    item.querySelector(".name").textContent = activity;
    item.querySelector(".value").textContent = `${seconds.toFixed(1)} s`;
    item.querySelector(".fill").style.width = `${(seconds / longest) * 100}%`;
    els.sessionBars.append(item);
  }
}

function downloadCsv() {
  const rows = [["start_seconds", "end_seconds", "duration_seconds", "activity"]];
  for (const [start, end, activity] of [...segments].sort((a, b) => a[0] - b[0])) {
    rows.push([start.toFixed(2), end.toFixed(2), (end - start).toFixed(2), activity]);
  }
  const csv = rows.map((row) => row.join(",")).join("\n");
  saveFile(new Blob([csv], { type: "text/csv" }), `session_${fileStamp()}.csv`);
}

function screenshot() {
  els.canvas.toBlob((blob) => saveFile(blob, `screenshot_${fileStamp()}.png`));
}

function fileStamp() {
  return new Date().toISOString().slice(0, 19).replace(/[-:]/g, "").replace("T", "_");
}

function saveFile(blob, name) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else els.stage.requestFullscreen();
}

function reset() {
  recognizer.reset();
  updatePanel(currentActivity ?? "—", 0, Infinity);
}

// ---------- Events ----------

els.startCamera.addEventListener("click", startCamera);
els.fileInput.addEventListener("change", () => {
  const file = els.fileInput.files[0];
  if (file) startFile(file);
  els.fileInput.value = "";
});
els.stopButton = $("stopButton");
els.stopButton.addEventListener("click", stop);
$("resetButton").addEventListener("click", reset);
$("shotButton").addEventListener("click", screenshot);
$("fullButton").addEventListener("click", toggleFullscreen);
els.downloadCsv.addEventListener("click", downloadCsv);

document.addEventListener("keydown", (event) => {
  if (!running || event.ctrlKey || event.metaKey || event.altKey) return;
  const key = event.key.toLowerCase();
  if (key === "f") toggleFullscreen();
  else if (key === "r") reset();
  else if (key === "s") screenshot();
  else if (key === "escape") stop();
});

if (!navigator.mediaDevices?.getUserMedia) {
  setStatus("This browser cannot use the camera here. Open the page over https in Chrome, Edge, Firefox or Safari.", true);
}

loadModel().catch((error) => {
  console.error(error);
  els.startCamera.textContent = "Model failed to load";
  setStatus("Could not load the pose model. Check your internet connection and reload the page.", true);
});
