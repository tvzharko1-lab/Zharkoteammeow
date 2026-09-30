import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.15/+esm";
const video = document.getElementById("webcam");
const canvas = document.getElementById("output_canvas");
const ctx = canvas.getContext("2d");
const statusDiv = document.getElementById("status");
const alertBox = document.getElementById("alert-box");
const alertText = document.getElementById("alert-text");
const headerDot = document.getElementById("header-dot");
const mainCard = document.getElementById("main-video-card");
const gestureBadge = document.getElementById("gesture-badge");
const countSlouchEl = document.getElementById("count-slouch");
const countYawnEl = document.getElementById("count-yawn");
const countDistanceEl = document.getElementById("count-distance");
const countCalibEl = document.getElementById("count-calib");
const postureStatus = document.getElementById("posture-status");
const postureBar = document.getElementById("bar-posture");
const blinkCountEl = document.getElementById("blink-count");
const calibrateBtn = document.getElementById("calibrate-btn");
const pipBtn = document.getElementById("pip-btn");
const tabBtns = document.querySelectorAll(".tab-btn");
let faceLandmarker;
let lastVideoTime = -1;
let lastLandmarks = null;
let baselineY = null;
let baselineDist = null;
let baselineFaceWidth = null;
let slouchLogs = [];
let yawnLogs = [];
let distanceLogs = [];
let calibLogs = [];
let activeTimeframe = "10m";
let isSlouchActive = false;
let isYawnActive = false;
let isTiltActive = false;
let isDistanceActive = false;
let isClosed = false;
let blinkTimestamps = [];
let pipWindow = null;
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
let lastSoundTime = 0;
function playWarningSound() {
  const now = Date.now();
  if (now - lastSoundTime < 3000) return;
  lastSoundTime = now;
  if (audioCtx.state === "suspended") {
    audioCtx.resume();
  }
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type = "sine";
  osc.frequency.setValueAtTime(520, audioCtx.currentTime);
  gain.gain.setValueAtTime(0.06, audioCtx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.35);
  osc.connect(gain);
  gain.connect(audioCtx.destination);
  osc.start();
  osc.stop(audioCtx.currentTime + 0.35);
}
function getDistance(p1, p2) {
  return Math.hypot(p1.x - p2.x, p1.y - p2.y);
}
function getEAR(eyeLandmarks) {
  const v1 = getDistance(eyeLandmarks[1], eyeLandmarks[5]);
  const v2 = getDistance(eyeLandmarks[2], eyeLandmarks[4]);
  const h = getDistance(eyeLandmarks[0], eyeLandmarks[3]);
  return (v1 + v2) / (2.0 * h);
}
function updateCountersDisplay() {
  const now = Date.now();
  let timeWindowMs = 10 * 60 * 1000;
  if (activeTimeframe === "30m") timeWindowMs = 30 * 60 * 1000;
  if (activeTimeframe === "1h")  timeWindowMs = 60 * 60 * 1000;
  const cutoff = now - timeWindowMs;
  countSlouchEl.innerText = slouchLogs.filter(t => t >= cutoff).length;
  countYawnEl.innerText = yawnLogs.filter(t => t >= cutoff).length;
  countDistanceEl.innerText = distanceLogs.filter(t => t >= cutoff).length;
  countCalibEl.innerText = calibLogs.filter(t => t >= cutoff).length;
}
tabBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    tabBtns.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    activeTimeframe = btn.dataset.time;
    updateCountersDisplay();
  });
});
function drawFaceSkeleton(landmarks, color = "#38bdf8") {
  const w = canvas.width;
  const h = canvas.height;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  const keyPoints = [1, 33, 133, 362, 263, 61, 291, 13, 14, 152, 234, 454];
  ctx.globalAlpha = 0.85;
  keyPoints.forEach(idx => {
    const p = landmarks[idx];
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 2.2, 0, 2 * Math.PI);
    ctx.fill();
  });
  ctx.lineWidth = 1;
  ctx.globalAlpha = 0.35;
  const eyeLeft = [33, 160, 158, 133, 153, 144, 33];
  const eyeRight = [362, 385, 387, 263, 373, 380, 362];
  const lips = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61];
  function drawPath(indices) {
    ctx.beginPath();
    indices.forEach((idx, i) => {
      const p = landmarks[idx];
      if (i === 0) ctx.moveTo(p.x * w, p.y * h);
      else ctx.lineTo(p.x * w, p.y * h);
    });
    ctx.stroke();
  }
  drawPath(eyeLeft);
  drawPath(eyeRight);
  drawPath(lips);
  ctx.restore();
}
function performCalibration(lm) {
  baselineY = lm[1].y;
  baselineDist = getDistance(lm[234], lm[454]);
  baselineFaceWidth = getDistance(lm[234], lm[454]);
  calibLogs.push(Date.now());
  updateCountersDisplay();
  alertBox.className = "card alert-card";
  alertText.innerText = "Поза успешно откалибрована!";
  headerDot.className = "status-dot green";
}
calibrateBtn.addEventListener("click", () => {
  if (lastLandmarks) performCalibration(lastLandmarks);
});
async function init() {
  try {
    statusDiv.innerText = "Камера...";
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    await video.play();
    statusDiv.innerText = "Загрузка модели...";
    const filesetResolver = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.15/wasm"
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numFaces: 1
    });
    statusDiv.innerText = "Активен";
    predictWebcam();
  } catch (err) {
    statusDiv.innerText = "Ошибка";
    console.error(err);
  }
}
function predictWebcam() {
  if (canvas.width !== video.videoWidth) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
  let startTimeMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    if (faceLandmarker) {
      const results = faceLandmarker.detectForVideo(video, startTimeMs);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const hasFace = results.faceLandmarks && results.faceLandmarks.length > 0 
      if (!hasFace) {
        lastLandmarks = null;
        mainCard.className = "card video-card state-noface";
        if (gestureBadge) gestureBadge.innerText = "Ошибка";
        alertBox.className = "card alert-card warning";
        alertText.innerText = "Ваше лицо не видно! Сядьте перед камерой.";
        postureStatus.innerText = "НЕТ ЛИЦА";
        postureStatus.className = "status-text bad";
        postureBar.style.width = "0%";
        postureBar.style.backgroundColor = "#f43f5e";
        headerDot.className = "status-dot red";
        playWarningSound();
      } else {
        const lm = results.faceLandmarks[0];
        lastLandmarks = lm;
        if (baselineFaceWidth === null) {
          performCalibration(lm);
        }
        let activeColor = "#38bdf8";
        const currentFaceWidth = getDistance(lm[234], lm[454]);
        const isTooClose = currentFaceWidth > baselineFaceWidth * 1.35;
        const isTooFar = currentFaceWidth < baselineFaceWidth * 0.70;
        if (isTooClose || isTooFar) {
          if (!isDistanceActive) {
            isDistanceActive = true;
            distanceLogs.push(Date.now());
            updateCountersDisplay();
          }
          activeColor = "#f97316";
          mainCard.className = "card video-card state-distance";
          if (isTooClose) {
            if (gestureBadge) gestureBadge.innerText = "Слишком близко!";
            alertBox.className = "card alert-card warning";
            alertText.innerText = "Слишком близко к монитору! Отодвиньтесь.";
          } else {
            if (gestureBadge) gestureBadge.innerText = "Слишком далеко!";
            alertBox.className = "card alert-card warning";
            alertText.innerText = "Слишком далеко! Придвиньтесь ближе.";
          }
          postureStatus.innerText = "ДИСТАНЦИЯ";
          postureStatus.className = "status-text warning";
          postureBar.style.width = "50%";
          postureBar.style.backgroundColor = "#f97316";
          headerDot.className = "status-dot yellow";
          playWarningSound();
        } else {
          isDistanceActive = false;
        }
        const topLip = lm[13];
        const bottomLip = lm[14];
        const mouthDistance = getDistance(topLip, bottomLip);

        if (!isDistanceActive && mouthDistance > 0.08) {
          if (!isYawnActive) {
            isYawnActive = true;
            yawnLogs.push(Date.now());
            updateCountersDisplay();
          }
          activeColor = "#f59e0b";
          mainCard.className = "card video-card state-yawn";
          if (gestureBadge) gestureBadge.innerText = "Зевок / Усталость";
          headerDot.className = "status-dot yellow";
        } else {
          if (!isDistanceActive) isYawnActive = false;
        }
        const leftEar = lm[234];
        const rightEar = lm[454];
        const earTilt = Math.abs(leftEar.y - rightEar.y);
        if (!isDistanceActive && earTilt > 0.09) {
          if (!isTiltActive) {
            isTiltActive = true;
            performCalibration(lm);
          }
          activeColor = "#3b82f6";
          mainCard.className = "card video-card state-calib";
          if (gestureBadge) gestureBadge.innerText = "Наклон головы";
          headerDot.className = "status-dot blue";
        } else {
          if (!isDistanceActive) isTiltActive = false;
        }
        if (!isDistanceActive && baselineY !== null && !isYawnActive && !isTiltActive) {
          const currentY = lm[1].y;
          const yDiff = currentY - baselineY;
          if (yDiff > 0.065) {
            if (!isSlouchActive) {
              isSlouchActive = true;
              slouchLogs.push(Date.now());
              updateCountersDisplay();
            }
            activeColor = "#f43f5e";
            mainCard.className = "card video-card state-slouch";
            if (gestureBadge) gestureBadge.innerText = "Сутулость!";
            alertBox.className = "card alert-card warning";
            alertText.innerText = "Вы сутулитесь! Выпрямите спину.";
            postureStatus.innerText = "СУТУЛОСТЬ";
            postureStatus.className = "status-text bad";
            postureBar.style.width = "35%";
            postureBar.style.backgroundColor = "#f43f5e";
            headerDot.className = "status-dot red";
            playWarningSound();
          } else {
            isSlouchActive = false;
            mainCard.className = "card video-card";
            if (gestureBadge) gestureBadge.innerText = "Поза ровная";
            alertBox.className = "card alert-card";
            alertText.innerText = "Всё отлично, осанка ровная.";
            postureStatus.innerText = "ОК";
            postureStatus.className = "status-text good";
            postureBar.style.width = "100%";
            postureBar.style.backgroundColor = "#10b981";
            headerDot.className = "status-dot green";
          }
        }
        drawFaceSkeleton(lm, activeColor);
        const leftEye = [lm[33], lm[160], lm[158], lm[133], lm[153], lm[144]];
        const rightEye = [lm[362], lm[385], lm[387], lm[263], lm[373], lm[380]];
        const avgEAR = (getEAR(leftEye) + getEAR(rightEye)) / 2;
        if (avgEAR < 0.21) {
          if (!isClosed) {
            isClosed = true;
            blinkTimestamps.push(Date.now());
          }
        } else {
          isClosed = false;
        }
        const oneMinuteAgo = Date.now() - 60000;
        blinkTimestamps = blinkTimestamps.filter(t => t > oneMinuteAgo);
        blinkCountEl.innerText = blinkTimestamps.length;
      }
    }
  }
  requestAnimationFrame(predictWebcam);
}
pipBtn.addEventListener("click", async () => {
  if (pipWindow) {
    pipWindow.close();
    pipWindow = null;
    return;
  }
  if ("documentPictureInPicture" in window) {
    try {
      pipWindow = await window.documentPictureInPicture.requestWindow({
        width: 210,
        height: 48,
      });
      pipWindow.document.title = "Posture Guard";
      [...document.styleSheets].forEach(sheet => {
        try {
          const css = [...sheet.cssRules].map(r => r.cssText).join("");
          const style = document.createElement("style");
          style.textContent = css;
          pipWindow.document.head.appendChild(style);
        } catch (e) {
          const link = document.createElement("link");
          link.rel = "stylesheet";
          link.href = sheet.href;
          pipWindow.document.head.appendChild(link);
        }
      });
      pipWindow.document.body.className = "pip-root";
      pipWindow.document.body.innerHTML = `
        <div class="pip-badge" id="pip-badge">
          <div class="pip-led"></div>
          <span class="pip-text" id="pip-text">ОСАНКА ОК</span>
        </div>
      `;
      pipWindow.addEventListener("pagehide", () => {
        pipWindow = null;
      });
    } catch (e) {
      console.error(e);
    }
  } else {
    alert("PiP не поддерживается в вашем браузере.");
  }
});
init();
