const $ = (id) => document.getElementById(id);
const video = $("video"), canvas = $("canvas"), ctx = canvas.getContext("2d");
const COLORS = {
  organic: "var(--organic)",
  recyclable: "var(--recyclable)",
  nonrecyclable: "var(--nonrecyclable)",
};
const NAMES = { organic: "Organic", recyclable: "Recyclable", nonrecyclable: "Non-recyclable" };
const HINTS = {
  organic: "Compost / green bin",
  recyclable: "Recycling bin",
  nonrecyclable: "General waste / special disposal",
};
const SEND_SIZE = 320;

let stream = null, running = false, smoothed = null, models = [], frames = [];

async function loadModels() {
  const res = await fetch("/api/models").then((r) => r.json());
  models = res.models;
  $("device").textContent = res.device.toUpperCase();
  $("modelSelect").innerHTML = models
    .map((m) => `<option value="${m.name}" ${m.name === res.default ? "selected" : ""}>${m.name}</option>`)
    .join("");
  showModelMeta();
}

function showModelMeta() {
  const m = models.find((x) => x.name === $("modelSelect").value);
  if (!m) return;
  const acc = m.test_acc != null ? `test acc ${(m.test_acc * 100).toFixed(2)}%` : `val acc ${(m.val_acc * 100).toFixed(2)}%`;
  $("modelMeta").textContent = `${m.params_M}M params · ${acc}${m.calibrated ? " · calibrated" : ""}`;
  smoothed = null;
}

function buildBars(classes) {
  $("bars").innerHTML = classes
    .map((c) => `<div class="bar"><div class="bar-head"><span>${NAMES[c] || c}</span><b id="p-${c}">0%</b></div>
      <div class="bar-track"><div class="bar-fill" id="b-${c}" style="background:${COLORS[c]}"></div></div></div>`)
    .join("");
}

function render(result) {
  const classes = Object.keys(result.probs);
  if (!$("bars").children.length) buildBars(classes);

  // exponential smoothing across frames to reduce flicker
  const alpha = $("smooth").value / 100;
  const p = classes.map((c) => result.probs[c]);
  smoothed = smoothed && running ? smoothed.map((s, i) => alpha * s + (1 - alpha) * p[i]) : p;

  const top = smoothed.indexOf(Math.max(...smoothed));
  const label = classes[top], conf = smoothed[top];
  const uncertain = conf * 100 < +$("threshold").value;
  const color = uncertain ? "var(--uncertain)" : COLORS[label];

  classes.forEach((c, i) => {
    $(`p-${c}`).textContent = `${(smoothed[i] * 100).toFixed(1)}%`;
    $(`b-${c}`).style.width = `${smoothed[i] * 100}%`;
  });
  document.documentElement.style.setProperty("--frame", color);
  $("videoWrap").style.borderColor = color;
  $("label").textContent = uncertain ? "Uncertain" : NAMES[label];
  $("hint").textContent = uncertain ? `Best guess: ${NAMES[label]}` : HINTS[label];
  $("ringText").textContent = `${Math.round(conf * 100)}%`;
  $("ringFill").style.strokeDashoffset = 326.7 * (1 - conf);
  $("overlayLabel").hidden = false;
  $("overlayLabel").textContent = uncertain ? "Uncertain" : `${NAMES[label]} ${Math.round(conf * 100)}%`;
  $("latency").textContent = result.latency_ms.toFixed(0);
}

// crop the centre square (what the scan box shows) and send it as JPEG
function grabSquare(source, w, h) {
  const s = Math.min(w, h);
  canvas.width = canvas.height = SEND_SIZE;
  ctx.drawImage(source, (w - s) / 2, (h - s) / 2, s, s, 0, 0, SEND_SIZE, SEND_SIZE);
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
}

async function classify(blob) {
  const res = await fetch(`/api/predict?model=${encodeURIComponent($("modelSelect").value)}`, {
    method: "POST",
    headers: { "Content-Type": "image/jpeg" },
    body: blob,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function loop() {
  while (running) {
    if (video.readyState >= 2) {
      try {
        render(await classify(await grabSquare(video, video.videoWidth, video.videoHeight)));
        const now = performance.now();
        frames = frames.filter((t) => now - t < 1000).concat(now);
        $("fps").textContent = frames.length;
      } catch (e) {
        console.error(e);
        await new Promise((r) => setTimeout(r, 500));
      }
    } else {
      await new Promise((r) => setTimeout(r, 50));
    }
  }
}

async function listCameras() {
  const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  const current = stream?.getVideoTracks()[0]?.getSettings().deviceId;
  $("cameraSelect").innerHTML = cams
    .map((c, i) => `<option value="${c.deviceId}" ${c.deviceId === current ? "selected" : ""}>${c.label || `Camera ${i + 1}`}</option>`)
    .join("");
}

async function startCamera(deviceId) {
  stopCamera();
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: deviceId ? { deviceId: { exact: deviceId } } : { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" },
      audio: false,
    });
  } catch (e) {
    $("statusText").textContent = `Camera error: ${e.message}`;
    return;
  }
  video.srcObject = stream;
  await video.play();
  $("still").hidden = true;
  video.hidden = false;
  $("placeholder").hidden = true;
  $("toggleCam").textContent = "Stop camera";
  $("toggleCam").classList.add("stop");
  $("dot").classList.add("live");
  $("statusText").textContent = "Live";
  running = true;
  smoothed = null;
  listCameras();
  loop();
}

function stopCamera() {
  running = false;
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  $("toggleCam").textContent = "Start camera";
  $("toggleCam").classList.remove("stop");
  $("dot").classList.remove("live");
  $("statusText").textContent = "Camera off";
  $("fps").textContent = "0";
}

$("toggleCam").onclick = () => (stream ? stopCamera() : startCamera());
$("cameraSelect").onchange = (e) => startCamera(e.target.value);
$("modelSelect").onchange = showModelMeta;
$("mirror").onchange = (e) => $("videoWrap").classList.toggle("mirror", e.target.checked);
$("threshold").oninput = (e) => ($("thrVal").textContent = `${e.target.value}%`);
$("smooth").oninput = (e) => ($("smoothVal").textContent = (e.target.value / 100).toFixed(1));

$("upload").onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  stopCamera();
  const img = $("still");
  img.src = URL.createObjectURL(file);
  await img.decode();
  img.hidden = false;
  video.hidden = true;
  $("placeholder").hidden = true;
  smoothed = null;
  render(await classify(await grabSquare(img, img.naturalWidth, img.naturalHeight)));
  e.target.value = "";
};

$("videoWrap").classList.add("mirror");
loadModels().catch((e) => ($("statusText").textContent = `Server error: ${e.message}`));
