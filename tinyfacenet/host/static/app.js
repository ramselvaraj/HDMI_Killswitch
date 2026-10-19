const $ = (id) => document.getElementById(id);

function fmtMs(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return `${Number(value).toFixed(1)} ms`;
}

function fmtValue(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toFixed(digits);
}

function thresholdToPercent(value) {
  return Math.round(Math.max(0, Math.min(1, Number(value) || 0)) * 100);
}

async function postJson(url, body = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`${url} failed: ${response.status}`);
  return response.json();
}

function applyStatus(status) {
  $("message").textContent = status.message || "No status";

  const state = status.state || "idle";
  $("statePill").textContent = state.toUpperCase();
  $("statePill").className = `state-pill ${state}`;

  const verdict = $("verdict");
  if (!status.running || state === "idle") {
    verdict.textContent = "IDLE";
    verdict.className = "verdict idle";
  } else if (!status.face_present) {
    verdict.textContent = "NO USER";
    verdict.className = "verdict idle";
  } else if (status.match) {
    verdict.textContent = "MATCH";
    verdict.className = "verdict match";
  } else {
    verdict.textContent = "NO MATCH";
    verdict.className = "verdict no-match";
  }

  $("score").textContent = fmtValue(status.best_score ?? status.last_score, 4);
  const thresholdPercent = thresholdToPercent(status.threshold);
  $("thresholdText").textContent = String(thresholdPercent);
  if (document.activeElement !== $("thresholdSlider")) {
    $("thresholdSlider").value = String(thresholdPercent);
  }

  const score = status.best_score ?? status.last_score;
  const width = score === null || score === undefined ? 0 : Math.max(0, Math.min(100, ((Number(score) + 1) / 2) * 100));
  $("scoreBar").style.width = `${width}%`;
  $("scoreBar").style.background = status.match ? "var(--green)" : "var(--blue)";

  $("cnnMs").textContent = fmtMs(status.hls_ms ?? status.cnn_ms);
  $("detectMs").textContent = fmtMs(status.detect_ms);
  $("cosineMs").textContent = fmtMs(status.cosine_ms);
  $("loopMs").textContent = fmtMs(status.loop_ms);
  $("fps").textContent = fmtValue(status.fps, 1);
  $("profiles").textContent = status.authorized_count ?? "--";
  $("tfmIp").textContent = status.tinyfacematch_ip || "--";
  $("cosineIp").textContent = status.cosine_ip || "--";

  if (status.enrollment) {
    $("enrollStatus").textContent = `${status.enrollment.name}: ${status.enrollment.captured}/${status.enrollment.target} samples`;
  } else {
    $("enrollStatus").textContent = "No enrollment in progress";
  }
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status");
    applyStatus(await response.json());
  } catch (error) {
    $("message").textContent = error.message;
    $("statePill").textContent = "ERROR";
    $("statePill").className = "state-pill error";
  }
}

$("startBtn").addEventListener("click", async () => {
  await postJson("/api/start");
  await refreshStatus();
});

$("stopBtn").addEventListener("click", async () => {
  await postJson("/api/stop");
  await refreshStatus();
});

$("reloadBtn").addEventListener("click", async () => {
  await postJson("/api/reload_profiles");
  await refreshStatus();
});

$("enrollBtn").addEventListener("click", async () => {
  await postJson("/api/enroll", {
    name: $("enrollName").value || "me",
    samples: Number($("enrollSamples").value || 3),
  });
  await refreshStatus();
});

let thresholdTimer = null;

async function postThreshold() {
  const threshold = Number($("thresholdSlider").value) / 100.0;
  await postJson("/api/threshold", { threshold });
  await refreshStatus();
}

$("thresholdSlider").addEventListener("input", () => {
  $("thresholdText").textContent = String(Math.round(Number($("thresholdSlider").value)));
  window.clearTimeout(thresholdTimer);
  thresholdTimer = window.setTimeout(() => {
    postThreshold().catch((error) => {
      $("message").textContent = error.message;
    });
  }, 180);
});

$("thresholdSlider").addEventListener("change", () => {
  window.clearTimeout(thresholdTimer);
  postThreshold().catch((error) => {
    $("message").textContent = error.message;
  });
});

refreshStatus();
setInterval(refreshStatus, 600);
