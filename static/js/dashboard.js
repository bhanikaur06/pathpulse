
/* ---------- Camera capture for SOS alerts ---------- */

function capturePhoto() {
  return new Promise((resolve) => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      resolve(null);
      return;
    }
    navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } })
      .then((stream) => {
        const video = document.createElement("video");
        video.srcObject = stream;
        video.play();
        video.onloadedmetadata = () => {
          setTimeout(() => {
            const canvas = document.createElement("canvas");
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            canvas.getContext("2d").drawImage(video, 0, 0);
            stream.getTracks().forEach((t) => t.stop());
            canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.8);
          }, 400);
        };
      })
      .catch(() => resolve(null));
  });
}


/* ---------- Browser push notifications ---------- */

function requestNotificationPermission() {
  if ("Notification" in window && Notification.permission === "default") {
    Notification.requestPermission();
  }
}

function showBrowserNotification(title, body, urgent) {
  if (!("Notification" in window)) return;
  if (Notification.permission !== "granted") return;
  const n = new Notification(title, {
    body: body,
    icon: "https://cdn-icons-png.flaticon.com/512/1827/1827392.png",
    requireInteraction: !!urgent,
  });
  n.onclick = () => { window.focus(); n.close(); };
}

/* Path Pulse — dashboard live tracking logic */

let map, marker, trail, trailLine = [];
let countdownTimer = null;
let countdownSeconds = 360; // 6 minutes — middle of a 5-8 min "no response" window
const SOS_WAIT_SECONDS = 360;
let lastSentAt = 0;
const MIN_PING_INTERVAL_MS = 8000;
let alertInProgress = false; // guards against repeated pings restarting the timer

function initMap(lat, lon) {
  map = L.map("map", { zoomControl: false }).setView([lat, lon], 15);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    attribution: "&copy; OpenStreetMap &copy; CARTO",
    maxZoom: 19,
  }).addTo(map);

  marker = L.circleMarker([lat, lon], {
    radius: 8, color: "#2dd4bf", fillColor: "#2dd4bf", fillOpacity: 0.9, weight: 2,
  }).addTo(map);

  trail = L.polyline([], { color: "#2dd4bf", weight: 3, opacity: 0.6 }).addTo(map);

  loadRiskZones();
}

function loadRiskZones() {
  fetch("/risk_zones")
    .then((r) => r.json())
    .then((zones) => {
      zones.forEach((z) => {
        L.circle([z.latitude, z.longitude], {
          radius: 400,
          color: "#ef4444",
          fillColor: "#ef4444",
          fillOpacity: 0.12,
          weight: 1,
        }).addTo(map).bindPopup(`Reported risk area (score ${z.risk_score})`);
      });
    })
    .catch(() => {});
}

function setStatus(status, reason, extra) {
  const pulse = document.getElementById("pulseWrap");
  const badge = document.getElementById("statusBadge");

  pulse.classList.remove("is-caution", "is-danger");
  badge.classList.remove("badge-safe", "badge-caution", "badge-danger");

  if (status === "SAFE") {
    badge.classList.add("badge-safe");
    badge.textContent = "SAFE";
  } else if (status === "NEW_ROUTE") {
    pulse.classList.add("is-caution");
    badge.classList.add("badge-caution");
    badge.textContent = "NEW ROUTE";
    showBrowserNotification(
      "⚠️ Path Pulse: Unusual route detected",
      reason || "This doesn't match your usual travel pattern.",
      true
    );
  }

  document.getElementById("statusReason").textContent =
    reason || "Matches your usual travel pattern.";

  if (extra) {
    const parts = [`speed ${extra.speed_kmh} km/h`];
    if (extra.ml_ready) parts.push(`ml score ${extra.ml_score}`);
    else parts.push("ML model not trained yet, rule-based only");
    if (extra.is_night)     parts.push("🌙 night mode active");
    if (extra.sudden_stop)  parts.push("⚠️ sudden stop detected");
    if (extra.area_risk > 0)
      parts.push(`🔴 area risk ${extra.area_risk} (${extra.nearest_risk_km} km)`);
    document.getElementById("mlReading").textContent = parts.join(" · ");

    // Show/hide the risk zone warning banner
    const riskBanner = document.getElementById("riskBanner");
    if (extra.area_risk >= 0.7 && extra.nearest_risk_km < 0.5) {
      riskBanner.classList.remove("d-none");
      document.getElementById("riskBannerText").textContent =
        `You are near a high-risk area (risk score ${extra.area_risk}, ${extra.nearest_risk_km} km away).`;
    } else {
      riskBanner.classList.add("d-none");
    }
  }
}

function showNewRoutePrompt(lat, lon) {
  if (alertInProgress) return; // don't restart an already-running timer
  alertInProgress = true;

  document.getElementById("newRouteModal").classList.remove("d-none");

  // Start the "no response" timer the moment the popup appears —
  // doing nothing is what should lead to an automatic SOS.
  startSosCountdown(lat, lon, "No response to new-route check");

  document.getElementById("confirmSafeBtn").onclick = () => {
    fetch("/confirm_safe", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: `lat=${lat}&lon=${lon}`,
    });
    document.getElementById("newRouteModal").classList.add("d-none");
    stopSosCountdown();
    toast("Route marked safe — Path Pulse is learning it.");
  };

  document.getElementById("notSafeBtn").onclick = () => {
    document.getElementById("newRouteModal").classList.add("d-none");
    toast("Countdown continues — tap Cancel on the banner if you're okay.");
    // countdown keeps running in the background; banner stays visible
  };
}

function startSosCountdown(lat, lon, reason) {
  countdownSeconds = SOS_WAIT_SECONDS;
  const banner = document.getElementById("sosBanner");
  const display = document.getElementById("countdownDisplay");
  banner.classList.remove("d-none");
  document.getElementById("pulseWrap").classList.add("is-danger");

  clearInterval(countdownTimer);
  countdownTimer = setInterval(() => {
    countdownSeconds--;
    const m = String(Math.floor(countdownSeconds / 60)).padStart(2, "0");
    const s = String(countdownSeconds % 60).padStart(2, "0");
    display.textContent = `${m}:${s}`;

    if (countdownSeconds <= 0) {
      clearInterval(countdownTimer);
      sendSos(lat, lon, reason);
      banner.classList.add("d-none");
      alertInProgress = false;
      document.getElementById("pulseWrap").classList.remove("is-danger");
    }
  }, 1000);

  document.getElementById("cancelSosBtn").onclick = () => {
    stopSosCountdown();
    document.getElementById("newRouteModal").classList.add("d-none");
    toast("SOS cancelled.");
  };
}

function stopSosCountdown() {
  clearInterval(countdownTimer);
  document.getElementById("sosBanner").classList.add("d-none");
  document.getElementById("pulseWrap").classList.remove("is-danger");
  alertInProgress = false;
}

function sendSos(lat, lon, reason) {
  capturePhoto().then((photoBlob) => {
    const formData = new FormData();
    formData.append("lat", lat);
    formData.append("lon", lon);
    formData.append("reason", reason || "Manual SOS");
    if (photoBlob) formData.append("photo", photoBlob, "surroundings.jpg");

    fetch("/sos", { method: "POST", body: formData })
      .then((r) => r.json())
      .then((data) => {
        showContactsInformedBanner(data);
        showBrowserNotification(
          "🚨 Path Pulse: SOS Sent",
          "Your emergency contacts have been notified with your location.",
          true
        );
      });
  });
}

function showContactsInformedBanner(data) {
  let banner = document.getElementById("contactsInformedBanner");
  if (!banner) {
    banner = document.createElement("div");
    banner.id = "contactsInformedBanner";
    banner.style.cssText =
      "position:fixed;top:16px;left:50%;transform:translateX(-50%);" +
      "background:#065f46;border:1px solid #10b981;color:#d1fae5;" +
      "padding:14px 22px;border-radius:12px;z-index:9500;" +
      "box-shadow:0 12px 30px -10px rgba(16,185,129,0.5);" +
      "font-size:0.9rem;max-width:90vw;text-align:center;";
    document.body.appendChild(banner);
  }

  const names = (data.contacts_notified && data.contacts_notified.length)
    ? data.contacts_notified.join(", ")
    : "your emergency contacts";
  const method = data.email_sent ? "email" : (data.sms_sent ? "SMS" : "the system");

  banner.innerHTML =
    "✅ <strong>Emergency contacts informed</strong><br>" +
    `${names} notified via ${method}.` +
    '<div><button onclick="document.getElementById(\'contactsInformedBanner\').remove()" ' +
    'style="margin-top:8px;background:transparent;border:1px solid #10b981;color:#d1fae5;' +
    'padding:4px 12px;border-radius:6px;cursor:pointer;">Dismiss</button></div>';

  banner.style.display = "block";

  setTimeout(() => {
    if (banner) banner.style.display = "none";
  }, 15000);
}

function toast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.classList.remove("d-none");
  setTimeout(() => t.classList.add("d-none"), 4000);
}

function handlePosition(pos) {
  const lat = pos.coords.latitude;
  const lon = pos.coords.longitude;

  document.getElementById("coords").textContent = `${lat.toFixed(5)}, ${lon.toFixed(5)}`;

  if (!map) initMap(lat, lon);
  marker.setLatLng([lat, lon]);
  map.panTo([lat, lon]);
  trailLine.push([lat, lon]);
  trail.setLatLngs(trailLine);

  const now = Date.now();
  if (now - lastSentAt < MIN_PING_INTERVAL_MS) return;
  lastSentAt = now;

  fetch("/update_location", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: `lat=${lat}&lon=${lon}`,
  })
    .then((r) => r.json())
    .then((data) => {
      setStatus(data.status, data.reason, data);
      if (data.status === "NEW_ROUTE") {
        showNewRoutePrompt(lat, lon);
      }
    });
}

function startTracking() {
  if (!navigator.geolocation) {
    toast("Geolocation isn't supported in this browser.");
    return;
  }
  navigator.geolocation.watchPosition(handlePosition, () => toast("Location permission denied."), {
    enableHighAccuracy: true,
    maximumAge: 5000,
  });
}

/* ---------- Phase 3: secret code word (typed) ---------- */

function submitCodeWord() {
  const word = document.getElementById("codeWordInput").value;
  document.getElementById("codeWordInput").value = "";
  navigator.geolocation.getCurrentPosition((pos) => {
    capturePhoto().then((photoBlob) => {
      const formData = new FormData();
      formData.append("word", word);
      formData.append("lat", pos.coords.latitude);
      formData.append("lon", pos.coords.longitude);
      if (photoBlob) formData.append("photo", photoBlob, "surroundings.jpg");

      fetch("/verify_code_word", { method: "POST", body: formData })
        .then((r) => r.json())
        .then((data) => {
          if (data.match) toast("🚨 Code word recognized — silent alert sent.");
        });
    });
  });
}

/* ---------- Phase 4: voice trigger (Web Speech API) ---------- */

let recognition = null;
let voiceActive = false;

function toggleVoiceTrigger() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const btn = document.getElementById("voiceToggleBtn");

  if (!SpeechRecognition) {
    toast("Voice trigger isn't supported in this browser (try Chrome).");
    return;
  }

  if (voiceActive) {
    recognition.stop();
    voiceActive = false;
    btn.textContent = "🎤 Enable voice trigger";
    btn.classList.remove("btn-danger");
    btn.classList.add("btn-ghost");
    return;
  }

  recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = false;
  recognition.lang = "en-US";

  recognition.onresult = (event) => {
    const transcript = event.results[event.results.length - 1][0].transcript.trim().toLowerCase();

    if (transcript.includes("help me")) {
      navigator.geolocation.getCurrentPosition((pos) => {
        sendSos(pos.coords.latitude, pos.coords.longitude, "Voice trigger: 'help me'");
      });
      return;
    }

    // also check against the user's actual code word, server-side
    navigator.geolocation.getCurrentPosition((pos) => {
      capturePhoto().then((photoBlob) => {
        const formData = new FormData();
        formData.append("word", transcript);
        formData.append("lat", pos.coords.latitude);
        formData.append("lon", pos.coords.longitude);
        if (photoBlob) formData.append("photo", photoBlob, "surroundings.jpg");

        fetch("/verify_code_word", { method: "POST", body: formData })
          .then((r) => r.json())
          .then((data) => {
            if (data.match) toast("🚨 Voice code word recognized — silent alert sent.");
          });
      });
    });
  };

  recognition.onerror = () => { voiceActive = false; btn.textContent = "🎤 Enable voice trigger"; };
  recognition.onend = () => { if (voiceActive) recognition.start(); }; // keep listening

  recognition.start();
  voiceActive = true;
  btn.textContent = "🛑 Voice trigger active — say 'help me' or your code word";
  btn.classList.remove("btn-ghost");
  btn.classList.add("btn-danger");
}

document.addEventListener("DOMContentLoaded", () => {
  requestNotificationPermission();
  startTracking();
  document.getElementById("codeWordBtn").addEventListener("click", submitCodeWord);
  document.getElementById("voiceToggleBtn").addEventListener("click", toggleVoiceTrigger);
  document.getElementById("manualSosBtn").addEventListener("click", () => {
    navigator.geolocation.getCurrentPosition((pos) => {
      sendSos(pos.coords.latitude, pos.coords.longitude, "Manual SOS");
    });
  });

  const simBtn = document.getElementById("simulateBtn");
  if (simBtn) {
    simBtn.addEventListener("click", () => {
      const simLat = 19.0760;
      const simLon = 72.8777;

      fetch("/update_location", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: `lat=${simLat}&lon=${simLon}`,
      })
        .then((r) => r.json())
        .then((data) => {
          document.getElementById("coords").textContent =
            `${simLat.toFixed(5)}, ${simLon.toFixed(5)} (simulated)`;
          setStatus(data.status, data.reason, data);
          if (data.status === "NEW_ROUTE") {
            showNewRoutePrompt(simLat, simLon);
          } else {
            toast("Simulated ping was SAFE — try again or confirm a route first.");
          }
        });
    });
  }
});
