const STATS_POLL_MS = 1500;
const LOGS_POLL_MS = 2000;

let knownLogTimestamps = new Set();

function updateClock() {
  const now = new Date();
  document.getElementById("clock").textContent = now.toLocaleTimeString();
}

function updateConnectionStatus(status) {
  const pill = document.getElementById("connectionStatus");
  const text = document.getElementById("statusText");
  pill.classList.remove("live", "offline");
  if (status === "live") {
    pill.classList.add("live");
    text.textContent = "LIVE";
  } else if (status === "offline") {
    pill.classList.add("offline");
    text.textContent = "OFFLINE";
  } else {
    text.textContent = "CONNECTING";
  }
}

function updateAlerts(ppeStatus, requiredItems) {
  const row = document.getElementById("alertsRow");
  if (!ppeStatus || !requiredItems) {
    row.innerHTML = "";
    return;
  }
  row.innerHTML = requiredItems
    .filter((item) => ppeStatus[item] && ppeStatus[item] !== "unknown")
    .map((item) => {
      const status = ppeStatus[item];
      const cls = status === "detected" ? "detected" : "missing";
      const text = `${item.toUpperCase()} ${status === "detected" ? "DETECTED" : "MISSING"}`;
      return `<span class="alert-chip ${cls}">${text}</span>`;
    })
    .join("");
}

function updateAreaButtons(activeAreaId) {
  document.querySelectorAll(".area-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.area === activeAreaId);
  });
}

async function fetchStats() {
  try {
    const res = await fetch("/api/stats");
    const data = await res.json();

    document.getElementById("statPersons").textContent = data.current_persons;
    document.getElementById("statAreaFoot").textContent = `in ${data.active_area_name}`;
    document.getElementById("logCount").textContent = `${data.total_logs} records`;
    document.getElementById("areaNameTag").textContent = data.active_area_name;
    updateAlerts(data.ppe_status, data.required_items);
    updateAreaButtons(data.active_area);

    updateConnectionStatus(data.camera_status);
  } catch (err) {
    console.warn("fetchStats failed:", err);
    updateConnectionStatus("offline");
  }
}

function makeCell(text) {
  const td = document.createElement("td");
  td.textContent = text;
  return td;
}

function makeSeverityCell(severity) {
  const td = document.createElement("td");
  const span = document.createElement("span");
  span.className = `severity-chip severity-${severity}`;
  span.textContent = severity;
  td.appendChild(span);
  return td;
}

function makePhotoCell(photoUrl) {
  const td = document.createElement("td");
  if (!photoUrl) {
    td.textContent = "—";
    return td;
  }
  const a = document.createElement("a");
  a.href = photoUrl;
  a.target = "_blank";
  const img = document.createElement("img");
  img.className = "log-thumb";
  img.src = photoUrl;
  img.alt = "Violation evidence";
  a.appendChild(img);
  td.appendChild(a);
  return td;
}

async function fetchLogs() {
  try {
    const res = await fetch("/api/logs");
    const data = await res.json();
    const tbody = document.getElementById("logBody");

    if (data.length === 0) {
      tbody.innerHTML = "";
      const row = document.createElement("tr");
      row.className = "empty-row";
      const td = document.createElement("td");
      td.colSpan = 6;
      td.textContent = "No violations recorded yet — monitoring in progress";
      row.appendChild(td);
      tbody.appendChild(row);
      return;
    }

    tbody.innerHTML = "";
    data.forEach((entry) => {
      const key = `${entry.timestamp}-${entry.person_id}-${entry.violation_type}`;
      const isNew = !knownLogTimestamps.has(key);
      knownLogTimestamps.add(key);

      const row = document.createElement("tr");
      if (isNew) row.classList.add("new-row");
      row.appendChild(makeCell(entry.timestamp));
      row.appendChild(makeCell(entry.person_id));
      row.appendChild(makeCell(entry.area));
      row.appendChild(makeCell(entry.violation_type));
      row.appendChild(makeSeverityCell(entry.severity));
      row.appendChild(makePhotoCell(entry.photo));
      tbody.appendChild(row);
    });
  } catch (err) {
    console.warn("fetchLogs failed, will retry on next poll:", err);
  }
}

async function setActiveArea(areaId) {
  try {
    await fetch("/api/area", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ area: areaId }),
    });
    updateAreaButtons(areaId);
    fetchStats();
  } catch (err) {
    console.warn("setActiveArea failed, UI will resync on next stats poll:", err);
  }
}

document.querySelectorAll(".area-btn").forEach((btn) => {
  btn.addEventListener("click", () => setActiveArea(btn.dataset.area));
});

setInterval(updateClock, 1000);
setInterval(fetchStats, STATS_POLL_MS);
setInterval(fetchLogs, LOGS_POLL_MS);

updateClock();
fetchStats();
fetchLogs();