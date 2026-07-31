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
    updateConnectionStatus("offline");
  }
}

function severityChip(severity) {
  return `<span class="severity-chip severity-${severity}">${severity}</span>`;
}

async function fetchLogs() {
  try {
    const res = await fetch("/api/logs");
    const data = await res.json();
    const tbody = document.getElementById("logBody");

    if (data.length === 0) {
      tbody.innerHTML = '<tr class="empty-row"><td colspan="6">No violations recorded yet — monitoring in progress</td></tr>';
      return;
    }

    tbody.innerHTML = "";
    data.forEach((entry) => {
      const key = `${entry.timestamp}-${entry.person_id}-${entry.violation_type}`;
      const isNew = !knownLogTimestamps.has(key);
      knownLogTimestamps.add(key);

      const row = document.createElement("tr");
      if (isNew) row.classList.add("new-row");
      row.innerHTML = `
        <td>${entry.timestamp}</td>
        <td>${entry.person_id}</td>
        <td>${entry.area}</td>
        <td>${entry.violation_type}</td>
        <td>${severityChip(entry.severity)}</td>
        <td><a href="${entry.photo}" target="_blank"><img class="log-thumb" src="${entry.photo}" alt="Violation evidence"></a></td>
      `;
      tbody.appendChild(row);
    });
  } catch (err) {
    // silent — next poll will retry
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
    // silent — UI will resync on next stats poll
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