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

const PPE_ORDER = ["helmet", "vest", "goggles", "gloves"];

function updateAlerts(ppeStatus) {
  const row = document.getElementById("alertsRow");
  if (!ppeStatus) {
    row.innerHTML = "";
    return;
  }
  row.innerHTML = PPE_ORDER
    .filter((item) => ppeStatus[item] && ppeStatus[item] !== "unknown")
    .map((item) => {
      const status = ppeStatus[item];
      const cls = status === "detected" ? "detected" : "missing";
      const text = `${item.toUpperCase()} ${status === "detected" ? "DETECTED" : "MISSING"}`;
      return `<span class="alert-chip ${cls}">${text}</span>`;
    })
    .join("");
}

async function fetchStats() {
  try {
    const res = await fetch("/api/stats");
    const data = await res.json();

    document.getElementById("statPersons").textContent = data.current_persons;
    document.getElementById("logCount").textContent = `${data.total_logs} records`;
    updateAlerts(data.ppe_status);

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
      const key = `${entry.timestamp}-${entry.description}`;
      const isNew = !knownLogTimestamps.has(key);
      knownLogTimestamps.add(key);

      const row = document.createElement("tr");
      if (isNew) row.classList.add("new-row");
      row.innerHTML = `
        <td>${entry.timestamp}</td>
        <td>${entry.camera}</td>
        <td>${entry.violation_type}</td>
        <td>${entry.description}</td>
        <td>${severityChip(entry.severity)}</td>
        <td>${entry.assigned_to}</td>
      `;
      tbody.appendChild(row);
    });
  } catch (err) {
    // silent — next poll will retry
  }
}

setInterval(updateClock, 1000);
setInterval(fetchStats, STATS_POLL_MS);
setInterval(fetchLogs, LOGS_POLL_MS);

updateClock();
fetchStats();
fetchLogs();