const fileInput = document.getElementById("file-input");
const uploadBtn = document.getElementById("upload-btn");
const enableContextCheckbox = document.getElementById("enable-context-checkbox");
const jobIdInput = document.getElementById("job-id-input");
const reconnectBtn = document.getElementById("reconnect-btn");
const jobIdDisplay = document.getElementById("job-id-display");
const progressSection = document.getElementById("progress-section");
const progressFill = document.getElementById("progress-bar-fill");
const progressLabel = document.getElementById("progress-label");
const chunkLog = document.getElementById("chunk-log");
const summarySection = document.getElementById("summary-section");
const finalSummary = document.getElementById("final-summary");
const chatSection = document.getElementById("chat-section");
const chatLog = document.getElementById("chat-log");
const chatInput = document.getElementById("chat-input");
const chatSendBtn = document.getElementById("chat-send-btn");

let currentJobId = null;

function resetView() {
  chunkLog.innerHTML = "";
  summarySection.hidden = true;
  chatSection.hidden = true;
  chatLog.innerHTML = "";
  progressSection.hidden = false;
  progressFill.style.width = "0%";
}

function showJobId(jobId) {
  currentJobId = jobId;
  jobIdDisplay.hidden = false;
  jobIdDisplay.textContent = `Job ID: ${jobId} (copy this if you want to reattach later)`;
}

function watchJob(jobId) {
  showJobId(jobId);
  const ws = new WebSocket(`ws://${location.host}/ws/${jobId}`);

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);

    if (msg.type === "progress") {
      const pct = msg.total ? Math.round((msg.chunk / msg.total) * 100) : 0;
      progressFill.style.width = `${pct}%`;
      progressLabel.textContent = `Chunk ${msg.chunk}/${msg.total} — ${msg.stage}`;
    }

    if (msg.type === "chunk_result") {
      const entry = document.createElement("div");
      entry.className = "chunk-entry";
      entry.innerHTML = `<strong>Chunk ${msg.chunk}/${msg.total}</strong> (${msg.frames_analyzed} frames)<p>${msg.summary}</p>`;
      chunkLog.appendChild(entry);
      chunkLog.scrollTop = chunkLog.scrollHeight;
    }

    if (msg.type === "done") {
      progressLabel.textContent = "Done.";
      summarySection.hidden = false;
      finalSummary.textContent = msg.final_summary;
      chatSection.hidden = false;
      uploadBtn.disabled = false;
    }

    if (msg.type === "error") {
      progressLabel.textContent = `Error: ${msg.message}`;
      uploadBtn.disabled = false;
    }
  };

  ws.onerror = () => {
    progressLabel.textContent = "WebSocket error — the job may still be running server-side, try reconnecting.";
    uploadBtn.disabled = false;
  };
}

uploadBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) {
    alert("Choose a video file first.");
    return;
  }

  uploadBtn.disabled = true;
  resetView();
  progressLabel.textContent = "Uploading...";

  const formData = new FormData();
  formData.append("file", file);
  formData.append("disable_context", (!enableContextCheckbox.checked).toString());

  // Upload starts the job immediately server-side - it keeps running
  // even if you close this tab right after this call returns.
  const uploadRes = await fetch("/upload", { method: "POST", body: formData });
  const { job_id } = await uploadRes.json();

  watchJob(job_id);
});

reconnectBtn.addEventListener("click", () => {
  const jobId = jobIdInput.value.trim();
  if (!jobId) {
    alert("Paste a job ID first.");
    return;
  }
  resetView();
  progressLabel.textContent = "Reattaching...";
  watchJob(jobId);
});

chatSendBtn.addEventListener("click", async () => {
  const question = chatInput.value.trim();
  if (!question || !currentJobId) return;

  const q = document.createElement("div");
  q.className = "chat-msg chat-msg-user";
  q.textContent = question;
  chatLog.appendChild(q);
  chatInput.value = "";
  chatSendBtn.disabled = true;

  const res = await fetch(`/chat/${currentJobId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  const data = await res.json();

  const a = document.createElement("div");
  a.className = "chat-msg chat-msg-ai";
  a.textContent = data.answer || "(no answer)";
  chatLog.appendChild(a);
  chatLog.scrollTop = chatLog.scrollHeight;
  chatSendBtn.disabled = false;
});

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") chatSendBtn.click();
});
