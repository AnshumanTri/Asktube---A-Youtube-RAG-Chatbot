const BACKEND_URL = "http://localhost:8000"; // TODO: change to Render URL after deploy

// --- DOM refs ---
const closeBtn = document.getElementById("close-btn");
const noVideoState = document.getElementById("no-video-state");
const notIndexedState = document.getElementById("not-indexed-state");
const chatState = document.getElementById("chat-state");

const videoTitleDisplay = document.getElementById("video-title-display");
const videoTitleChat = document.getElementById("video-title-chat");
const indexBtn = document.getElementById("index-btn");
const indexStatus = document.getElementById("index-status");

const chatMessages = document.getElementById("chat-messages");
const questionInput = document.getElementById("question-input");
const sendBtn = document.getElementById("send-btn");

// currentVideoId is also mirrored to chrome.storage.local so it survives
// service worker sleep / panel reloads, per MV3 side panel guidance.
let currentVideoId = null;

closeBtn.addEventListener("click", () => window.close());

// --- Per-video chat history (chrome.storage.local) ---
// Stored shape: { chatHistories: { [videoId]: [{sender, text, sources}, ...] } }

async function getChatHistory(videoId) {
  const { chatHistories = {} } = await chrome.storage.local.get("chatHistories");
  return chatHistories[videoId] || [];
}

async function appendToChatHistory(videoId, entry) {
  const { chatHistories = {} } = await chrome.storage.local.get("chatHistories");
  const updated = [...(chatHistories[videoId] || []), entry];
  chatHistories[videoId] = updated;
  await chrome.storage.local.set({ chatHistories });
}

console.log("[sidepanel.js] loaded");

// Re-detect the active tab whenever the background script notifies us
// of a tab switch or in-tab navigation (e.g. clicking a different video).
chrome.runtime.onMessage.addListener((message) => {
  console.log("[sidepanel.js] received message:", message);
  if (message.type === "ACTIVE_TAB_CHANGED") {
    init();
  }
});

// --- Extract video ID from a YouTube URL (mirrors backend logic) ---
function extractVideoId(url) {
  const patterns = [/[?&]v=([A-Za-z0-9_-]{11})/, /youtu\.be\/([A-Za-z0-9_-]{11})/, /shorts\/([A-Za-z0-9_-]{11})/];
  for (const pattern of patterns) {
    const match = url.match(pattern);
    if (match) return match[1];
  }
  return null;
}

function showState(state) {
  noVideoState.classList.add("hidden");
  notIndexedState.classList.add("hidden");
  chatState.classList.add("hidden");
  state.classList.remove("hidden");
}

// --- Check backend if this video is already indexed ---
async function checkIndexStatus(videoId) {
  try {
    const res = await fetch(`${BACKEND_URL}/status/${videoId}`);
    if (!res.ok) return false;
    const data = await res.json();
    return data.indexed;
  } catch (err) {
    return false;
  }
}

// --- Init: detect current tab, restore that video's chat if any ---
async function init() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  console.log("[sidepanel.js] init() running, active tab url:", tab?.url);

  const isYouTubeVideo = tab?.url && (tab.url.includes("youtube.com/watch") || tab.url.includes("youtu.be/"));
  if (!isYouTubeVideo) {
    console.log("[sidepanel.js] not a YouTube video tab");
    currentVideoId = null;
    showState(noVideoState);
    return;
  }

  const videoId = extractVideoId(tab.url);
  console.log("[sidepanel.js] extracted videoId:", videoId, "| currentVideoId was:", currentVideoId);
  if (!videoId) {
    currentVideoId = null;
    showState(noVideoState);
    return;
  }

  // Avoid redundant work if we're already showing this exact video
  if (videoId === currentVideoId) {
    console.log("[sidepanel.js] same video as before, skipping");
    return;
  }

  currentVideoId = videoId;
  const title = tab.title?.replace(" - YouTube", "") || "This video";
  await chrome.storage.local.set({ currentVideoId: videoId, currentVideoTitle: title });

  const alreadyIndexed = await checkIndexStatus(videoId);

  if (alreadyIndexed) {
    videoTitleChat.textContent = title;
    await restoreChatHistory(videoId);
    showState(chatState);
  } else {
    videoTitleDisplay.textContent = title;
    showState(notIndexedState);
  }
}

// --- Restore saved messages for a video into the DOM (does not re-save them) ---
async function restoreChatHistory(videoId) {
  chatMessages.innerHTML = "";
  const history = await getChatHistory(videoId);
  for (const entry of history) {
    renderMessage(entry.text, entry.sender, { sources: entry.sources || [], persist: false });
  }
}

// --- Index button click ---
indexBtn.addEventListener("click", async () => {
  if (!currentVideoId) return;

  indexBtn.disabled = true;
  indexStatus.textContent = "Indexing video... this can take 10-30 seconds.";
  indexStatus.classList.remove("error");

  try {
    const res = await fetch(`${BACKEND_URL}/index`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: currentVideoId }),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Indexing failed.");
    }

    indexStatus.textContent = "";
    videoTitleChat.textContent = videoTitleDisplay.textContent;
    await restoreChatHistory(currentVideoId);
    showState(chatState);
  } catch (err) {
    indexStatus.textContent = err.message;
    indexStatus.classList.add("error");
    indexBtn.disabled = false;
  }
});

// --- Chat: render a message bubble, optionally with clickable timestamp sources ---
// persist=true (default) saves this message to the current video's chat history.
// Pass persist=false when restoring already-saved history, to avoid double-saving.
function renderMessage(text, sender, { isLoading = false, sources = [], persist = true } = {}) {
  const wrapper = document.createElement("div");
  wrapper.className = `msg msg-${sender}${isLoading ? " loading" : ""}`;
  wrapper.textContent = text;
  chatMessages.appendChild(wrapper);

  if (sources.length > 0) {
    const sourcesDiv = document.createElement("div");
    sourcesDiv.className = "msg-sources";
    for (const src of sources) {
      const link = document.createElement("a");
      link.className = "source-link";
      link.textContent = `▶ ${src.label}`;
      link.href = src.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      sourcesDiv.appendChild(link);
    }
    chatMessages.appendChild(sourcesDiv);
  }

  chatMessages.scrollTop = chatMessages.scrollHeight;

  if (persist && !isLoading && currentVideoId) {
    appendToChatHistory(currentVideoId, { sender, text, sources });
  }

  return wrapper;
}

// --- Chat: send question ---
async function sendQuestion() {
  const question = questionInput.value.trim();
  if (!question || !currentVideoId) return;

  renderMessage(question, "user");
  questionInput.value = "";
  sendBtn.disabled = true;
  questionInput.disabled = true;

  const loadingMsg = renderMessage("Thinking...", "bot", { isLoading: true });

  try {
    const res = await fetch(`${BACKEND_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: currentVideoId, question }),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Something went wrong.");
    }

    loadingMsg.remove();
    renderMessage(data.answer, "bot", { sources: data.sources || [] });
  } catch (err) {
    loadingMsg.remove();
    renderMessage(`Error: ${err.message}`, "bot");
  } finally {
    sendBtn.disabled = false;
    questionInput.disabled = false;
    questionInput.focus();
  }
}

sendBtn.addEventListener("click", sendQuestion);
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendQuestion();
});

// --- Start ---
init();