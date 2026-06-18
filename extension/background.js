// Service worker for the side panel extension.
// MV3 service workers sleep after ~30s idle - this file only runs setup code,
// no state is stored here. Anything that needs to persist lives in chrome.storage.local.

chrome.runtime.onInstalled.addListener(() => {
  chrome.sidePanel
    .setPanelBehavior({ openPanelOnActionClick: true })
    .catch((error) => console.error("Failed to set side panel behavior:", error));
});

console.log("[background.js] service worker loaded");

// Notify the side panel whenever the active tab changes or navigates,
// so it can re-detect the current video. The panel itself decides what
// to do with this (re-render that video's saved chat, or show "no video").
// sendMessage rejects with "Receiving end does not exist" if the panel
// isn't open - that's expected and harmless, so we swallow it.
function notifyPanel(reason) {
  console.log("[background.js] notifyPanel triggered by:", reason);
  chrome.runtime.sendMessage({ type: "ACTIVE_TAB_CHANGED" }).catch((err) => {
    console.log("[background.js] sendMessage failed (panel likely closed):", err.message);
  });
}

chrome.tabs.onActivated.addListener((activeInfo) => {
  console.log("[background.js] onActivated fired", activeInfo);
  notifyPanel("onActivated");
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  console.log("[background.js] onUpdated fired", { tabId, changeInfo, tabActive: tab.active });
  if (changeInfo.url && tab.active) {
    notifyPanel("onUpdated");
  }
});