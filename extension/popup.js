document.addEventListener("DOMContentLoaded", () => {
  chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
    if (!tab) {
      setStatus("No active tab found.", true);
      return;
    }

    chrome.tabs.sendMessage(tab.id, { type: "GET_COUNTS" }, (response) => {
      if (chrome.runtime.lastError || !response) {
        setStatus("Content script not active on this page.", true);
        return;
      }

      document.getElementById("warn-count").textContent = response.warn ?? 0;
      document.getElementById("harmful-count").textContent = response.harmful ?? 0;

      const total = (response.warn ?? 0) + (response.harmful ?? 0);
      if (total === 0) {
        setStatus("No flagged content detected yet.");
      } else {
        setStatus(`${total} item${total === 1 ? "" : "s"} flagged on this page.`);
      }
    });
  });
});

function setStatus(msg, isError = false) {
  const el = document.getElementById("status");
  el.textContent = msg;
  el.className = "status" + (isError ? " error" : "");
}
