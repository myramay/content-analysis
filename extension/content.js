/**
 * content.js — ED Content Detector browser extension
 *
 * - Scans text elements every 3 seconds using setInterval + MutationObserver
 * - WARN verdict: soft yellow blur overlay + click-to-reveal badge
 * - HARMFUL verdict: replaces element with red intervention card + "Show anyway"
 * - Responds to popup requests for flagged item counts
 */

const API_URL = "http://localhost:8000/classify";
const SCAN_INTERVAL_MS = 3000;
const BATCH_SIZE = 5;
const MIN_TEXT_LEN = 35;
const MAX_TEXT_LEN = 1000;

// Track processed elements and running counts
const scanned = new WeakSet();
const pending = [];
const counts = { warn: 0, harmful: 0 };

// Inject shared styles once
(function injectStyles() {
  const style = document.createElement("style");
  style.textContent = `
    [data-ed-verdict="WARN"] {
      filter: blur(3px);
      transition: filter 0.25s ease;
    }
    .ed-warn-badge {
      display: inline-block;
      background: #fbbf24;
      color: #1c1917;
      font-size: 12px;
      font-weight: 700;
      padding: 3px 10px;
      border-radius: 4px;
      margin: 4px 0 2px;
      cursor: pointer;
      font-family: system-ui, sans-serif;
      user-select: none;
      line-height: 1.6;
    }
    .ed-warn-badge:hover { background: #f59e0b; }

    .ed-intervention {
      background: #fef2f2;
      border: 2px solid #ef4444;
      border-radius: 8px;
      padding: 14px 16px;
      font-family: system-ui, sans-serif;
    }
    .ed-intervention-title {
      margin: 0 0 6px;
      font-weight: 700;
      color: #b91c1c;
      font-size: 14px;
    }
    .ed-intervention-body {
      margin: 0 0 10px;
      font-size: 13px;
      color: #374151;
      line-height: 1.5;
    }
    .ed-show-anyway {
      background: transparent;
      border: 1px solid #d1d5db;
      border-radius: 4px;
      padding: 4px 12px;
      font-size: 12px;
      cursor: pointer;
      color: #6b7280;
      font-family: system-ui, sans-serif;
    }
    .ed-show-anyway:hover { border-color: #9ca3af; color: #374151; }
  `;
  document.head.appendChild(style);
})();

// ── Respond to popup requests ──────────────────────────────────────────────
chrome.runtime.onMessage.addListener((msg, _sender, respond) => {
  if (msg.type === "GET_COUNTS") {
    respond({ ...counts });
    return true; // keep channel open for async response
  }
});

// ── Overlay helpers ────────────────────────────────────────────────────────

function applyWarn(el) {
  if (el.dataset.edVerdict) return;
  el.dataset.edVerdict = "WARN";
  counts.warn++;

  const badge = document.createElement("div");
  badge.className = "ed-warn-badge";
  badge.textContent = "⚠ Potentially triggering content — click to reveal";
  badge.addEventListener("click", () => {
    el.style.filter = "none";
    el.removeAttribute("data-ed-verdict");
    badge.remove();
    counts.warn = Math.max(0, counts.warn - 1);
  });

  el.insertAdjacentElement("beforebegin", badge);
}

function applyHarmful(el) {
  if (el.dataset.edVerdict) return;
  el.dataset.edVerdict = "HARMFUL";
  counts.harmful++;

  const originalHTML = el.innerHTML;

  const card = document.createElement("div");
  card.className = "ed-intervention";
  card.innerHTML = `
    <p class="ed-intervention-title">🚨 This content may promote harmful behaviors</p>
    <p class="ed-intervention-body">
      Content related to eating disorders or extreme body image ideals has been hidden.
    </p>
  `;

  const btn = document.createElement("button");
  btn.className = "ed-show-anyway";
  btn.textContent = "Show anyway";
  btn.addEventListener("click", () => {
    el.innerHTML = originalHTML;
    el.removeAttribute("data-ed-verdict");
    counts.harmful = Math.max(0, counts.harmful - 1);
  });
  card.appendChild(btn);

  el.innerHTML = "";
  el.appendChild(card);
}

// ── Element candidate logic ────────────────────────────────────────────────

function isCandidate(el) {
  if (scanned.has(el)) return false;
  if (el.dataset.edVerdict) return false;
  if (el.closest("[data-ed-verdict]")) return false;

  const text = el.innerText?.trim() ?? "";
  if (text.length < MIN_TEXT_LEN || text.length > MAX_TEXT_LEN) return false;

  // Skip container divs that hold other block elements
  const blockChild = el.querySelector("p, div, article, section, blockquote");
  if (blockChild) return false;

  return true;
}

function seedPending() {
  const selector = "p, blockquote, figcaption, li, h2, h3, span, div";
  for (const el of document.querySelectorAll(selector)) {
    if (isCandidate(el)) {
      scanned.add(el); // mark immediately to avoid double-queueing
      pending.push(el);
    }
  }
}

// ── Classification ─────────────────────────────────────────────────────────

async function classifyElement(el) {
  if (!document.contains(el)) return;
  const text = el.innerText?.trim();
  if (!text) return;

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) return;
    const data = await res.json();

    if (data.verdict === "WARN") applyWarn(el);
    else if (data.verdict === "HARMFUL") applyHarmful(el);
  } catch {
    // API unavailable — fail silently
  }
}

async function processBatch() {
  const batch = pending.splice(0, BATCH_SIZE);
  for (const el of batch) {
    await classifyElement(el);
  }
}

// ── MutationObserver: pick up dynamically added content (infinite scroll) ─

const observer = new MutationObserver((mutations) => {
  for (const m of mutations) {
    for (const node of m.addedNodes) {
      if (node.nodeType !== Node.ELEMENT_NODE) continue;
      if (isCandidate(node)) {
        scanned.add(node);
        pending.push(node);
      }
      // Also check children of added nodes
      for (const child of node.querySelectorAll?.("p, blockquote, figcaption, li") ?? []) {
        if (isCandidate(child)) {
          scanned.add(child);
          pending.push(child);
        }
      }
    }
  }
});

observer.observe(document.body, { childList: true, subtree: true });

// ── Boot ───────────────────────────────────────────────────────────────────

seedPending();

setInterval(() => {
  seedPending();   // catch anything missed since last cycle
  processBatch();
}, SCAN_INTERVAL_MS);
