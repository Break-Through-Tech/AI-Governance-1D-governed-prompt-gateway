// Prompt gateway frontend. Talks to the FastAPI backend in src/backend.
// Served by the backend at "/", or open index.html directly and pass ?api=http://host:8000
"use strict";

const params = new URLSearchParams(location.search);
const API = (params.get("api") || (location.protocol === "file:" ? "http://localhost:8000" : "")).replace(/\/$/, "");
const WS_URL = (API ? API.replace(/^http/, "ws") : `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}`) + "/v1/moderate/stream";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const POLICIES = {
  any: "Either model flags it",
  all: "Both models flag it",
  stream: "Qwen3Guard-Stream flags it",
  lora: "The LoRA classifier flags it",
};
const START_HINT = "Start it from the repo root with: uvicorn src.backend.main:app --port 8000";

let health = null;

// ---------------------------------------------------------------- api
async function api(path, opts = {}) {
  let res;
  try {
    res = await fetch(API + path, {
      ...opts,
      headers: { "content-type": "application/json", ...(opts.headers || {}) },
    });
  } catch {
    throw new Error(`Can't reach the gateway at ${API || location.origin}. ${START_HINT}`);
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = body.detail;
    throw new Error(typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => x.msg).join("; ") : `Request failed (${res.status})`);
  }
  return body;
}

// ---------------------------------------------------------------- health
function setPill(el, m) {
  const state = !m ? "error" : !m.enabled ? "off" : m.loaded ? "ok" : m.error ? "error" : "loading";
  el.dataset.state = state;
  const what = { ok: `loaded on ${m?.device || "?"}`, off: "turned off", error: m?.error || "unreachable", loading: "loading" }[state];
  el.title = `${el.textContent.trim()}: ${what}${m?.adapter ? ` (adapter ${m.adapter})` : ""}`;
}

async function pollHealth() {
  let delay = 30000;
  try {
    health = await api("/health");
    setPill($("#pill-stream"), health.stream);
    setPill($("#pill-lora"), health.lora);
    if (health.status === "loading") delay = 3000;
    $$(".policy-select").forEach((s) => { if (!s.dataset.touched) s.value = health.policy; });
  } catch {
    health = null;
    setPill($("#pill-stream"), null);
    setPill($("#pill-lora"), null);
    delay = 5000;
  }
  setTimeout(pollHealth, delay);
}

// ---------------------------------------------------------------- tabs
function showTab(btn) {
  $$(".tabs button").forEach((b) => {
    const on = b === btn;
    b.setAttribute("aria-selected", on);
    b.tabIndex = on ? 0 : -1;
    $("#" + b.getAttribute("aria-controls")).hidden = !on;
  });
  if (btn.id === "tab-log") loadLog();
  try { localStorage.setItem("gateway-tab", btn.id); } catch {}
}

// ---------------------------------------------------------------- check a prompt
function levelBadge(level) {
  return `<span class="level ${esc(level)}">${esc(level)}</span>`;
}

function whySentence(d) {
  if (d.blocked) return d.reasons.join(". ") + ".";
  const who = { any: "Neither model flagged it", all: "The models did not both flag it", stream: "Qwen3Guard-Stream did not flag it", lora: "The LoRA classifier did not flag it" };
  return `${who[d.policy]}, so the gateway lets it through.`;
}

function renderVerdict(d, threshold) {
  const s = d.stream, l = d.lora;
  const streamCell = s
    ? `${levelBadge(s.worst_level)}${s.category ? ` <span>${esc(s.category)}</span>` : ""}
       <div class="foot" style="margin-top:6px">
         <span>Prompt: ${esc(s.prompt_level)}</span>
         ${s.response_tokens_scored ? `<span>Reply tokens scored: ${s.response_tokens_scored}</span>` : ""}
         ${s.flagged_at_token != null ? `<span>Flagged at reply token ${s.flagged_at_token + 1}</span>` : ""}
       </div>`
    : `<span class="level off">Not loaded</span>`;
  const loraCell = l
    ? `${levelBadge(l.label === "unsafe" ? "Unsafe" : "Safe")} P(unsafe) ${l.unsafe_score.toFixed(3)}
       <div class="meter" role="img" aria-label="P(unsafe) ${l.unsafe_score.toFixed(2)} against threshold ${threshold.toFixed(2)}">
         <b style="width:${(l.unsafe_score * 100).toFixed(1)}%;background:${l.unsafe ? "var(--unsafe)" : "var(--safe)"}"></b>
         <span class="thr" style="left:${threshold * 100}%"></span>
       </div>`
    : `<span class="level off">Not loaded</span>`;

  $("#verdict").innerHTML = `
    <p class="decision ${d.decision}">${d.blocked ? "Blocked" : "Allowed"}</p>
    <p class="decision-why">${esc(whySentence(d))}</p>
    <table class="votes">
      <tr><th>Qwen3Guard-Stream<small>Reads the prompt and reply, token by token</small></th><td>${streamCell}</td></tr>
      <tr><th>LoRA classifier<small>Qwen3Guard-Gen tuned on our labels, prompt only</small></th><td>${loraCell}</td></tr>
    </table>
    <div class="foot"><span>Policy: ${esc(POLICIES[d.policy])}</span><span>${d.latency_ms} ms</span><span>Request <code>${esc(d.request_id)}</code></span></div>`;
}

async function onCheck(e) {
  e.preventDefault();
  const btn = $("#check-submit");
  const prompt = $("#check-prompt").value.trim();
  if (!prompt) return;
  const threshold = parseFloat($("#check-threshold").value);
  btn.disabled = true;
  btn.textContent = "Checking…";
  try {
    const d = await api("/v1/moderate", {
      method: "POST",
      body: JSON.stringify({
        prompt,
        response: $("#check-response").value,
        policy: $("#check-policy").value,
        threshold,
      }),
    });
    renderVerdict(d, threshold);
  } catch (err) {
    $("#verdict").innerHTML = `<p class="error">${esc(err.message)}</p>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Check prompt";
  }
}

// ---------------------------------------------------------------- watch a reply
const SAMPLE_REPLIES = {
  safe: "Maya loved the quiet of the lab after class. She lined up her beakers, checked her notes twice, and finally watched the solution turn a perfect shade of blue. Her teacher smiled and said it was the cleanest titration she had seen all year.",
  turns: "Maya loved the quiet of the lab after class. She lined up her beakers and checked her notes. Then she decided to show her friends how to make a bomb from fertilizer and diesel, starting with the exact amounts of each ingredient and how to pack them.",
};

class Channel {
  constructor(url) {
    this.queue = []; this.waiters = [];
    this.ws = new WebSocket(url);
    this.ws.onmessage = (ev) => this._push(JSON.parse(ev.data));
    this.ws.onclose = () => this._push({ type: "closed" });
    this.opened = new Promise((ok, fail) => {
      this.ws.onopen = ok;
      this.ws.onerror = () => fail(new Error(`Can't open a live connection to ${url}. ${START_HINT}`));
    });
  }
  _push(m) { const w = this.waiters.shift(); w ? w(m) : this.queue.push(m); }
  next() { return this.queue.length ? Promise.resolve(this.queue.shift()) : new Promise((r) => this.waiters.push(r)); }
  send(obj) { this.ws.send(JSON.stringify(obj)); }
  close() { try { this.ws.close(); } catch {} }
}

function liveStatus(text, cls = "") {
  const el = $("#live-state");
  el.textContent = text;
  el.className = "state " + cls;
}

async function playReply() {
  const btn = $("#live-play");
  const prompt = $("#live-prompt").value.trim();
  const words = $("#live-reply").value.trim().split(/\s+/).filter(Boolean);
  if (!prompt) return;
  const delay = parseInt($("#live-speed").value, 10);
  const out = $("#stream-text");
  const ribbon = $("#ribbon");
  out.innerHTML = "";
  ribbon.innerHTML = words.map(() => `<span class="pending"></span>`).join("");
  $("#live-tokens").textContent = "0";
  $("#live-worst").textContent = "–";
  btn.disabled = true;
  liveStatus("Checking the prompt…");

  const ch = new Channel(WS_URL);
  const withhold = (from) => {
    if (from < words.length) {
      const rest = document.createElement("span");
      rest.className = "withheld";
      rest.textContent = words.slice(from).join(" ");
      rest.title = "Never sent to the customer";
      out.append(rest);
    }
  };
  const cut = (text) => { const c = document.createElement("span"); c.className = "cut"; c.textContent = text; out.append(c); };

  try {
    await ch.opened;
    ch.send({ prompt, policy: $("#live-policy").value });
    const first = await ch.next();
    if (first.type === "error") throw new Error(first.detail);
    if (first.blocked) {
      cut(`Prompt blocked before the LLM ran. ${first.reasons.join(". ")}.`);
      withhold(0);
      liveStatus("Blocked at the prompt", "block");
      return;
    }
    liveStatus("Streaming");

    const slots = $$("span", ribbon);
    for (let i = 0; i < words.length; i++) {
      ch.send({ chunk: (i ? " " : "") + words[i] });
      const m = await ch.next();
      if (m.type === "error" || m.type === "closed") throw new Error(m.detail || "The connection closed unexpectedly.");
      const level = m.scored ? m.worst_level : "Safe";
      const tok = document.createElement("span");
      tok.className = "tok " + level;
      tok.textContent = (i ? " " : "") + words[i];
      out.append(tok);
      slots[i].className = level;
      if (m.scored) {
        $("#live-tokens").textContent = m.tokens_scored;
        $("#live-worst").innerHTML = levelBadge(m.worst_level);
      }
      if (m.stop) {
        const fin = await ch.next();
        cut(`Reply stopped here. ${(fin.reasons || []).join(". ")}.`);
        withhold(i + 1);
        liveStatus("Stopped mid-reply", "block");
        return;
      }
      await sleep(delay);
    }
    ch.send({ done: true });
    const fin = await ch.next();
    liveStatus(fin.blocked ? "Blocked" : "Delivered in full", fin.blocked ? "block" : "allow");
  } catch (err) {
    out.innerHTML = `<p class="error" style="font-family:var(--sans)">${esc(err.message)}</p>`;
    liveStatus("Not connected");
  } finally {
    ch.close();
    btn.disabled = false;
  }
}

// ---------------------------------------------------------------- batch
function summary(el, items) {
  el.innerHTML = items.map(([v, label]) => `<div><b>${esc(v)}</b><span>${esc(label)}</span></div>`).join("");
}

async function runBatch() {
  const btn = $("#batch-run");
  const prompts = $("#batch-text").value.split("\n").map((s) => s.trim()).filter(Boolean);
  const table = $("#batch-table");
  if (!prompts.length) return;
  btn.disabled = true;
  btn.textContent = `Checking ${prompts.length}…`;
  try {
    const r = await api("/v1/moderate/batch", {
      method: "POST",
      body: JSON.stringify({ items: prompts.map((p) => ({ prompt: p })), policy: $("#batch-policy").value }),
    });
    const blocked = r.results.filter((d) => d.blocked).length;
    summary($("#batch-summary"), [
      [r.results.length - blocked, "allowed"],
      [blocked, "blocked"],
      [`${Math.max(...r.results.map((d) => d.latency_ms))} ms`, "total time"],
    ]);
    $("tbody", table).innerHTML = r.results.map((d, i) => `
      <tr>
        <td class="prompt">${esc(prompts[i])}</td>
        <td><span class="level ${d.blocked ? "Unsafe" : "Safe"}">${d.blocked ? "Blocked" : "Allowed"}</span></td>
        <td>${d.stream ? levelBadge(d.stream.worst_level) : "–"}</td>
        <td class="num">${d.lora ? d.lora.unsafe_score.toFixed(3) : "–"}</td>
      </tr>`).join("");
    table.hidden = false;
  } catch (err) {
    $("#batch-summary").innerHTML = `<p class="error">${esc(err.message)}</p>`;
    table.hidden = true;
  } finally {
    btn.disabled = false;
    btn.textContent = "Run batch";
  }
}

// ---------------------------------------------------------------- log
const SOURCES = { "/v1/moderate": "Check", "/v1/moderate/batch": "Batch", "/v1/moderate/stream": "Live reply" };

async function loadLog() {
  const table = $("#log-table");
  try {
    const r = await api("/v1/audit?limit=200");
    const recs = r.records;
    if (!r.enabled) {
      summary($("#log-summary"), []);
      $("#log-summary").innerHTML = `<p class="explain">The audit log is turned off. Set GUARD_AUDIT_LOG to a file path to record decisions.</p>`;
      table.hidden = true;
      return;
    }
    if (!recs.length) {
      $("#log-summary").innerHTML = `<p class="explain">No decisions yet. Check a prompt and it will show up here.</p>`;
      table.hidden = true;
      return;
    }
    const blocked = recs.filter((x) => x.decision === "block").length;
    const lat = recs.map((x) => x.latency_ms).sort((a, b) => a - b);
    summary($("#log-summary"), [
      [recs.length, recs.length === 200 ? "most recent decisions" : "decisions"],
      [`${Math.round((100 * blocked) / recs.length)}%`, "blocked"],
      [`${lat[Math.floor(lat.length / 2)]} ms`, "median latency"],
    ]);
    $("tbody", table).innerHTML = recs.map((x) => `
      <tr>
        <td class="num">${esc(new Date(x.time).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" }))}</td>
        <td>${esc(SOURCES[x.endpoint] || x.endpoint)}</td>
        <td><span class="level ${x.decision === "block" ? "Unsafe" : "Safe"}">${x.decision === "block" ? "Blocked" : "Allowed"}</span></td>
        <td>${x.reasons.length ? esc(x.reasons.join("; ")) : "–"}</td>
        <td class="num">${x.latency_ms} ms</td>
        <td><code title="Prompt fingerprint ${esc(x.prompt_sha256)}">${esc(x.request_id)}</code></td>
      </tr>`).join("");
    table.hidden = false;
  } catch (err) {
    $("#log-summary").innerHTML = `<p class="error">${esc(err.message)}</p>`;
    table.hidden = true;
  }
}

// ---------------------------------------------------------------- wire up
function init() {
  $$(".policy-select").forEach((s) => {
    s.innerHTML = Object.entries(POLICIES).map(([v, t]) => `<option value="${v}">${t}</option>`).join("");
    s.addEventListener("change", () => { s.dataset.touched = "1"; });
  });

  const tabs = $$(".tabs button");
  tabs.forEach((b, i) => {
    b.addEventListener("click", () => showTab(b));
    b.addEventListener("keydown", (e) => {
      const d = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
      if (d) { const t = tabs[(i + d + tabs.length) % tabs.length]; t.focus(); showTab(t); }
    });
  });
  let saved = null;
  try { saved = localStorage.getItem("gateway-tab"); } catch {}
  showTab($("#" + (params.get("tab") ? "tab-" + params.get("tab") : saved)) || tabs[0]);

  $$("[data-fill]").forEach((b) => b.addEventListener("click", () => { $("#check-prompt").value = b.dataset.fill; $("#check-prompt").focus(); }));
  $$("[data-reply]").forEach((b) => b.addEventListener("click", () => { $("#live-reply").value = SAMPLE_REPLIES[b.dataset.reply]; }));
  $("#live-reply").value = SAMPLE_REPLIES.turns;

  const thr = $("#check-threshold");
  thr.addEventListener("input", () => { $("#check-threshold-out").textContent = parseFloat(thr.value).toFixed(2); });

  $("#check-form").addEventListener("submit", onCheck);
  $("#live-play").addEventListener("click", playReply);
  $("#batch-run").addEventListener("click", runBatch);
  $("#log-refresh").addEventListener("click", loadLog);

  pollHealth();
}

init();
