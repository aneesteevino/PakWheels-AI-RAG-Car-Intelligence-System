/* ══════════════════════════════════════════════════════════════════════════
   PakWheels RAG – Frontend App Logic
   ══════════════════════════════════════════════════════════════════════════ */

const API = window.location.origin; // same origin as backend

let sidebarOpen = true;

// ── Sidebar toggle ──────────────────────────────────────────────────────────
function toggleSidebar() {
  const sidebar = document.getElementById("sidebar");
  const main    = document.querySelector(".main");
  if (window.innerWidth <= 700) {
    sidebar.classList.toggle("open");
  } else {
    sidebarOpen = !sidebarOpen;
    sidebar.classList.toggle("hidden", !sidebarOpen);
    main.classList.toggle("expanded", !sidebarOpen);
  }
}

// ── Auto-resize textarea ────────────────────────────────────────────────────
function autoResize(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 140) + "px";
}

// ── Enter key handler ───────────────────────────────────────────────────────
function handleKey(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendQuery();
  }
}

// ── Format helpers ──────────────────────────────────────────────────────────
function fmtNumber(n) {
  if (n == null) return "—";
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + "M";
  if (n >= 1_000)     return (n / 1_000).toFixed(0)     + "K";
  return String(n);
}

function fmtPrice(p) {
  if (p == null) return "N/A";
  return "PKR " + fmtNumber(p);
}

function escHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// ── Tag colour per query type ───────────────────────────────────────────────
function queryTag(type) {
  const map = {
    average:"average", cheapest:"cheapest", expensive:"cheapest",
    filter:"filter", compare:"compare", list:"list", count:"count", general:"general"
  };
  const cls = "tag-" + (map[type] || "general");
  return `<span class="query-type-tag ${cls}">${escHtml(type)}</span>`;
}

// ── Load global stats ───────────────────────────────────────────────────────
async function loadStats() {
  try {
    const r = await fetch(`${API}/api/stats`);
    if (!r.ok) return;
    const d = await r.json();
    document.getElementById("statTotal").textContent  = (d.total_listings / 1000).toFixed(1) + "K";
    document.getElementById("statAvg").textContent    = fmtNumber(d.price_range?.avg);
    document.getElementById("statCities").textContent = d.cities?.length ?? "—";
    document.getElementById("statMakes").textContent  = Object.keys(d.makes ?? {}).length;
  } catch (_) { /* backend still starting */ }
}

// ── Load suggestions ────────────────────────────────────────────────────────
async function loadSuggestions() {
  try {
    console.log('Starting to load suggestions...');
    const r = await fetch(`${API}/api/suggestions`);
    if (!r.ok) {
      console.error('Failed to fetch suggestions:', r.status);
      return;
    }
    const { suggestions } = await r.json();
    console.log('Fetched suggestions:', suggestions);

    // Store suggestions globally for onclick handlers
    window.suggestionsList = suggestions;
    console.log('Stored suggestions globally:', window.suggestionsList);

    // Sidebar suggestions with data-index for event delegation
    const sidebar = document.getElementById("suggestions");
    if (!sidebar) {
      console.error('Sidebar element not found');
      return;
    }
    
    sidebar.innerHTML = suggestions.map((s, i) =>
      `<div class="suggestion-chip" data-index="${i}" style="cursor: pointer; border: 1px solid #333;">${escHtml(s)}</div>`
    ).join("");
    console.log('Sidebar suggestions HTML set');

    // Welcome chips (first 6) with data-index for event delegation
    const chips = document.getElementById("welcomeChips");
    if (chips) {
      chips.innerHTML = suggestions.slice(0, 6).map((s, i) =>
        `<div class="welcome-chip" data-index="${i}" style="cursor: pointer; border: 1px solid #333;">${escHtml(s)}</div>`
      ).join("");
      console.log('Welcome chips HTML set');
    }

    console.log('Suggestions loaded with index handlers');
  } catch (e) { 
    console.error('Error loading suggestions:', e);
  }
}

function useQueryByIndex(index) {
  console.log('useQueryByIndex called with index:', index);
  console.log('Available suggestions:', window.suggestionsList);
  if (window.suggestionsList && window.suggestionsList[index]) {
    console.log('Found suggestion at index:', window.suggestionsList[index]);
    useQuery(window.suggestionsList[index]);
  } else {
    console.error('No suggestion found at index:', index);
  }
}

function useQuery(q) {
  console.log('useQuery called with:', q);
  const input = document.getElementById("queryInput");
  if (!input) {
    console.error('Input element not found');
    return;
  }
  console.log('Setting input value to:', q);
  input.value = q;
  autoResize(input);
  input.focus();
  console.log('Input value is now:', input.value);
}

// ── Append messages ─────────────────────────────────────────────────────────
function appendUserMessage(text) {
  const chat = document.getElementById("chatArea");
  const welcome = document.getElementById("welcomeCard");
  if (welcome) welcome.remove();

  chat.innerHTML += `
    <div class="message user">
      <div class="avatar user">👤</div>
      <div class="bubble">${escHtml(text)}</div>
    </div>`;
  chat.scrollTop = chat.scrollHeight;
}

function appendTyping() {
  const chat = document.getElementById("chatArea");
  const id = "typing-" + Date.now();
  chat.innerHTML += `
    <div class="message" id="${id}">
      <div class="avatar ai">🤖</div>
      <div class="bubble">
        <div class="typing-dot"><span></span><span></span><span></span></div>
      </div>
    </div>`;
  chat.scrollTop = chat.scrollHeight;
  return id;
}

function removeTyping(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

// ── Build car card HTML ─────────────────────────────────────────────────────
function carCardHtml(car, idx) {
  return `
    <div class="car-card" onclick="openModal(${idx})">
      <div class="car-card-title">${escHtml(car.make)} ${escHtml(car.model)}</div>
      <div class="car-card-price">${escHtml(car.price_fmt || fmtPrice(car.price))}</div>
      <div class="car-card-meta">
        📍 ${escHtml(car.city)}<br>
        📅 ${car.year ?? "—"} &nbsp;·&nbsp; ⛽ ${escHtml(car.fuel_type)}<br>
        🚗 ${escHtml(car.transmission)} &nbsp;·&nbsp; 📏 ${car.mileage ? fmtNumber(car.mileage)+" km" : "—"}
      </div>
      <span class="car-card-badge">${escHtml(car.body || "Car")}</span>
    </div>`;
}

// ── Build stats pills ───────────────────────────────────────────────────────
function statsPillsHtml(stats) {
  if (!stats || !stats.count) return "";
  const pills = [];
  pills.push(`<div class="stats-pill">🔢 <strong>${stats.count.toLocaleString()}</strong> matches</div>`);
  if (stats.avg_price_fmt)
    pills.push(`<div class="stats-pill">📊 Avg <strong>${escHtml(stats.avg_price_fmt)}</strong></div>`);
  if (stats.min_price_fmt)
    pills.push(`<div class="stats-pill">⬇️ Min <strong>${escHtml(stats.min_price_fmt)}</strong></div>`);
  if (stats.max_price_fmt)
    pills.push(`<div class="stats-pill">⬆️ Max <strong>${escHtml(stats.max_price_fmt)}</strong></div>`);
  return `<div class="stats-bar">${pills.join("")}</div>`;
}

// ── Append AI response ──────────────────────────────────────────────────────
let _lastCars = [];

function appendAIMessage(data) {
  _lastCars = data.retrieved_cars || [];
  const chat = document.getElementById("chatArea");

  const carsHtml = _lastCars.length ? `
    <div class="cars-section">
      <div class="cars-section-title">🔍 RETRIEVED LISTINGS</div>
      <div class="car-cards">
        ${_lastCars.map((c, i) => carCardHtml(c, i)).join("")}
      </div>
    </div>` : "";

  // Format answer: convert **bold** and bullet dashes
  const answerFormatted = escHtml(data.answer)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/^- /gm, "• ");

  chat.innerHTML += `
    <div class="message" id="ai-${Date.now()}">
      <div class="avatar ai">🤖</div>
      <div class="bubble">
        ${queryTag(data.query_type)}
        ${statsPillsHtml(data.stats)}
        <div class="answer-text">${answerFormatted}</div>
        ${carsHtml}
        <button class="explain-btn" onclick="explainResult()">💡 Explain this result</button>
        <div style="font-size:11px;color:var(--text3);margin-top:8px">
          ⚡ ${data.elapsed_ms}ms
        </div>
      </div>
    </div>`;
  chat.scrollTop = chat.scrollHeight;
}

function appendErrorMessage(msg) {
  const chat = document.getElementById("chatArea");
  chat.innerHTML += `
    <div class="message">
      <div class="avatar ai">🤖</div>
      <div class="bubble"><div class="error-msg">⚠️ ${escHtml(msg)}</div></div>
    </div>`;
  chat.scrollTop = chat.scrollHeight;
}

// ── Send query ──────────────────────────────────────────────────────────────
async function sendQuery() {
  const input = document.getElementById("queryInput");
  const btn   = document.getElementById("sendBtn");
  const query = input.value.trim();
  if (!query) return;

  input.value = "";
  autoResize(input);
  btn.disabled = true;

  appendUserMessage(query);
  const typingId = appendTyping();

  try {
    const resp = await fetch(`${API}/api/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 5 }),
    });

    removeTyping(typingId);

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      appendErrorMessage(err.detail || `Server error: ${resp.status}`);
      return;
    }

    const data = await resp.json();
    appendAIMessage(data);
  } catch (e) {
    removeTyping(typingId);
    appendErrorMessage("Could not reach the server. Make sure the backend is running on port 8000.");
  } finally {
    btn.disabled = false;
    input.focus();
  }
}

// ── Explain result ──────────────────────────────────────────────────────────
async function explainResult() {
  if (!_lastCars.length) return;
  const query = "Explain in detail why these cars were retrieved and how the prices compare in the market.";
  document.getElementById("queryInput").value = query;
  await sendQuery();
}

// ── Car modal ───────────────────────────────────────────────────────────────
function openModal(idx) {
  const car = _lastCars[idx];
  if (!car) return;

  const fields = [
    ["Make",          car.make],
    ["Model",         car.model],
    ["Year",          car.year],
    ["City",          car.city],
    ["Body Type",     car.body],
    ["Engine",        car.engine_cc ? car.engine_cc + "cc" : "—"],
    ["Fuel Type",     car.fuel_type],
    ["Transmission",  car.transmission],
    ["Assembly",      car.assembly],
    ["Color",         car.color],
    ["Registered",    car.registered],
    ["Mileage",       car.mileage ? fmtNumber(car.mileage) + " km" : "—"],
  ];

  document.getElementById("modalContent").innerHTML = `
    <div class="modal-car-title">${escHtml(car.make)} ${escHtml(car.model)}</div>
    <div class="modal-price">${escHtml(car.price_fmt || fmtPrice(car.price))}</div>
    <div class="modal-grid">
      ${fields.map(([lbl, val]) => `
        <div class="modal-item">
          <label>${escHtml(lbl)}</label>
          <span>${escHtml(String(val ?? "—"))}</span>
        </div>`).join("")}
    </div>
    <span class="modal-chip">Ad Ref: ${escHtml(String(car.ad_ref))}</span>`;

  document.getElementById("modalOverlay").classList.add("open");
}

function closeModal() {
  document.getElementById("modalOverlay").classList.remove("open");
}

// ── Init ────────────────────────────────────────────────────────────────────
(async () => {
  console.log('Initializing app...');
  
  // Add global click handler for suggestion chips
  document.addEventListener('click', function(e) {
    if (e.target.classList.contains('suggestion-chip') || e.target.classList.contains('welcome-chip')) {
      console.log('Chip clicked via event delegation');
      const index = parseInt(e.target.getAttribute('data-index'));
      console.log('Index from data-index:', index);
      if (!isNaN(index) && window.suggestionsList && window.suggestionsList[index]) {
        console.log('Using suggestion:', window.suggestionsList[index]);
        useQuery(window.suggestionsList[index]);
      }
    }
  });
  
  await Promise.all([loadStats(), loadSuggestions()]);
  document.getElementById("queryInput").focus();
  console.log('App initialized');
})();