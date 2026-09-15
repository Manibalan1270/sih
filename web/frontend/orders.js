/* ROBOTON task announcement. Posts work to the gateway; names no robot. */

(() => {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const $ = (id) => document.getElementById(id);

  const layers = {
    lanes: $("layer-lanes"),
    nodes: $("layer-nodes"),
    robots: $("layer-robots"),
  };
  const svg = $("map");
  const tbody = document.querySelector("#task-table tbody");

  let nodesById = new Map();
  let stations = [];
  let stationEls = new Map();
  let robotDots = new Map();
  let mine = new Set();
  let pickup = null;
  let drop = null;
  let live = false;

  function el(tag, attrs = {}, parent = null) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  }

  const clear = (node) => { while (node.firstChild) node.removeChild(node.firstChild); };
  const nameOf = (id) => {
    const node = nodesById.get(id);
    return node ? node.name : `node ${id}`;
  };

  function note(text, kind) {
    const box = $("order-note");
    box.textContent = text;
    box.className = kind ? `note ${kind}` : "note";
  }

  // ---- the floor ----------------------------------------------------------

  function drawMap(data) {
    nodesById = new Map(data.nodes.map((n) => [n.id, n]));
    for (const layer of Object.values(layers)) clear(layer);
    stationEls = new Map();
    robotDots = new Map();

    const xs = data.nodes.map((n) => n.x);
    const ys = data.nodes.map((n) => n.y);
    const pad = 3000;
    const minX = Math.min(...xs) - pad, maxX = Math.max(...xs) + pad;
    const minY = Math.min(...ys) - pad, maxY = Math.max(...ys) + pad;
    svg.setAttribute("viewBox", `${minX} ${minY} ${maxX - minX} ${maxY - minY}`);

    const laneWidth = data.lane_offset_mm * 2 + 800;
    for (const e of data.edges) {
      const u = nodesById.get(e.u), v = nodesById.get(e.v);
      if (!u || !v) continue;
      el("line", {
        x1: u.x, y1: u.y, x2: v.x, y2: v.y,
        class: `lane${e.single_lane ? " single" : ""}`,
        "stroke-width": e.single_lane ? 1400 : laneWidth,
      }, layers.lanes);
    }

    for (const n of data.nodes) {
      if (n.pickup || n.drop || n.parking || n.charger) {
        const cls = ["station", n.pickup ? "pickup" : "", n.drop ? "drop" : "",
          n.parking ? "parking" : ""].join(" ").trim();
        const rect = el("rect", {
          class: cls, x: n.x - 700, y: n.y - 700, width: 1400, height: 1400,
        }, layers.nodes);
        if (!n.parking && !n.charger) {
          rect.classList.add("selectable");
          rect.addEventListener("click", () => choose(n.id));
          stationEls.set(n.id, rect);
        }
        if (n.pickup || n.drop) {
          el("text", { class: "station-label", x: n.x, y: n.y + 1600 }, layers.nodes)
            .textContent = n.name;
        }
      } else if (n.junction) {
        el("circle", { class: "node junction", cx: n.x, cy: n.y, r: 200 }, layers.nodes);
      }
    }

    $("empty").hidden = true;
    paint();
  }

  function drawRobots(robots) {
    const seen = new Set();
    for (const r of robots) {
      let dot = robotDots.get(r.id);
      if (!dot) {
        dot = el("circle", { r: 460, class: "robot-dot" }, layers.robots);
        robotDots.set(r.id, dot);
      }
      dot.setAttribute("cx", r.x);
      dot.setAttribute("cy", r.y);
      dot.setAttribute("class", `robot-dot ${r.state}`);
      seen.add(r.id);
    }
    for (const [id, dot] of robotDots) {
      if (!seen.has(id)) { dot.remove(); robotDots.delete(id); }
    }
  }

  // ---- choosing a journey -------------------------------------------------

  function choose(id) {
    // Two clicks describe a journey. Clicking a chosen end clears it rather
    // than making a zero-length task the gateway would only refuse.
    if (pickup === id) pickup = null;
    else if (drop === id) drop = null;
    else if (pickup === null) pickup = id;
    else drop = id;
    paint();
  }

  function paint() {
    for (const [id, rect] of stationEls) {
      rect.classList.toggle("is-pickup", id === pickup);
      rect.classList.toggle("is-drop", id === drop);
    }
    $("pickup").value = pickup === null ? "" : String(pickup);
    $("drop").value = drop === null ? "" : String(drop);

    const from = $("pickup-name");
    from.textContent = pickup === null ? "not set" : nameOf(pickup);
    from.classList.toggle("unset", pickup === null);
    const to = $("drop-name");
    to.textContent = drop === null ? "not set" : nameOf(drop);
    to.classList.toggle("unset", drop === null);

    $("submit-order").disabled = !live || pickup === null || drop === null;
  }

  function fillSelects() {
    for (const which of ["pickup", "drop"]) {
      const select = $(which);
      clear(select);
      const blank = document.createElement("option");
      blank.value = "";
      blank.textContent = stations.length ? "Pick a station" : "No run in progress";
      select.appendChild(blank);
      for (const s of stations) {
        const option = document.createElement("option");
        option.value = String(s.node);
        option.textContent = `${s.name} · ${s.kind}`;
        select.appendChild(option);
      }
    }
    paint();
  }

  async function loadStations() {
    try {
      const res = await fetch("/api/orders/stations");
      stations = res.ok ? (await res.json()).stations : [];
    } catch {
      stations = [];
    }
    fillSelects();
  }

  async function loadMine() {
    try {
      const res = await fetch("/api/orders");
      if (res.ok) mine = new Set((await res.json()).orders.map((o) => o.task_id));
    } catch {
      /* the badge is a convenience; the run is unaffected if it cannot load */
    }
  }

  // ---- work in the system -------------------------------------------------

  const ORDER = { HELD: 0, PENDING: 1, SCHEDULED: 2, COMPLETED: 3 };

  function renderTasks(tasks) {
    const sorted = [...tasks].sort((a, b) =>
      ((ORDER[a.status] ?? 9) - (ORDER[b.status] ?? 9)) || b.id - a.id);
    clear(tbody);
    for (const t of sorted.slice(0, 150)) {
      const row = document.createElement("tr");
      row.innerHTML =
        `<td>#${t.id}${mine.has(t.id) ? '<span class="tag">yours</span>' : ""}</td>` +
        `<td>${nameOf(t.pickup)} &rarr; ${nameOf(t.drop)}</td>` +
        `<td>${t.priority}</td>` +
        `<td><span class="pill ${t.status}">${t.status}</span></td>` +
        `<td>${t.holder === null || t.holder === undefined ? "—" : t.holder}</td>`;
      tbody.appendChild(row);
    }
  }

  // ---- frames -------------------------------------------------------------

  function applyFrame(f) {
    if (!live) { live = true; loadStations(); }
    $("run-line").textContent =
      `${f.scenario} · seed ${f.seed} · ${f.robots.length} AMRs`;
    setLamp(
      f.finished ? "done" : f.paused ? "paused" : "live",
      f.finished ? "finished" : f.paused ? "paused" : "live",
    );

    drawRobots(f.robots);
    renderTasks(f.tasks || []);

    if (f.finished) {
      live = false;
      paint();
      if (!$("order-note").textContent) {
        note("This run has finished. Start another to announce more work.", null);
      }
    } else {
      paint();
    }
  }

  function goIdle() {
    live = false;
    stations = [];
    fillSelects();
    $("empty").hidden = false;
    $("run-line").textContent = "Nothing running yet";
    setLamp("idle", "no run");
  }

  let linkState = "live";
  let lamp = { state: "idle", label: "no run" };

  function setLamp(state, label) {
    lamp = { state, label };
    paintLamp();
  }

  /* The lamp reports the run, except when the link to it is broken -- then it
     reports that instead, because a stale frame shown as "live" is a lie. */
  function paintLamp() {
    let { state, label } = lamp;
    if (linkState === "offline") { state = "error"; label = "no telemetry"; }
    else if (linkState === "polling") { label = `${label} \u00b7 polling`; }
    $("lamp").dataset.state = state;
    $("lamp-label").textContent = label;
    $("run-status").textContent = label;
  }

  function connect() {
    Telemetry.connect({
      onMap: (data) => { drawMap(data); loadStations(); },
      onFrame: applyFrame,
      onIdle: goIdle,
      onLink: (state) => { linkState = state; paintLamp(); },
    });
  }

  // ---- announcing ---------------------------------------------------------

  $("order-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (pickup === null || drop === null) {
      note("Choose where to collect and where to deliver.", "error");
      return;
    }
    const body = { pickup, drop, priority: Number($("priority").value) };
    const res = await fetch("/api/orders", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = typeof data.detail === "string" ? data.detail : `Refused (${res.status}).`;
      note(detail, "error");
      return;
    }
    note(`Task #${data.order.task_id} announced: ${nameOf(body.pickup)} → ${nameOf(body.drop)}. The fleet is bidding.`, "ok");
    pickup = drop = null;
    paint();
    await loadMine();
  });

  $("clear-order").addEventListener("click", () => {
    pickup = drop = null;
    paint();
    note("", null);
  });

  for (const which of ["pickup", "drop"]) {
    $(which).addEventListener("change", () => {
      const value = $(which).value === "" ? null : Number($(which).value);
      if (which === "pickup") pickup = value; else drop = value;
      paint();
    });
  }

  fillSelects();
  loadMine();
  connect();
})();
