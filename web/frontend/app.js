/* ROBOTON fleet view. Reads frames from the backend; sends nothing to any robot. */

(() => {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const $ = (id) => document.getElementById(id);

  const layers = {
    zones: $("layer-zones"),
    lanes: $("layer-lanes"),
    nodes: $("layer-nodes"),
    routes: $("layer-routes"),
    waits: $("layer-waits"),
    robots: $("layer-robots"),
  };
  const svg = $("map");
  const tbody = document.querySelector("#fleet-table tbody");

  let map = null;               // geometry as sent by /api/map
  let nodesById = new Map();
  let edgesById = new Map();
  let robotEls = new Map();     // robot id -> { g, body, heading, battery, label, row }
  let selected = null;          // robot id whose route is drawn
  let lastHeading = new Map();
  let running = false;
  let paused = false;

  // ---- helpers ------------------------------------------------------------

  function el(tag, attrs = {}, parent = null) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function fmtClock(ms) {
    const s = ms / 1000;
    const m = Math.floor(s / 60);
    const rest = (s - m * 60).toFixed(1).padStart(4, "0");
    return `${String(m).padStart(2, "0")}:${rest}`;
  }

  function fmtSeconds(ms) {
    return `${Math.round(ms / 1000)} s`;
  }

  // ---- floor plan ---------------------------------------------------------

  function drawMap(data) {
    map = data;
    nodesById = new Map(data.nodes.map((n) => [n.id, n]));
    edgesById = new Map(data.edges.map((e) => [e.id, e]));
    for (const layer of Object.values(layers)) clear(layer);
    robotEls = new Map();
    clear(tbody);
    selected = null;

    const xs = data.nodes.map((n) => n.x);
    const ys = data.nodes.map((n) => n.y);
    const pad = 3000;
    const minX = Math.min(...xs) - pad, maxX = Math.max(...xs) + pad;
    const minY = Math.min(...ys) - pad, maxY = Math.max(...ys) + pad;
    svg.setAttribute("viewBox", `${minX} ${minY} ${maxX - minX} ${maxY - minY}`);

    const laneWidth = data.lane_offset_mm * 2 + 800;
    const singleWidth = 1400;

    // Zones as floor tint, one rectangle per zone's bounding box.
    if (data.zones.length) {
      const bounds = new Map();
      for (const n of data.nodes) {
        if (n.zone < 0) continue;
        const b = bounds.get(n.zone) || { x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity };
        b.x0 = Math.min(b.x0, n.x); b.y0 = Math.min(b.y0, n.y);
        b.x1 = Math.max(b.x1, n.x); b.y1 = Math.max(b.y1, n.y);
        bounds.set(n.zone, b);
      }
      for (const [zone, b] of [...bounds.entries()].sort((a, c) => a[0] - c[0])) {
        el("rect", {
          class: "zone", x: b.x0 - 1500, y: b.y0 - 1500,
          width: b.x1 - b.x0 + 3000, height: b.y1 - b.y0 + 3000, "data-zone": zone,
        }, layers.zones);
      }
    }

    // Lanes: a band of concrete, a dashed centre line for two-lane aisles,
    // hazard hatching along a single-lane corridor.
    for (const e of data.edges) {
      const u = nodesById.get(e.u), v = nodesById.get(e.v);
      const attrs = { x1: u.x, y1: u.y, x2: v.x, y2: v.y };
      const cls = ["lane", e.single_lane ? "single" : "", e.choke ? "choke" : ""].join(" ").trim();
      el("line", { ...attrs, class: cls, "stroke-width": e.single_lane ? singleWidth : laneWidth }, layers.lanes);
      if (e.single_lane) {
        el("line", { ...attrs, class: "lane-hazard" }, layers.lanes);
      } else {
        el("line", { ...attrs, class: "lane-centre" }, layers.lanes);
      }
    }

    // Nodes: junction dots, marker rings, and stations drawn as floor squares.
    for (const n of data.nodes) {
      const isStation = n.pickup || n.drop || n.parking || n.charger;
      if (isStation) {
        const cls = ["station", n.pickup ? "pickup" : "", n.drop ? "drop" : "",
          n.parking ? "parking" : ""].join(" ").trim();
        el("rect", { class: cls, x: n.x - 700, y: n.y - 700, width: 1400, height: 1400 }, layers.nodes);
        if (n.charger) {
          // A charger is a fixture in a bay, not a different kind of bay.
          el("circle", { class: "charger-mark", cx: n.x, cy: n.y, r: 260 }, layers.nodes);
        }
        if (n.pickup || n.drop) {
          el("text", { class: "station-label", x: n.x, y: n.y + 1500 }, layers.nodes).textContent = n.name;
        }
      } else if (n.junction) {
        el("circle", { class: "node junction", cx: n.x, cy: n.y, r: 220 }, layers.nodes);
      } else {
        el("circle", { class: "node", cx: n.x, cy: n.y, r: 160 }, layers.nodes);
      }
      if (n.marker) el("circle", { class: "node marker", cx: n.x, cy: n.y, r: 420 }, layers.nodes);
    }

    $("empty").hidden = true;
  }

  // ---- robots -------------------------------------------------------------

  const BODY_R = 450;
  const RING_R = 640;
  const RING_LEN = 2 * Math.PI * RING_R;

  function makeRobot(r) {
    const g = el("g", { class: "robot" }, layers.robots);
    const track = el("circle", { class: "battery-track", r: RING_R }, g);
    const battery = el("circle", {
      class: "battery", r: RING_R, "stroke-dasharray": `${RING_LEN} ${RING_LEN}`,
      transform: "rotate(-90)",
    }, g);
    const body = el("circle", { class: "body", r: BODY_R }, g);
    const cargo = el("rect", { class: "cargo", x: -170, y: -170, width: 340, height: 340 }, g);
    const heading = el("line", { class: "heading", x1: 0, y1: 0, x2: BODY_R + 250, y2: 0 }, g);
    const label = el("text", { class: "label" }, g);
    label.textContent = r.id;
    g.addEventListener("click", () => select(r.id));

    const row = document.createElement("tr");
    row.innerHTML = "<td></td><td></td><td></td><td></td><td></td>";
    row.addEventListener("click", () => select(r.id));
    tbody.appendChild(row);

    const entry = { g, body, battery, heading, label, row, cargo, track };
    robotEls.set(r.id, entry);
    return entry;
  }

  function laneOffset(r) {
    // Mirror Robot.footprint_mm: keep right on a two-lane aisle so opposing
    // traffic is drawn apart; on a single lane there is only the centre line.
    if (r.edge === null || r.next === null) return [0, 0];
    const e = edgesById.get(r.edge);
    if (!e || e.single_lane) return [0, 0];
    const a = nodesById.get(r.node), b = nodesById.get(r.next);
    if (!a || !b) return [0, 0];
    const dx = b.x - a.x, dy = b.y - a.y;
    const len = Math.max(1, Math.hypot(dx, dy));
    const off = map.lane_offset_mm;
    return [-dy * off / len, dx * off / len];
  }

  function updateRobot(r) {
    const entry = robotEls.get(r.id) || makeRobot(r);
    const [ox, oy] = laneOffset(r);
    const x = r.x + ox, y = r.y + oy;
    let heading = r.heading;
    if (heading < 0) heading = lastHeading.get(r.id) ?? 0;
    lastHeading.set(r.id, heading);

    entry.g.setAttribute("transform", `translate(${x} ${y})`);
    entry.heading.setAttribute("transform", `rotate(${heading})`);
    const low = r.battery <= 20;
    entry.g.setAttribute("class",
      `robot ${r.state}${low ? " low" : ""}${r.task !== null && r.leg === "TO_DROP" ? " carrying" : ""}${selected === r.id ? " selected" : ""}`);
    entry.battery.setAttribute("stroke-dashoffset", RING_LEN * (1 - r.battery / 100));

    const cells = entry.row.children;
    cells[0].textContent = r.id;
    cells[1].innerHTML = `<span class="pill ${r.state}">${r.state}</span>`;
    cells[2].innerHTML = `<span class="bat"><span class="bat-track"><span class="bat-fill${low ? " low" : ""}" style="width:${r.battery}%"></span></span>${r.battery}%</span>`;
    cells[3].textContent = r.task === null ? "—" : `#${r.task}${r.leg === "TO_DROP" ? "▸drop" : "▸pick"}${r.queue > 1 ? "+" : ""}`;
    cells[4].textContent = r.wait ? (r.wait.blocker >= 0 ? `AMR ${r.wait.blocker}` : r.wait.kind) : "";
    entry.row.title = r.wait ? `${r.wait.kind} at ${r.wait.resource}` : "";
    entry.row.className = selected === r.id ? "selected" : "";
  }

  function drawWaits(robots) {
    clear(layers.waits);
    const byId = new Map(robots.map((r) => [r.id, r]));
    for (const r of robots) {
      if (!r.wait || r.wait.blocker < 0) continue;
      const b = byId.get(r.wait.blocker);
      if (!b) continue;
      el("line", { class: "wait-link", x1: r.x, y1: r.y, x2: b.x, y2: b.y }, layers.waits);
    }
  }

  function drawRoute(robots) {
    clear(layers.routes);
    if (selected === null) return;
    const r = robots.find((q) => q.id === selected);
    if (!r || !r.route.length) return;
    const points = [`${r.x},${r.y}`];
    for (const id of r.route) {
      const n = nodesById.get(id);
      if (n) points.push(`${n.x},${n.y}`);
    }
    el("polyline", { class: "route", points: points.join(" ") }, layers.routes);
  }

  function select(id) {
    selected = selected === id ? null : id;
  }

  // ---- frame ---------------------------------------------------------------

  function applyFrame(f) {
    $("clock").textContent = fmtClock(f.now_ms);
    const seen = new Set();
    for (const r of f.robots) {
      updateRobot(r);
      seen.add(r.id);
    }
    for (const [id, entry] of robotEls) {
      if (!seen.has(id)) {
        entry.g.remove(); entry.row.remove(); robotEls.delete(id);
      }
    }
    drawWaits(f.robots);
    drawRoute(f.robots);
    syncRunControl(f);

    const c = f.counters;
    $("c-done").textContent = c.tasks_completed;
    $("c-total").textContent = c.tasks_total;
    $("c-progress").style.width = c.tasks_total ? `${100 * c.tasks_completed / c.tasks_total}%` : "0";
    const coll = $("c-coll");
    coll.textContent = c.collisions;
    coll.classList.toggle("alert", c.collisions > 0);
    $("c-yields").textContent = c.yields;
    $("c-pending").textContent = c.tasks_pending;
    $("c-frames").textContent = c.frames_sent.toLocaleString();
    $("c-stopped").textContent = fmtSeconds(c.stopped_ms);
    $("c-makespan").textContent = f.finished ? fmtSeconds(c.makespan_ms) : "—";
    $("fleet-count").textContent = `${f.robots.length} robots · ${f.scenario} seed ${f.seed}${f.source === "webots" ? " · from Webots" : ""}`;

    const note = $("run-note");
    if (f.error) {
      note.textContent = `Run stopped: ${f.error}`;
      note.className = "note error";
    } else if (f.finished) {
      note.textContent = c.tasks_completed === c.tasks_total
        ? `Every task completed. ${c.collisions === 0 ? "No collisions." : c.collisions + " collision(s)."}`
        : `Time limit reached with ${c.tasks_total - c.tasks_completed} task(s) unfinished.`;
      note.className = "note";
    } else {
      note.textContent = f.paused ? "Paused." : "";
      note.className = "note";
    }

    running = !f.finished && !f.error;
    paused = f.paused;
    $("pause").disabled = !running || f.source !== "dashboard";
    $("pause").textContent = paused ? "Resume" : "Pause";
  }

  // ---- transport -----------------------------------------------------------

  /* This page has no run state of its own to report -- it is a spectator by
     construction -- so the lamp carries the link instead. A floor drawn from
     frames that stopped arriving must not keep claiming to be live. */
  function setLink(state) {
    const lamp = $("lamp");
    const label = $("lamp-label");
    if (!lamp || !label) return;
    if (state === "offline") { lamp.dataset.state = "error"; label.textContent = "no telemetry"; }
    else if (state === "polling") { lamp.dataset.state = "paused"; label.textContent = "spectator · polling"; }
    else { lamp.dataset.state = "live"; label.textContent = "spectator"; }
  }

  function connect() {
    Telemetry.connect({
      onMap: drawMap,
      onFrame: applyFrame,
      onIdle: () => { $("empty").hidden = false; $("pause").disabled = true; },
      onLink: setLink,
    });
  }

  /* Run Control describes the run you are watching until you say otherwise.
     Both pages used to pin the scenario to a hardcoded default, so the overview
     offered to start bench3 while visual30 was on screen -- pressing Start would
     silently swap the fleet for a different one. Once the operator edits a
     field, their choice stands and nothing here overwrites it. */
  let formTouched = false;

  for (const id of ["scenario", "seed", "robots", "tasks"]) {
    const el = $(id);
    if (el) el.addEventListener("input", () => { formTouched = true; });
  }
  $("scenario").addEventListener("change", () => { formTouched = true; });

  function syncRunControl(f) {
    if (formTouched) return;
    const select = $("scenario");
    if (f.scenario && [...select.options].some((o) => o.value === f.scenario)) {
      select.value = f.scenario;
    }
    if (typeof f.seed === "number") $("seed").value = f.seed;
    const speed = $("speed");
    if (f.speed && [...speed.options].some((o) => Number(o.value) === f.speed)) {
      speed.value = String(f.speed);
    }
  }

  async function loadScenarios() {
    const list = await (await fetch("/api/scenarios")).json();
    const select = $("scenario");
    for (const s of list) {
      const opt = document.createElement("option");
      opt.value = s.name;
      opt.textContent = `${s.name} — ${s.physical_robots ? s.physical_robots + " physical" : s.robots + " AMRs"}`;
      select.appendChild(opt);
    }
    select.value = list.some((s) => s.name === "visual30") ? "visual30" : list[0].name;
  }

  $("run-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const body = {
      scenario: $("scenario").value,
      seed: Number($("seed").value) || 0,
      speed: Number($("speed").value),
    };
    const robots = Number($("robots").value);
    if (robots > 0) body.robots = robots;
    const tasks = Number($("tasks").value);
    if (tasks > 0) body.tasks = tasks;
    const res = await fetch("/api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!res.ok) {
      const detail = (await res.json()).detail;
      $("run-note").textContent = `Could not start: ${detail}`;
      $("run-note").className = "note error";
    }
  });

  $("pause").addEventListener("click", async () => {
    await fetch(paused ? "/api/resume" : "/api/pause", { method: "POST" });
  });

  $("speed").addEventListener("change", async () => {
    if (!running) return;
    await fetch("/api/speed", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ speed: Number($("speed").value) }) });
  });

  window.addEventListener("error", (event) => {
    const note = $("run-note");
    note.textContent = `Dashboard error: ${event.message}`;
    note.className = "note error";
  });

  loadScenarios();
  connect();
})();
