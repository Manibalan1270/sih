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
  const floor = document.querySelector(".floor");
  const insp = {
    card: $("inspector"),
    close: $("insp-close"),
    id: $("insp-id"),
    state: $("insp-state"),
    eta: $("insp-eta"),
    goal: $("insp-goal"),
    battery: $("insp-battery"),
    batteryBar: $("insp-battery-bar"),
    task: $("insp-task"),
    distance: $("insp-distance"),
    waitRow: $("insp-wait-row"),
    wait: $("insp-wait"),
    hops: $("insp-hops"),
    path: $("insp-path"),
  };

  let map = null;               // geometry as sent by /api/map
  let nodesById = new Map();
  let edgesById = new Map();
  let robotEls = new Map();     // robot id -> { g, body, heading, battery, label, row }
  let selected = null;          // robot id whose route is drawn
  let lastHeading = new Map();
  let lastRobots = [];          // the frame on screen, so a click can redraw
  let lastPathKey = "";         // rebuild the inspector's hop list only on change

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

  // A one-way aisle is drawn with chevrons along it, spaced so a long aisle gets
  // several and a short spur-length one still gets its single arrow. Which way an
  // aisle runs is the map's most consequential property now, so it is on the
  // drawing rather than in a legend.
  function drawChevrons(u, v, parent) {
    const dx = v.x - u.x, dy = v.y - u.y;
    const len = Math.hypot(dx, dy);
    if (len < 1) return;
    const ux = dx / len, uy = dy / len;
    const nx = -uy, ny = ux;
    const spacing = 3000, size = 420;
    const count = Math.max(1, Math.round(len / spacing));
    for (let i = 1; i <= count; i += 1) {
      const t = (len * i) / (count + 1);
      const cx = u.x + ux * t, cy = u.y + uy * t;
      const tipX = cx + ux * size, tipY = cy + uy * size;
      const backX = cx - ux * size * 0.4, backY = cy - uy * size * 0.4;
      el("polyline", {
        class: "lane-arrow",
        points: [
          `${backX + nx * size},${backY + ny * size}`,
          `${tipX},${tipY}`,
          `${backX - nx * size},${backY - ny * size}`,
        ].join(" "),
      }, parent);
    }
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

  function fmtEta(ms) {
    const s = Math.max(0, Math.round(ms / 1000));
    if (s < 60) return `${s} s`;
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }

  function fmtMetres(mm) {
    return mm >= 10000 ? `${(mm / 1000).toFixed(0)} m` : `${(mm / 1000).toFixed(1)} m`;
  }

  function nodeLabel(id) {
    const n = nodesById.get(id);
    return n && n.name ? n.name : `#${id}`;
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
    insp.card.hidden = true;
    lastRobots = [];
    lastPathKey = "";

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

    // Lanes: a band of concrete. Every aisle on these maps is a single lane, and
    // the drawing says which kind -- hazard hatching along a two-way corridor,
    // where robots meet one at a time, and a chevron along a one-way aisle showing
    // which way it runs. A two-lane aisle, if a map ever has one again, keeps its
    // dashed centre line.
    for (const e of data.edges) {
      const u = nodesById.get(e.u), v = nodesById.get(e.v);
      const attrs = { x1: u.x, y1: u.y, x2: v.x, y2: v.y };
      const single = e.single_file ?? e.single_lane;
      const cls = ["lane", single ? "single" : "", e.choke ? "choke" : ""].join(" ").trim();
      el("line", { ...attrs, class: cls, "stroke-width": single ? singleWidth : laneWidth }, layers.lanes);
      if (e.single_lane) {
        el("line", { ...attrs, class: "lane-hazard" }, layers.lanes);
      } else if (e.one_way) {
        drawChevrons(u, v, layers.lanes);
      } else if (!single) {
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

  const BODY_R = 470;
  const RING_R = 660;
  const RING_LEN = 2 * Math.PI * RING_R;

  function makeRobot(r) {
    const g = el("g", { class: "robot" }, layers.robots);
    const track = el("circle", { class: "battery-track", r: RING_R }, g);
    const battery = el("circle", {
      class: "battery", r: RING_R, "stroke-dasharray": `${RING_LEN} ${RING_LEN}`,
      transform: "rotate(-90)",
    }, g);
    const body = el("circle", { class: "body", r: BODY_R }, g);
    const cargo = el("rect", { class: "cargo", x: -180, y: -180, width: 360, height: 360 }, g);
    const heading = el("line", { class: "heading", x1: 0, y1: 0, x2: BODY_R + 260, y2: 0 }, g);
    const label = el("text", { class: "label" }, g);
    label.textContent = r.id;
    g.addEventListener("click", (event) => {
      event.stopPropagation();   // the floor itself clears the selection
      select(r.id);
    });

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
    // traffic is drawn apart; on a single-file aisle -- a corridor, a one-way
    // aisle or a station spur -- there is only the centre line.
    if (r.edge === null || r.next === null) return [0, 0];
    const e = edgesById.get(r.edge);
    if (!e || (e.single_file ?? e.single_lane)) return [0, 0];
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

  function routeRemainingMm(r) {
    // Walk the polyline the route layer draws: from where the robot is now,
    // through every node still ahead of it.
    let total = 0, px = r.x, py = r.y;
    for (const id of r.route.slice(1)) {
      const n = nodesById.get(id);
      if (!n) break;
      total += Math.hypot(n.x - px, n.y - py);
      px = n.x; py = n.y;
    }
    return total;
  }

  function placeInspector(x, y) {
    // Floor coordinates are millimetres in the SVG's viewBox; the card is an
    // HTML overlay, so the CTM is what puts the two in the same place.
    const ctm = svg.getScreenCTM();
    if (!ctm) return;
    const pt = svg.createSVGPoint();
    pt.x = x; pt.y = y;
    const at = pt.matrixTransform(ctm);
    const rect = floor.getBoundingClientRect();
    const card = insp.card;
    const w = card.offsetWidth, h = card.offsetHeight;
    const gap = 28, edge = 12;
    // Right of the robot by default; flip to its left near the side panel.
    let left = at.x - rect.left + gap;
    if (left + w > rect.width - edge) left = at.x - rect.left - w - gap;
    let top = at.y - rect.top - h / 2;
    left = Math.max(edge, Math.min(left, rect.width - w - edge));
    top = Math.max(edge, Math.min(top, rect.height - h - edge));
    card.style.transform = `translate(${Math.round(left)}px, ${Math.round(top)}px)`;
  }

  function renderInspector(robots) {
    const r = selected === null ? null : robots.find((q) => q.id === selected);
    if (!r) {
      insp.card.hidden = true;
      lastPathKey = "";
      return;
    }
    insp.card.hidden = false;
    insp.id.textContent = r.id;
    insp.state.textContent = r.state;
    insp.state.className = `pill ${r.state}`;

    const goal = r.route.length ? r.route[r.route.length - 1] : null;
    insp.eta.textContent = r.eta_ms == null ? "—" : fmtEta(r.eta_ms);
    insp.goal.textContent = goal === null ? "standing by" : `to ${nodeLabel(goal)}`;

    const low = r.battery <= 20;
    insp.battery.textContent = `${r.battery}%`;
    insp.battery.className = `inspector-figure${low ? " low" : ""}`;
    insp.batteryBar.style.width = `${r.battery}%`;
    insp.batteryBar.className = `bat-fill${low ? " low" : ""}`;

    insp.task.textContent = r.task === null
      ? "none"
      : `#${r.task} ${r.leg === "TO_DROP" ? "carrying" : "to pickup"}` +
        (r.queue > 1 ? ` +${r.queue - 1} queued` : "");
    insp.distance.textContent = r.route.length > 1 ? fmtMetres(routeRemainingMm(r)) : "—";

    insp.waitRow.hidden = !r.wait;
    if (r.wait) {
      insp.wait.textContent = r.wait.blocker >= 0
        ? `AMR ${r.wait.blocker} · ${r.wait.kind}`
        : `${r.wait.kind} at ${r.wait.resource}`;   // resource is already a label
      insp.wait.className = "wait";
    }

    const hops = Math.max(0, r.route.length - 1);
    insp.hops.textContent = hops ? `${hops} hop${hops === 1 ? "" : "s"}` : "at rest";
    const key = `${r.id}:${r.route.join(",")}`;
    if (key !== lastPathKey) {
      // Rebuilding every frame would fight the reader scrolling a long route.
      lastPathKey = key;
      clear(insp.path);
      const path = r.route.length ? r.route : [r.node];
      path.forEach((id, i) => {
        const n = nodesById.get(id);
        const li = document.createElement("li");
        li.textContent = nodeLabel(id);
        if (i === 0) li.className = "here";
        else if (i === path.length - 1) {
          li.className = ["goal", n && n.pickup ? "pickup-goal" : "",
            n && n.drop ? "drop-goal" : ""].join(" ").trim();
        }
        insp.path.appendChild(li);
      });
    }

    const [ox, oy] = laneOffset(r);   // anchor on the drawn body, not the centre line
    placeInspector(r.x + ox, r.y + oy);
  }

  function select(id) {
    selected = selected === id ? null : id;
    lastPathKey = "";
    // Redraw at once rather than waiting for a frame: a finished run sends none.
    for (const r of lastRobots) updateRobot(r);
    drawRoute(lastRobots);
    renderInspector(lastRobots);
  }

  svg.addEventListener("click", () => { if (selected !== null) select(selected); });
  insp.close.addEventListener("click", () => { if (selected !== null) select(selected); });
  // The card is placed in pixels from an SVG transform, so a resize moves it.
  window.addEventListener("resize", () => {
    if (selected !== null) renderInspector(lastRobots);
  });

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
    lastRobots = f.robots;
    drawWaits(f.robots);
    drawRoute(f.robots);
    renderInspector(f.robots);

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

  }

  // ---- transport -----------------------------------------------------------

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/telemetry`);
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "map") drawMap(msg.data);
      else if (msg.type === "frame") applyFrame(msg.data);
      else if (msg.type === "idle") { $("empty").hidden = false; insp.card.hidden = true; }
    };
    ws.onclose = () => setTimeout(connect, 1000);
  }

  window.addEventListener("error", (event) => {
    const note = $("run-note");
    note.textContent = `Dashboard error: ${event.message}`;
    note.className = "note error";
  });

  connect();
})();
