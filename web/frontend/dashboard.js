/* ROBOTON fleet overview. Reads frames; commands no robot. */

(() => {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const $ = (id) => document.getElementById(id);

  const svg = $("mini");
  const lanes = $("mini-lanes");
  const nodes = $("mini-nodes");
  const robotLayer = $("mini-robots");
  const tbody = document.querySelector("#fleet-table tbody");

  let nodesById = new Map();
  let robotDots = new Map();
  let rows = new Map();
  let paused = false;
  let running = false;

  function el(tag, attrs = {}, parent = null) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  }

  const clear = (node) => { while (node.firstChild) node.removeChild(node.firstChild); };

  function fmtClock(ms) {
    const total = Math.floor(ms / 1000);
    return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
  }

  const fmtSeconds = (ms) => `${Math.round(ms / 1000)} s`;

  // ---- the miniature ------------------------------------------------------

  function drawMini(map) {
    nodesById = new Map(map.nodes.map((n) => [n.id, n]));
    clear(lanes); clear(nodes); clear(robotLayer);
    robotDots = new Map();

    const xs = map.nodes.map((n) => n.x);
    const ys = map.nodes.map((n) => n.y);
    const pad = 2600;
    const minX = Math.min(...xs) - pad, maxX = Math.max(...xs) + pad;
    const minY = Math.min(...ys) - pad, maxY = Math.max(...ys) + pad;
    svg.setAttribute("viewBox", `${minX} ${minY} ${maxX - minX} ${maxY - minY}`);

    const laneWidth = map.lane_offset_mm * 2 + 800;
    for (const e of map.edges) {
      const u = nodesById.get(e.u), v = nodesById.get(e.v);
      if (!u || !v) continue;
      el("line", {
        x1: u.x, y1: u.y, x2: v.x, y2: v.y,
        class: `lane${e.single_lane ? " single" : ""}`,
        "stroke-width": e.single_lane ? 1400 : laneWidth,
      }, lanes);
    }

    for (const n of map.nodes) {
      if (n.pickup || n.drop || n.parking || n.charger) {
        const cls = ["station", n.pickup ? "pickup" : "", n.drop ? "drop" : "",
          n.parking ? "parking" : ""].join(" ").trim();
        el("rect", { class: cls, x: n.x - 700, y: n.y - 700, width: 1400, height: 1400 }, nodes);
      } else if (n.junction) {
        el("circle", { class: "node junction", cx: n.x, cy: n.y, r: 200 }, nodes);
      }
    }

    // The title block carries the drawing's provenance. Seed is there because
    // every number this project reports is only meaningful with one.
    $("tb-map").textContent = map.name;
    $("tb-scale").textContent =
      `${Math.round((maxX - minX) / 1000)} × ${Math.round((maxY - minY) / 1000)} m`;
  }

  function drawRobots(robots) {
    const seen = new Set();
    for (const r of robots) {
      let dot = robotDots.get(r.id);
      if (!dot) {
        dot = el("circle", { r: 520, class: "robot-dot" }, robotLayer);
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

  // ---- fleet table --------------------------------------------------------

  function drawFleet(robots) {
    const seen = new Set();
    for (const r of robots) {
      let row = rows.get(r.id);
      if (!row) {
        row = document.createElement("tr");
        row.innerHTML = "<td></td><td></td><td></td><td></td><td></td>";
        tbody.appendChild(row);
        rows.set(r.id, row);
      }
      const low = r.battery <= 20;
      const cells = row.children;
      cells[0].textContent = r.id;
      cells[1].innerHTML = `<span class="pill ${r.state}">${r.state}</span>`;
      cells[2].innerHTML =
        `<span class="bat"><span class="bat-track"><span class="bat-fill${low ? " low" : ""}" style="width:${r.battery}%"></span></span>${r.battery}%</span>`;
      cells[3].textContent = r.task === null
        ? "—"
        : `#${r.task} ${r.leg === "TO_DROP" ? "to drop" : "to pickup"}`;
      cells[4].textContent = r.wait && r.wait.blocker >= 0 ? `AMR ${r.wait.blocker}` : "—";
      seen.add(r.id);
    }
    for (const [id, row] of rows) {
      if (!seen.has(id)) { row.remove(); rows.delete(id); }
    }
  }

  // ---- frames -------------------------------------------------------------

  function setLamp(state, label) {
    $("lamp").dataset.state = state;
    $("lamp-label").textContent = label;
    $("run-status").textContent = label;
  }

  function applyFrame(f) {
    const c = f.counters;
    $("c-done").textContent = c.tasks_completed;
    $("c-total").textContent = c.tasks_total;
    $("c-progress").style.width =
      c.tasks_total ? `${(100 * c.tasks_completed) / c.tasks_total}%` : "0";

    const collisions = $("c-coll");
    collisions.textContent = c.collisions;
    collisions.classList.toggle("is-alarm", c.collisions > 0);
    const failures = $("c-fail");
    failures.textContent = c.coordination_failures;
    failures.classList.toggle("is-alarm", c.coordination_failures > 0);
    $("c-frames").textContent = c.frames_sent.toLocaleString();
    $("c-stopped").textContent = fmtSeconds(c.stopped_ms);

    $("tb-seed").textContent = f.seed;
    $("tb-clock").textContent = fmtClock(f.now_ms);
    $("run-line").textContent =
      `${f.scenario} · seed ${f.seed} · ${f.robots.length} AMRs · ${f.allocator || "auction"} allocation`;
    $("watch-sub").textContent = `— ${f.robots.length} AMRs on ${f.scenario}`;

    drawRobots(f.robots);
    drawFleet(f.robots);

    const note = $("run-note");
    if (f.error) {
      note.textContent = `Run stopped: ${f.error}`;
      note.className = "hero-note alarm";
      setLamp("error", "stopped");
    } else if (f.finished) {
      note.className = "hero-note";
      note.textContent = c.tasks_completed === c.tasks_total
        ? `Every task delivered in ${fmtSeconds(c.makespan_ms)}. ${c.collisions === 0 ? "No collisions." : c.collisions + " collision(s)."}`
        : `Time limit reached with ${c.tasks_total - c.tasks_completed} task(s) undone.`;
      setLamp("done", "finished");
    } else {
      note.className = "hero-note";
      note.textContent = f.paused ? "Paused. The fleet holds position." : "";
      setLamp(f.paused ? "paused" : "live", f.paused ? "paused" : "live");
    }

    running = !f.finished && !f.error;
    paused = !!f.paused;
    $("pause").disabled = !running || f.source !== "dashboard";
    $("pause").textContent = paused ? "Resume" : "Pause";
  }

  function goIdle() {
    setLamp("idle", "no run");
    $("pause").disabled = true;
    running = false;
    $("run-line").textContent = "Nothing running yet";
    $("run-note").textContent = "Start a run to put the fleet to work.";
    $("run-note").className = "hero-note";
  }

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/telemetry`);
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "map") drawMini(msg.data);
      else if (msg.type === "frame") applyFrame(msg.data);
      else if (msg.type === "idle") goIdle();
    };
    ws.onclose = () => setTimeout(connect, 1000);
  }

  // ---- run control --------------------------------------------------------

  async function loadScenarios() {
    const list = await (await fetch("/api/scenarios")).json();
    const select = $("scenario");
    for (const s of list) {
      const option = document.createElement("option");
      option.value = s.name;
      option.textContent = `${s.name} — ${s.physical_robots ? s.physical_robots + " physical" : s.robots + " AMRs"}`;
      select.appendChild(option);
    }
    select.value = list.some((s) => s.name === "bench3") ? "bench3" : list[0].name;
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

    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const detail = (await res.json().catch(() => ({}))).detail;
      const note = $("run-note");
      note.textContent = `Could not start: ${detail || res.status}`;
      note.className = "hero-note alarm";
    }
  });

  $("pause").addEventListener("click", async () => {
    await fetch(paused ? "/api/resume" : "/api/pause", { method: "POST" });
  });

  $("speed").addEventListener("change", async () => {
    if (!running) return;
    await fetch("/api/speed", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ speed: Number($("speed").value) }),
    });
  });

  loadScenarios();
  connect();
})();
