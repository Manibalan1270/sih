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
        class: `lane${(e.single_file ?? e.single_lane) ? " single" : ""}`,
        "stroke-width": (e.single_file ?? e.single_lane) ? 1400 : laneWidth,
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
        dot = el("circle", { r: 540, class: "robot-dot" }, robotLayer);
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
      `${f.scenario} · seed ${f.seed} · ${f.robots.length} AMRs · ${f.allocator || "auction"} allocation` +
      (f.source === "webots" ? " · rendered in Webots" : "");
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
    for (const id of ["scenario", "bench-scenario"]) {
      const select = $(id);
      for (const s of list) {
        const option = document.createElement("option");
        option.value = s.name;
        option.textContent = `${s.name} — ${s.physical_robots ? s.physical_robots + " physical" : s.robots + " AMRs"}`;
        select.appendChild(option);
      }
      select.value = list.some((s) => s.name === "bench3") ? "bench3" : list[0].name;
    }
  }

  const CONFIG_HINT = {
    B: "Books its whole route in time windows and enters a junction only after the robots booked ahead of it have left.",
    A: "No bidding, no plans shared: each AMR stops when its own forward sensor sees something and waits.",
  };

  function showConfigHint() {
    $("config-hint").textContent = CONFIG_HINT[$("config").value];
  }
  $("config").addEventListener("change", showConfigHint);
  showConfigHint();

  $("run-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const body = {
      scenario: $("scenario").value,
      seed: Number($("seed").value) || 0,
      speed: Number($("speed").value),
      config: $("config").value,
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

  // ---- benchmark: stop-and-wait against ROBOTON ---------------------------
  //
  // The page reports what was measured and does not editorialise it into a
  // speed-up. Where configuration A collides, its makespan is a time it could
  // only reach by driving through its own fleet, so the headline leads with the
  // collisions and says plainly that the times are not comparable.

  const pct = (value) => (value === null || value === undefined
    ? "—"
    : `${value > 0 ? "+" : ""}${value.toFixed(1)}%`);

  function benchRow(label, a, b, diff, tone) {
    const tr = document.createElement("tr");
    for (const [text, cls] of [[label, "bench-measure"], [a, ""], [b, ""], [diff, tone || ""]]) {
      const td = document.createElement("td");
      td.textContent = text;
      if (cls) td.className = cls;
      tr.appendChild(td);
    }
    return tr;
  }

  function renderBenchmark(result) {
    const [a, b] = result.configurations;
    const body = document.querySelector("#bench-table tbody");
    while (body.firstChild) body.removeChild(body.firstChild);

    body.appendChild(benchRow(
      "Collisions", String(a.collisions), String(b.collisions),
      a.collisions === b.collisions ? "—" : `${b.collisions - a.collisions}`,
      b.collisions < a.collisions ? "bench-good" : b.collisions > a.collisions ? "bench-bad" : "",
    ));
    body.appendChild(benchRow(
      "Coordination failures",
      String(a.coordination_failures), String(b.coordination_failures),
      `${b.coordination_failures - a.coordination_failures}`,
      b.coordination_failures < a.coordination_failures ? "bench-good" : "",
    ));
    body.appendChild(benchRow(
      "Tasks delivered",
      `${a.tasks_done}/${result.tasks_total}`, `${b.tasks_done}/${result.tasks_total}`,
      `${b.tasks_done - a.tasks_done}`,
      b.tasks_done > a.tasks_done ? "bench-good" : b.tasks_done < a.tasks_done ? "bench-bad" : "",
    ));
    body.appendChild(benchRow(
      "Runs that finished",
      `${a.runs_finished}/${result.seeds}`, `${b.runs_finished}/${result.seeds}`,
      `${b.runs_finished - a.runs_finished}`,
      b.runs_finished > a.runs_finished ? "bench-good" : b.runs_finished < a.runs_finished ? "bench-bad" : "",
    ));
    // A configuration that never delivered its task set has no makespan to show:
    // the figure is the last completion it managed, which gets *smaller* the
    // longer it is jammed.
    const span = (cfg) => (cfg.finished_all
      ? `${fmtSeconds(cfg.makespan_mean_ms)} ± ${fmtSeconds(cfg.makespan_std_ms)}`
      : "did not finish");
    body.appendChild(benchRow(
      "Makespan, mean", span(a), span(b),
      result.makespan_valid ? pct(result.delta.makespan_pct) : "not comparable",
      result.makespan_valid && result.delta.makespan_pct < 0 ? "bench-good"
        : result.makespan_valid ? "bench-bad" : "bench-void",
    ));
    // Time held is comparable even when the makespans are not: it is how long the
    // fleet spent stationary, whether or not either configuration got through the
    // work.
    body.appendChild(benchRow(
      "Time held, mean per run",
      fmtSeconds(a.stopped_mean_ms), fmtSeconds(b.stopped_mean_ms),
      pct(result.delta.stopped_pct),
      result.delta.stopped_pct < 0 ? "bench-good" : "bench-bad",
    ));
    body.appendChild(benchRow(
      "Deadlocked runs", String(a.deadlocks), String(b.deadlocks),
      `${b.deadlocks - a.deadlocks}`, b.deadlocks < a.deadlocks ? "bench-good" : "",
    ));

    const headline = $("bench-headline");
    const avoided = result.delta.collisions_avoided;
    headline.innerHTML = "";
    const figure = document.createElement("span");
    figure.className = "bench-figure";
    figure.textContent = avoided > 0 ? `${avoided} collisions avoided` : `${b.collisions} collisions`;
    const sub = document.createElement("span");
    sub.className = "bench-figure-sub";
    sub.textContent = ` — ${result.seeds} seed${result.seeds === 1 ? "" : "s"} of `
      + `${result.scenario}, ${result.robots} AMRs, identical task sets`;
    headline.appendChild(figure);
    headline.appendChild(sub);

    const note = $("bench-note");
    if (!a.finished_all && b.finished_all) {
      // The result the density claim is about: the baseline stops working.
      note.className = "bench-note caveat";
      note.textContent =
        `Stop-and-wait did not get through the work: ${a.runs_finished} of ${result.seeds} `
        + `run(s) finished, ${result.tasks_total - a.tasks_done} task(s) left undone at the `
        + `time limit, and ${a.collisions} collisions along the way. ROBOTON delivered `
        + `${b.tasks_done}/${result.tasks_total} in ${fmtSeconds(b.makespan_mean_ms)} with `
        + `${b.collisions} collisions. There is no makespan to compare against, because one `
        + `of the two never produced one.`;
    } else if (!result.makespan_valid) {
      note.className = "bench-note caveat";
      note.textContent =
        `Stop-and-wait finished in ${fmtSeconds(a.makespan_mean_ms)}, but it did so with `
        + `${a.collisions} collisions across ${result.seeds} run(s): in the simulator robots `
        + `overlap and carry on, so that is a time it could not have achieved without driving `
        + `through its own fleet. The two makespans are therefore not a like-for-like `
        + `comparison, and ROBOTON's ${pct(result.delta.makespan_pct)} is what collision-free `
        + `traffic costs on this floor, not a slowdown against a workable baseline.`;
    } else if (result.meets_ac3) {
      note.className = "bench-note";
      note.textContent =
        `Both configurations ran clean, so the times compare directly: ROBOTON is `
        + `${pct(result.delta.makespan_pct)} on mean makespan and meets AC-3's 20% bar.`;
    } else {
      note.className = "bench-note";
      note.textContent =
        `Both configurations ran clean, so the times compare directly: ROBOTON is `
        + `${pct(result.delta.makespan_pct)} on mean makespan, short of AC-3's 20% bar.`;
    }
    $("bench-result").hidden = false;
  }

  let benchPoll = null;

  function applyBenchStatus(status) {
    const progress = $("bench-progress");
    const button = $("bench-start");
    if (status.state === "running") {
      button.disabled = true;
      progress.className = "bench-progress";
      progress.textContent = `running ${status.done_runs}/${status.total_runs} runs…`;
      return true;
    }
    button.disabled = false;
    if (status.state === "error") {
      progress.className = "bench-progress alarm";
      progress.textContent = status.error;
    } else if (status.state === "done" && status.result) {
      progress.className = "bench-progress";
      progress.textContent = `${status.total_runs} runs complete`;
      renderBenchmark(status.result);
    } else {
      progress.textContent = "";
    }
    return false;
  }

  async function pollBenchmark() {
    const status = await (await fetch("/api/benchmark")).json();
    if (!applyBenchStatus(status) && benchPoll) {
      clearInterval(benchPoll);
      benchPoll = null;
    }
  }

  $("bench-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const body = {
      scenario: $("bench-scenario").value,
      seeds: Number($("bench-seeds").value) || 5,
    };
    const robots = Number($("bench-robots").value);
    if (robots > 0) body.robots = robots;

    $("bench-start").disabled = true;
    $("bench-progress").textContent = "starting…";
    const res = await fetch("/api/benchmark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const detail = (await res.json().catch(() => ({}))).detail;
      $("bench-progress").className = "bench-progress alarm";
      $("bench-progress").textContent = `Could not start: ${detail || res.status}`;
      $("bench-start").disabled = false;
      return;
    }
    applyBenchStatus(await res.json());
    if (!benchPoll) benchPoll = setInterval(pollBenchmark, 1000);
  });

  loadScenarios();
  connect();
  pollBenchmark();
})();
