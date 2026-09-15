/* The one link from a page to the run (FR-8.2).

   Every page is a spectator, and this is the only thing that reads the run.
   It prefers the WebSocket and falls back to polling the REST views when the
   socket cannot be established, because the failure is otherwise invisible:
   uvicorn started without a WebSocket implementation answers /ws/telemetry
   with 404, the socket closes before it ever opens, and a page that only
   retries in silence reports "no run in progress" while the fleet is working.

   Nothing here sends to the fleet -- it reads /ws/telemetry, /api/state,
   /api/map and /api/status, and holds no path to the mesh (FR-8.4, BR-7). */

(function (global) {
  "use strict";

  const POLL_MS = 500;
  const RETRY_MIN_MS = 1000;
  const RETRY_MAX_MS = 10000;

  /* Link states, in the order a page should trust them:
       live    -- the socket is open and frames are arriving
       polling -- the socket is down; frames come from REST instead
       offline -- neither works, so what is on screen is stale */
  function connect(handlers) {
    const onMap = handlers.onMap || (() => {});
    const onFrame = handlers.onFrame || (() => {});
    const onIdle = handlers.onIdle || (() => {});
    const onLink = handlers.onLink || (() => {});

    let socket = null;
    let pollTimer = null;
    let retryMs = RETRY_MIN_MS;
    let link = null;
    /* Which run the drawn map belongs to, as "scenario:seed" -- REST has no
       generation counter, so that pair is what identifies a run. Only a frame
       carries it; a map payload does not. A map drawn before its first frame is
       marked PENDING and named by the frame that follows. Keying it off the map
       payload instead is what made every poll refetch and redraw the floor,
       wiping the robots off it ten times a second. */
    const PENDING = Symbol("pending");
    let mapKey = null;

    function handleMap(data) { mapKey = PENDING; onMap(data); }

    function handleFrame(data) {
      if (mapKey === PENDING) mapKey = keyOf(data);
      onFrame(data);
    }

    function setLink(state) {
      if (state === link) return;
      link = state;
      onLink(state);
    }

    // -- the socket ----------------------------------------------------------

    function open() {
      let ws;
      try {
        const proto = location.protocol === "https:" ? "wss" : "ws";
        ws = new WebSocket(`${proto}://${location.host}/ws/telemetry`);
      } catch (err) {
        startPolling();
        scheduleRetry();
        return;
      }
      socket = ws;
      ws.onopen = () => {
        retryMs = RETRY_MIN_MS;
        stopPolling();
        setLink("live");
      };
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === "map") handleMap(msg.data);
        else if (msg.type === "frame") handleFrame(msg.data);
        else if (msg.type === "idle") onIdle(msg.data);
      };
      ws.onerror = () => {}; // onclose always follows; handle it there
      ws.onclose = () => {
        socket = null;
        startPolling();
        scheduleRetry();
      };
    }

    function scheduleRetry() {
      setTimeout(open, retryMs);
      retryMs = Math.min(retryMs * 2, RETRY_MAX_MS);
    }

    // -- the fallback --------------------------------------------------------

    /* Identifies a run. Frames only -- a map payload carries neither field. */
    function keyOf(frame) {
      return `${frame.scenario}:${frame.seed}`;
    }

    function startPolling() {
      if (pollTimer !== null) return;
      poll();
      pollTimer = setInterval(poll, POLL_MS);
    }

    function stopPolling() {
      if (pollTimer === null) return;
      clearInterval(pollTimer);
      pollTimer = null;
    }

    async function poll() {
      try {
        const res = await fetch("/api/state");
        if (res.status === 404) {
          setLink("polling");
          onIdle(await (await fetch("/api/status")).json());
          return;
        }
        if (!res.ok) throw new Error(`state ${res.status}`);
        const frame = await res.json();
        // The map is large and static for a run; refetch it only when the run
        // behind it changes, which a frame's scenario and seed identify.
        const key = keyOf(frame);
        if (key !== mapKey) {
          const mapRes = await fetch("/api/map");
          if (mapRes.ok) handleMap(await mapRes.json());
        }
        setLink("polling");
        handleFrame(frame);
      } catch (err) {
        setLink("offline");
      }
    }

    open();
    // If the socket does not open promptly, start polling rather than show an
    // empty floor: a 404 handshake can take longer to fail than a frame takes
    // to arrive, and the page should never be blank while a run is live.
    setTimeout(() => { if (link !== "live") startPolling(); }, 1500);

    return {
      get link() { return link; },
      close() {
        stopPolling();
        if (socket) { socket.onclose = null; socket.close(); socket = null; }
      },
    };
  }

  global.Telemetry = { connect };
})(window);
