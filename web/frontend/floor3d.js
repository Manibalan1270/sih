/* ROBOTON live floor in three dimensions.
 *
 * The same map and the same frames the plan view draws, seen from inside the
 * warehouse. app.js hands this module geometry and poses; it sends nothing
 * anywhere. It is a second spectator, not a second source of truth (FR-8.4).
 *
 * Map millimetres become scene metres. Plan x stays x and plan y becomes z, so
 * the overview camera -- south of the floor, looking north -- shows the plan
 * the way the 2D view draws it, tilted back. The camera can also ride an AMR:
 * Chase sits behind it, POV looks out from its front.
 *
 * The building is dressed the way a warehouse looks -- concrete, painted aisle
 * lines, steel shelving stocked with cartons. The fleet's colours still come
 * from the stylesheet's tokens, so a robot reads the same here as on the plan,
 * and the day and night prints reskin both.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const MM = 0.001;

// Proportions shared with scripts/generate_world.py, so this view and the
// Webots world show the same warehouse. Where the racks stand comes with the
// map (simulator/racks.py).
const RACK_HEIGHT = 1.6;
const SINGLE_LANE_WIDTH = 1.4;
const STATION_PAD = 1.4;
const PICK_PAD = 0.9;       // a pickup's pad, on the aisle beside its rack

// The shelving inside a rack block: bays of about a metre, cartons two by two
// on each of three decks, the way pallet racking is stocked.
const SLOT = 1.07;
const DECK_TOPS = [0.12, 0.64, 1.16];
const STOCKED = 0.86;       // share of carton places that hold one

// The building's own colours, the same in both prints: these are things, not
// signals, so they are coloured like the things they are.
const STEEL = "#2f63a8";
const UPRIGHT = "#22467a";
const DECKING = "#39424f";
const SAFETY_YELLOW = "#d6a935";
const TAPE_BLACK = "#16191d";
const RUBBER = "#14181d";
const LAMP = "#eaf6ff";

// Larger than the true 250 mm radius for the reason the plan view draws 470:
// at warehouse scale a true-size AMR is a speck.
const ROBOT_R = 0.42;
const CHASSIS_H = 0.3;
const CLEARANCE = 0.02;
const DECK = CLEARANCE + CHASSIS_H;

// Floor layers stack a few millimetres apart, and each is pulled forward in
// depth by its layer number (polygon offset), so they never fight at the far
// end of a 140 m hall. The clip planes follow the camera for the same reason.
const Y_ZONE = 0.006;
const Y_LANE = 0.014;
const Y_MARK = 0.024;
const Y_STRIPE = 0.028;
const Y_PAD = 0.034;
const Y_PAD_EDGE = 0.04;
const Y_RING = 0.048;
const Y_ROUTE = 0.056;

const FOV = 45;
const SHADOW_MAP = 2048;
const SUN = new THREE.Vector3(-0.3, 0.95, 0.5).normalize();   // from the south-west, high
const RIDE_SHADOW_R = 26;   // m of floor the shadow map covers round a ridden AMR
const RACK_CHUNK = 18;      // m: racks are drawn in chunks so off-screen ones are skipped
const CONCRETE_TILE = 6;    // m per concrete texture tile: slab joints land on the aisle grid
const DASH_M = 0.5;         // route dash pitch, the plan's 300-on-200-off
const DASH_SPEED = 1.4;     // m/s the dashes march toward the goal
const LABEL_PX = 17;        // AMR tags hold this height on screen at any zoom
const SNAP_M = 6;           // a jump this far between frames is a new pose, not motion

// Riding an AMR: Chase sits behind and above it, POV looks out from its front.
const CHASE_DIST = 4.2;
const CHASE_MIN = 1.8;
const CHASE_MAX = 30;
const CHASE_LIFT = 0.42;    // camera height gained per metre of distance
const CHASE_AHEAD = 1.6;    // aim this far in front of the robot

// The plan view's state colours (style.css, .robot.* .body).
const STATE_TOKEN = {
  IDLE: "graphite", BIDDING: "planning", PLANNING: "planning", MOVING: "beacon",
  YIELD: "hold", REPLAN: "hold", AT_DROP: "claim", CHARGING: "verify", FAULT: "alarm",
};

const TOKENS = [
  "ink-900", "ink-850", "ink-800", "ink-700", "ink-600", "ink-500", "rule", "rule-soft",
  "graphite", "chalk", "accent-ink", "beacon", "hold", "verify", "alarm", "claim",
  "pickup", "drop",
];

function readPalette() {
  const css = getComputedStyle(document.documentElement);
  const palette = { planning: "#c9903c" };   // PLANNING's fill is a literal in the stylesheet
  for (const token of TOKENS) palette[token] = css.getPropertyValue(`--${token}`).trim() || "#7d90a2";
  return palette;
}

function mix(a, b, t) {
  return new THREE.Color(a).lerp(new THREE.Color(b), t);
}

function shortestTurn(from, to) {
  const full = Math.PI * 2;
  return ((((to - from + Math.PI) % full) + full) % full) - Math.PI;
}

// A small seeded generator, so a map is always stocked the same way.
function seeded(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hash(text) {
  let h = 2166136261;
  for (let i = 0; i < text.length; i += 1) h = Math.imul(h ^ text.charCodeAt(i), 16777619);
  return h >>> 0;
}

// ---- flat floor geometry -----------------------------------------------------

// Floor markings gathered into one mesh per material: a 100-robot hall has
// hundreds of aisle segments, and a draw call each would be most of the frame.
// A batch built with a tile size maps its texture in world metres, so concrete
// runs continuously from the slab across every aisle laid over it.
class FlatBatch {
  constructor(tile = 0) {
    this.tile = tile;
    this.pos = [];
    this.uv = [];
    this.index = [];
  }

  get empty() {
    return this.index.length === 0;
  }

  quad(corners, y, uvs = [0, 0, 1, 0, 1, 1, 0, 1]) {
    const [a, b, c] = corners;
    // Wind every quad to face up, whichever way round its corners were listed.
    const up = (b[1] - a[1]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[1] - a[1]) > 0;
    const base = this.pos.length / 3;
    for (const [x, z] of corners) this.pos.push(x, y, z);
    if (this.tile) for (const [x, z] of corners) this.uv.push(x / this.tile, z / this.tile);
    else this.uv.push(...uvs);
    if (up) this.index.push(base, base + 1, base + 2, base, base + 2, base + 3);
    else this.index.push(base, base + 2, base + 1, base, base + 3, base + 2);
  }

  // A band of width w along p -> q. u counts pitches along it from u0 metres,
  // so a repeating texture keeps its spacing whatever the band's length.
  strip(p, q, w, y, pitch = 0, u0 = 0) {
    const dx = q[0] - p[0], dz = q[1] - p[1];
    const len = Math.hypot(dx, dz);
    if (len < 1e-6) return 0;
    const nx = (-dz / len) * (w / 2), nz = (dx / len) * (w / 2);
    const ua = pitch ? u0 / pitch : 0, ub = pitch ? (u0 + len) / pitch : 1;
    this.quad(
      [[p[0] + nx, p[1] + nz], [q[0] + nx, q[1] + nz], [q[0] - nx, q[1] - nz], [p[0] - nx, p[1] - nz]],
      y, [ua, 1, ub, 1, ub, 0, ua, 0],
    );
    return len;
  }

  disc(x, z, r, y, segments = 28) {
    const base = this.pos.length / 3;
    this.pos.push(x, y, z);
    this.uv.push(...(this.tile ? [x / this.tile, z / this.tile] : [0.5, 0.5]));
    for (let i = 0; i <= segments; i += 1) {
      const a = (i / segments) * Math.PI * 2;
      const px = x + Math.cos(a) * r, pz = z + Math.sin(a) * r;
      this.pos.push(px, y, pz);
      this.uv.push(...(this.tile ? [px / this.tile, pz / this.tile]
        : [0.5 + Math.cos(a) / 2, 0.5 + Math.sin(a) / 2]));
    }
    for (let i = 1; i <= segments; i += 1) this.index.push(base, base + i + 1, base + i);
  }

  square(x, z, half, y) {
    this.quad([[x - half, z - half], [x + half, z - half], [x + half, z + half], [x - half, z + half]], y);
  }

  outline(x, z, half, w, y) {
    const a = [x - half, z - half], b = [x + half, z - half];
    const c = [x + half, z + half], d = [x - half, z + half];
    const o = w / 2;   // run each side past the corner so the corners close
    this.strip([a[0] - o, a[1]], [b[0] + o, b[1]], w, y);
    this.strip([d[0] - o, d[1]], [c[0] + o, c[1]], w, y);
    this.strip(a, d, w, y);
    this.strip(b, c, w, y);
  }

  build() {
    const g = new THREE.BufferGeometry();
    const count = this.pos.length / 3;
    const normals = new Float32Array(count * 3);
    for (let i = 0; i < count; i += 1) normals[i * 3 + 1] = 1;
    g.setAttribute("position", new THREE.Float32BufferAttribute(this.pos, 3));
    g.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
    g.setAttribute("uv", new THREE.Float32BufferAttribute(this.uv, 2));
    g.setIndex(this.index);
    return g;
  }
}

// ---- textures ----------------------------------------------------------------

function canvasTexture(canvas, repeat) {
  const t = new THREE.CanvasTexture(canvas);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  if (repeat) t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}

function surface(size, paint) {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  paint(c.getContext("2d"), size);
  return c;
}

// Power-floated concrete, near white so the material's tint sets its shade:
// soft cloud, grit, and a slab joint along the tile's edge.
function concreteTexture() {
  const rnd = seeded(7);
  return canvasTexture(surface(512, (g, n) => {
    g.fillStyle = "#f0f0f0";
    g.fillRect(0, 0, n, n);
    for (let i = 0; i < 70; i += 1) {
      const x = rnd() * n, y = rnd() * n, r = 30 + rnd() * 120;
      const shade = rnd() < 0.55 ? "0,0,0" : "255,255,255";
      const cloud = g.createRadialGradient(x, y, 0, x, y, r);
      cloud.addColorStop(0, `rgba(${shade},0.06)`);
      cloud.addColorStop(1, `rgba(${shade},0)`);
      g.fillStyle = cloud;
      g.fillRect(x - r, y - r, r * 2, r * 2);
    }
    for (let i = 0; i < 9000; i += 1) {
      const v = 185 + Math.floor(rnd() * 70);
      g.fillStyle = `rgba(${v},${v},${v},0.55)`;
      g.fillRect(rnd() * n, rnd() * n, 1 + rnd() * 1.6, 1 + rnd() * 1.6);
    }
    g.strokeStyle = "rgba(0,0,0,0.22)";
    g.lineWidth = 3;
    g.strokeRect(0, 0, n, n);
  }), true);
}

// Corrugated board with a strip of packing tape down the middle of each face.
function cardboardTexture() {
  const rnd = seeded(11);
  return canvasTexture(surface(128, (g, n) => {
    g.fillStyle = "#c99c5e";
    g.fillRect(0, 0, n, n);
    for (let x = 0; x < n; x += 4) {
      g.fillStyle = "rgba(0,0,0,0.035)";
      g.fillRect(x, 0, 2, n);
    }
    for (let i = 0; i < 900; i += 1) {
      g.fillStyle = rnd() < 0.5 ? "rgba(90,58,20,0.10)" : "rgba(255,236,200,0.10)";
      g.fillRect(rnd() * n, rnd() * n, 1 + rnd() * 3, 1);
    }
    g.fillStyle = "rgba(222,190,132,0.95)";
    g.fillRect(n * 0.42, 0, n * 0.16, n);
    g.fillStyle = "rgba(255,255,255,0.16)";
    g.fillRect(n * 0.44, 0, n * 0.03, n);
    g.strokeStyle = "rgba(70,44,16,0.38)";
    g.lineWidth = 5;
    g.strokeRect(0, 0, n, n);
  }), false);
}

// A floor fiducial: the square code an AMR reads to know where it is.
function fiducialTexture() {
  const rnd = seeded(3);
  return canvasTexture(surface(64, (g, n) => {
    g.fillStyle = "#ffffff";
    g.fillRect(0, 0, n, n);
    g.fillStyle = "#000000";
    g.fillRect(6, 6, n - 12, n - 12);
    g.fillStyle = "#ffffff";
    const cell = (n - 20) / 5;
    for (let i = 0; i < 5; i += 1) {
      for (let j = 0; j < 5; j += 1) if (rnd() < 0.5) g.fillRect(10 + i * cell, 10 + j * cell, cell, cell);
    }
  }), false);
}

// Diagonal bars for the tape along a two-way corridor, white so the material
// tints them; alpha-tested rather than blended so overlaps stay clean.
function stripeTexture() {
  return canvasTexture(surface(64, (g) => {
    g.fillStyle = "#fff";
    for (let x = -64; x < 128; x += 32) {
      g.beginPath();
      g.moveTo(x, 64); g.lineTo(x + 16, 64); g.lineTo(x + 80, 0); g.lineTo(x + 64, 0);
      g.closePath();
      g.fill();
    }
  }), true);
}

// A carton on a deck never shows its underside, so its box has no bottom: a
// sixth fewer triangles across twenty thousand cartons. BoxGeometry builds its
// faces +x, -x, +y, -y, +z, -z, so the fourth group is the one to drop.
function openBottomBox() {
  const g = new THREE.BoxGeometry(1, 1, 1);
  const index = Array.from(g.index.array);
  const bottom = g.groups[3];
  index.splice(bottom.start, bottom.count);
  g.setIndex(index);
  g.clearGroups();
  return g;
}

function dashTexture() {
  const c = document.createElement("canvas");
  c.width = 64; c.height = 4;
  const g = c.getContext("2d");
  g.fillStyle = "#fff";
  g.fillRect(0, 0, 38, 4);
  return canvasTexture(c, true);
}

const MONO = '"JetBrains Mono", "Cascadia Mono", Consolas, monospace';

function textCanvas(text, { px, pad = 0, background = null }) {
  const font = `600 ${px}px ${MONO}`;
  const probe = document.createElement("canvas").getContext("2d");
  probe.font = font;
  const c = document.createElement("canvas");
  c.width = Math.ceil(probe.measureText(text).width + pad * 2);
  c.height = Math.ceil(px * 1.3 + pad);
  const g = c.getContext("2d");
  if (background) {
    g.fillStyle = background;
    g.beginPath();
    g.roundRect(0, 0, c.width, c.height, c.height / 2.6);
    g.fill();
  }
  g.font = font;
  g.fillStyle = "#fff";
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText(text, c.width / 2, c.height / 2 + px * 0.04);
  return c;
}

// ---- the view ----------------------------------------------------------------

export function createFloor3D(container, hooks = {}) {
  let palette = readPalette();

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  // Capped: past 1.5 the extra pixels cost a laptop GPU more than they show.
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  const canvas = renderer.domElement;
  canvas.setAttribute("aria-label", "Warehouse floor in 3D");
  container.appendChild(canvas);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(palette["ink-900"]);
  // Only while riding an AMR: the far end of the hall falls away into the dark.
  const fog = new THREE.Fog(scene.background.clone(), 20, 70);

  const camera = new THREE.PerspectiveCamera(FOV, 1, 0.1, 4000);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.09;
  controls.screenSpacePanning = false;   // pan along the floor, not the screen
  controls.zoomToCursor = true;
  controls.minDistance = 2;
  controls.maxPolarAngle = Math.PI * 0.47;  // never below the floor

  // Lit so a floor facing straight up shows its token colour: the hemisphere
  // and the sun together come to about pi, which is unit radiance in three.js.
  const sky = new THREE.HemisphereLight(0xffffff, 0x9aa6b2, 1.9);
  const sun = new THREE.DirectionalLight(0xffffff, 1.45);
  sun.castShadow = true;
  sun.shadow.mapSize.set(SHADOW_MAP, SHADOW_MAP);
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 0.03;
  scene.add(sky, sun, sun.target);

  // ---- materials: made once, recoloured when the print changes ---------------

  const themed = [];      // [material, token or palette -> colour, props]
  let mapThemed = [];     // the same, for materials that die with the map
  function tone(material, key, props = ["color"], list = themed) {
    list.push([material, key, props]);
    paint(material, key, props);
    return material;
  }
  function paint(material, key, props) {
    const value = typeof key === "function" ? key(palette) : palette[key];
    for (const prop of props) material[prop].set(value);
  }
  function applyPalette() {
    palette = readPalette();
    for (const [m, key, props] of themed) paint(m, key, props);
    for (const [m, key, props] of mapThemed) paint(m, key, props);
    scene.background.set(palette["ink-900"]);
    fog.color.copy(scene.background);
    dirty = true;
  }

  const concrete = concreteTexture();
  concrete.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
  const cardboard = cardboardTexture();
  const fiducial = fiducialTexture();
  const stripes = stripeTexture();
  const dashes = dashTexture();
  const routeDashes = dashTexture();

  const floorMat = (opts = {}) => new THREE.MeshStandardMaterial({ roughness: 0.95, metalness: 0, ...opts });
  const flatMat = (opts = {}) => new THREE.MeshBasicMaterial(opts);
  const glass = (opacity) => ({ transparent: true, opacity, depthWrite: false });
  // Pull a floor layer forward in depth by its place in the stack.
  const layer = (n, material) => Object.assign(material, {
    polygonOffset: true, polygonOffsetFactor: -n, polygonOffsetUnits: -2 * n,
  });
  // Concrete shades, all from the print: the hall floor, lighter worn aisles.
  const slabShade = (t) => (p) => mix(p["ink-850"], p["ink-500"], t);

  const mats = {
    slab: tone(floorMat(), "ink-850"),
    slabLine: tone(new THREE.LineBasicMaterial(glass(0.6)), "graphite"),
    floor: tone(floorMat({ map: concrete }), slabShade(0.45)),
    zoneA: layer(1, tone(floorMat({ map: concrete }), slabShade(0.4))),
    zoneB: layer(1, tone(floorMat({ map: concrete }), slabShade(0.5))),
    lane: layer(2, tone(floorMat({ map: concrete }), slabShade(0.66))),
    single: layer(2, tone(floorMat({ map: concrete }), slabShade(0.6))),
    choke: layer(2, tone(floorMat({ map: concrete }), slabShade(0.78))),
    edgeLine: layer(3, floorMat({ color: SAFETY_YELLOW, roughness: 0.8 })),
    tapeBase: layer(3, floorMat({ color: TAPE_BLACK })),
    tape: layer(4, floorMat({ color: SAFETY_YELLOW, map: stripes, alphaTest: 0.5, roughness: 0.8 })),
    arrow: layer(3, tone(floorMat({ roughness: 0.8 }), (p) => mix(p["ink-500"], p.chalk, 0.5))),
    centre: layer(3, tone(floorMat({ map: dashes, alphaTest: 0.5 }), (p) => mix(p["ink-500"], p.chalk, 0.4))),
    marker: layer(4, flatMat({ map: fiducial, ...glass(0.5) })),   // black and white in both prints
    pickupFill: layer(5, tone(flatMat(glass(0.3)), "pickup")),
    dropFill: layer(5, tone(flatMat(glass(0.3)), "drop")),
    pickupEdge: layer(6, tone(flatMat(), "pickup")),
    dropEdge: layer(6, tone(flatMat(), "drop")),
    bayEdge: layer(6, tone(flatMat(), (p) => mix(p["ink-500"], p.chalk, 0.25))),
    parkingEdge: layer(6, tone(flatMat(), (p) => mix(p["ink-500"], p.chalk, 0.25))),
    charger: layer(7, tone(flatMat(), "verify")),
    pole: tone(new THREE.MeshStandardMaterial({ roughness: 0.6 }), "graphite"),
    lampPickup: tone(flatMat(), "pickup"),
    lampDrop: tone(flatMat(), "drop"),
    // Shelving and stock are most of the triangles on screen, so they take
    // the cheaper Lambert lighting; matte steel and board lose nothing by it.
    upright: new THREE.MeshLambertMaterial({ color: UPRIGHT }),
    beam: new THREE.MeshLambertMaterial({ color: STEEL }),
    decking: new THREE.MeshLambertMaterial({ color: DECKING }),
    carton: new THREE.MeshLambertMaterial({ map: cardboard }),
    // Racks cast their shadow through one plain block each, not thousands of
    // cartons; the block itself draws nothing.
    proxy: new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: false }),
    rubber: new THREE.MeshStandardMaterial({ color: RUBBER, roughness: 0.95 }),
    chassis: tone(new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.15 }),
      (p) => mix(p["accent-ink"], p["ink-600"], 0.45)),
    sensor: new THREE.MeshStandardMaterial({ color: "#1d232b", roughness: 0.4, metalness: 0.4 }),
    lamp: flatMat({ color: LAMP }),
    nose: tone(flatMat(), "accent-ink"),
    track: layer(7, tone(flatMat(glass(0.55)), "ink-900")),
    battery: layer(8, tone(flatMat(), "verify")),
    batteryLow: layer(8, tone(flatMat(), "alarm")),
    halo: layer(8, tone(flatMat(), "chalk")),
    route: layer(9, tone(flatMat({ map: routeDashes, alphaTest: 0.5 }), "beacon")),
    goal: tone(flatMat({ ...glass(0.45), side: THREE.DoubleSide }), "beacon"),
    goalRing: layer(9, tone(flatMat(), "beacon")),
    wait: tone(flatMat(glass(0.85)), "hold"),
    consign: tone(flatMat(glass(0.5)), "pickup"),
    pickLight: tone(flatMat(), "pickup"),
  };

  const stateMats = new Map();
  function matsFor(state) {
    const key = STATE_TOKEN[state] || "graphite";
    let m = stateMats.get(key);
    if (!m) {
      m = {
        lid: tone(new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.05, emissiveIntensity: 0.22 }),
          key, ["color", "emissive"]),
        strip: tone(flatMat({ side: THREE.DoubleSide }), key),
      };
      stateMats.set(key, m);
    }
    return m;
  }

  // ---- shared robot geometry ---------------------------------------------------

  // The heading arrow on the lid, kept short of the lidar at the front.
  const nose = new THREE.Shape();
  nose.moveTo(ROBOT_R * 0.42, 0);
  nose.lineTo(ROBOT_R * -0.14, ROBOT_R * 0.3);
  nose.lineTo(ROBOT_R * 0.0, 0);
  nose.lineTo(ROBOT_R * -0.14, -ROBOT_R * 0.3);
  nose.closePath();

  const flat = (geometry) => geometry.rotateX(-Math.PI / 2);
  const geo = {
    bumper: new THREE.CylinderGeometry(ROBOT_R * 1.03, ROBOT_R * 1.03, 0.07, 40),
    chassis: new THREE.CylinderGeometry(ROBOT_R, ROBOT_R * 0.97, CHASSIS_H, 40),
    strip: new THREE.CylinderGeometry(ROBOT_R * 1.008, ROBOT_R * 1.008, 0.045, 40, 1, true),
    lid: new THREE.CylinderGeometry(ROBOT_R * 0.9, ROBOT_R * 0.92, 0.05, 40),
    nose: flat(new THREE.ShapeGeometry(nose)),
    lidar: new THREE.CylinderGeometry(0.075, 0.085, 0.08, 24),
    lens: new THREE.CylinderGeometry(0.088, 0.088, 0.022, 24, 1, true),
    lamp: new THREE.BoxGeometry(0.035, 0.04, 0.24),
    wheel: new THREE.CylinderGeometry(0.1, 0.1, 0.06, 18).rotateX(Math.PI / 2),
    cargo: new THREE.BoxGeometry(0.52, 0.36, 0.52),
    track: flat(new THREE.RingGeometry(ROBOT_R + 0.1, ROBOT_R + 0.18, 56)),
    halo: flat(new THREE.RingGeometry(ROBOT_R + 0.25, ROBOT_R + 0.33, 56)),
    beam: new THREE.CylinderGeometry(0.06, 0.06, 2.6, 12, 1, true),
    shell: new THREE.BoxGeometry(1, 1, 1),
    pickLight: new THREE.SphereGeometry(0.14, 16, 12),
    pickStem: new THREE.CylinderGeometry(0.03, 0.03, 0.44, 8),
    goalRing: flat(new THREE.RingGeometry(0.42, 0.52, 40)),
  };

  // ---- scene graph ---------------------------------------------------------------

  let staticGroup = new THREE.Group();
  let staticOwned = [];
  const robotsGroup = new THREE.Group();
  const overlayGroup = new THREE.Group();
  scene.add(staticGroup, robotsGroup, overlayGroup);

  const goal = new THREE.Group();
  const beam = new THREE.Mesh(geo.beam, mats.goal);
  beam.position.y = 1.3;
  const goalRing = new THREE.Mesh(geo.goalRing, mats.goalRing);
  goalRing.position.y = Y_ROUTE;
  goal.add(beam, goalRing);
  goal.visible = false;
  overlayGroup.add(goal);

  let bounds = null;
  let mapKey = "";          // a new run on the same floor keeps the reader's camera
  let pendingFit = null;    // a fit asked for while the canvas had no size
  let fitMode = "oblique";
  let userMoved = false;    // the reader has orbited, panned or zoomed since the last fit
  const robots = new Map(); // id -> entry
  let selected = null;
  let routeMesh = null;
  let waitMeshes = [];
  let follow = null;        // the AMR the camera rides, and how
  let frameGap = 100;       // ms between frames, measured
  let lastFrameAt = 0;
  let active = false;
  let raf = 0;
  let dirty = true;
  let lastTick = 0;

  // ---- static floor ------------------------------------------------------------

  function setMap(data) {
    endFollow();
    scene.remove(staticGroup);
    for (const thing of staticOwned) thing.dispose();
    staticOwned = [];
    mapThemed = [];
    clearRobots();
    resetConsignments();   // the cartons they point at are about to be rebuilt

    const group = new THREE.Group();
    const own = (thing) => { staticOwned.push(thing); return thing; };
    const add = (batch, material, y = 0) => {
      if (batch.empty) return;
      const mesh = new THREE.Mesh(own(batch.build()), material);
      mesh.position.y = y;
      mesh.receiveShadow = true;
      group.add(mesh);
    };
    const instanced = (geometry, material, points, y, shadow = false) => {
      if (!points.length) return null;
      const mesh = new THREE.InstancedMesh(own(geometry), material, points.length);
      const m = new THREE.Matrix4();
      points.forEach(([x, z], i) => mesh.setMatrixAt(i, m.makeTranslation(x, y, z)));
      mesh.receiveShadow = true;
      mesh.castShadow = shadow;
      group.add(mesh);
      return mesh;
    };

    const nodes = new Map(data.nodes.map((n) => [n.id, n]));
    const at = (n) => [n.x * MM, n.y * MM];
    const xs = data.nodes.map((n) => n.x * MM), zs = data.nodes.map((n) => n.y * MM);
    const pad = 3;
    const x0 = Math.min(...xs) - pad, x1 = Math.max(...xs) + pad;
    const z0 = Math.min(...zs) - pad, z1 = Math.max(...zs) + pad;
    bounds = { x0, x1, z0, z1, cx: (x0 + x1) / 2, cz: (z0 + z1) / 2, w: x1 - x0, d: z1 - z0 };

    // The slab: a plate with a drafted edge, concrete on top.
    const slabGeo = own(new THREE.BoxGeometry(bounds.w, 0.2, bounds.d));
    const slab = new THREE.Mesh(slabGeo, mats.slab);
    slab.position.set(bounds.cx, -0.105, bounds.cz);
    const slabEdge = new THREE.LineSegments(own(new THREE.EdgesGeometry(slabGeo)), mats.slabLine);
    slabEdge.position.copy(slab.position);
    group.add(slab, slabEdge);
    const hall = new FlatBatch(CONCRETE_TILE);
    hall.quad([[x0, z0], [x1, z0], [x1, z1], [x0, z1]], 0);
    add(hall, mats.floor);

    // Zones: a shade per zone's bounding box, alternating like the plan.
    if (data.zones.length) {
      const boxes = new Map();
      for (const n of data.nodes) {
        if (n.zone < 0) continue;
        const b = boxes.get(n.zone) || { x0: Infinity, z0: Infinity, x1: -Infinity, z1: -Infinity };
        b.x0 = Math.min(b.x0, n.x * MM); b.z0 = Math.min(b.z0, n.y * MM);
        b.x1 = Math.max(b.x1, n.x * MM); b.z1 = Math.max(b.z1, n.y * MM);
        boxes.set(n.zone, b);
      }
      const tints = [new FlatBatch(CONCRETE_TILE), new FlatBatch(CONCRETE_TILE)];
      [...boxes.entries()].sort((a, c) => a[0] - c[0]).forEach(([, b], i) => {
        tints[i % 2].quad([[b.x0 - 1.5, b.z0 - 1.5], [b.x1 + 1.5, b.z0 - 1.5],
          [b.x1 + 1.5, b.z1 + 1.5], [b.x0 - 1.5, b.z1 + 1.5]], Y_ZONE + (i % 2) * 0.002);
      });
      add(tints[0], mats.zoneA);
      add(tints[1], mats.zoneB);
    }

    // Aisles, marked the way the plan marks them and a warehouse paints them:
    // yellow lines along each aisle, hazard tape along a two-way corridor,
    // arrows down a one-way aisle, a dashed centre on a two-lane one.
    const laneW = (data.lane_offset_mm * 2 + 800) * MM;
    const lanes = {
      lane: new FlatBatch(CONCRETE_TILE), single: new FlatBatch(CONCRETE_TILE), choke: new FlatBatch(CONCRETE_TILE),
    };
    const lines = new FlatBatch(), tapeBase = new FlatBatch(), tape = new FlatBatch();
    const arrows = new FlatBatch(), centre = new FlatBatch();
    const joints = new Map();   // node id -> the widest non-choke band meeting there
    const widthOf = (e) => ((e.single_file ?? e.single_lane) ? SINGLE_LANE_WIDTH : laneW);

    // Every aisle end at every node: which way it leaves and how wide it is.
    const ends = new Map();     // node id -> [{ x, z, w }]
    const endsOf = new Map();   // edge id -> [its end at u, its end at v]
    for (const e of data.edges) {
      const u = nodes.get(e.u), v = nodes.get(e.v);
      if (!u || !v) continue;
      const dx = (v.x - u.x) * MM, dz = (v.y - u.y) * MM, len = Math.hypot(dx, dz) || 1;
      const pair = [{ x: dx / len, z: dz / len, w: widthOf(e) }, { x: -dx / len, z: -dz / len, w: widthOf(e) }];
      [e.u, e.v].forEach((id, i) => {
        if (!ends.has(id)) ends.set(id, []);
        ends.get(id).push(pair[i]);
      });
      endsOf.set(e.id, pair);
    }

    // How far from a node a line painted `off` metres to the left of an
    // aisle's centre (looking out of the node) must stop to clear the other
    // aisles there: until it leaves each one's band, further the shallower
    // they meet. An aisle carrying straight on lets it run through, as does an
    // aisle that leaves on the far side -- a T-junction breaks one line, not
    // both. At a dead end or a station it stops at the pad.
    const clearance = (id, self, off, len) => {
      const others = (ends.get(id) || []).filter((o) => o !== self);
      if (!others.length) return Math.min(self.w / 2, len / 3);
      let t = 0;
      for (const o of others) {
        const cos = self.x * o.x + self.z * o.z;
        const cross = self.x * o.z - self.z * o.x;
        if (Math.abs(cross) < 0.2) continue;
        const exit = (o.w / 2 + Math.sign(cross) * off * cos) / Math.abs(cross);
        if (exit <= 0 || exit * cos + off * cross < -0.1) continue;
        t = Math.max(t, exit);
      }
      return Math.min(t, len * 0.45);
    };

    for (const e of data.edges) {
      const u = nodes.get(e.u), v = nodes.get(e.v);
      if (!u || !v) continue;
      const p = at(u), q = at(v);
      const single = e.single_file ?? e.single_lane;
      const w = widthOf(e);
      const kind = e.choke ? "choke" : single ? "single" : "lane";
      lanes[kind].strip(p, q, w, 0);
      const diagonal = Math.abs(q[0] - p[0]) > 0.01 && Math.abs(q[1] - p[1]) > 0.01;
      for (const id of [e.u, e.v]) {
        const best = joints.get(id);
        const round = diagonal || Boolean(best && best.round);
        if (kind !== "choke" && (!best || best.w < w)) joints.set(id, { w, kind, round });
        else if (!best) joints.set(id, { w, kind, round });
        else best.round = round;
      }

      const dx = q[0] - p[0], dz = q[1] - p[1];
      const len = Math.hypot(dx, dz);
      if (len < 0.01) continue;
      const ux = dx / len, uz = dz / len, nx = -uz, nz = ux;
      // Painted lines run unbroken along an aisle and stop short of every
      // aisle that crosses it, as they do on a real floor.
      const [endU, endV] = endsOf.get(e.id);
      const along = (off, band, batch, pitch = 0) => {
        // Seen from v the aisle runs the other way, so the same line is on its right.
        const ta = clearance(e.u, endU, off, len), tb = clearance(e.v, endV, -off, len);
        const a = [p[0] + ux * ta + nx * off, p[1] + uz * ta + nz * off];
        const b = [q[0] - ux * tb + nx * off, q[1] - uz * tb + nz * off];
        batch.strip(a, b, band, 0, pitch);
      };
      if (e.single_lane) {
        for (const side of [1, -1]) {
          along(side * (w / 2 - 0.1), 0.14, tapeBase);
          along(side * (w / 2 - 0.1), 0.14, tape, 0.14);
        }
      } else {
        for (const side of [1, -1]) along(side * (w / 2 - 0.09), 0.06, lines);
        if (e.one_way) {
          const size = 0.45, arm = 0.11;
          const count = Math.max(1, Math.round(len / 3));
          for (let i = 1; i <= count; i += 1) {
            const t = (len * i) / (count + 1);
            const cx = p[0] + ux * t, cz = p[1] + uz * t;
            const tip = [cx + ux * size, cz + uz * size];
            const bx = cx - ux * size * 0.4, bz = cz - uz * size * 0.4;
            arrows.strip([bx + nx * size, bz + nz * size], tip, arm, 0);
            arrows.strip(tip, [bx - nx * size, bz - nz * size], arm, 0);
          }
        } else if (!single) {
          centre.strip(p, q, 0.05, 0, DASH_M * 1.8);
        }
      }
    }

    // Fill where aisles meet, or every turn shows a notch: a square on a
    // right-angled grid, a round join wherever a diagonal aisle comes in.
    for (const [id, j] of joints) {
      const n = nodes.get(id);
      if (!n) continue;
      if (j.round) lanes[j.kind].disc(n.x * MM, n.y * MM, j.w / 2, 0);
      else lanes[j.kind].square(n.x * MM, n.y * MM, j.w / 2, 0);
    }
    add(lanes.lane, mats.lane, Y_LANE);
    add(lanes.single, mats.single, Y_LANE + 0.001);
    add(lanes.choke, mats.choke, Y_LANE + 0.002);
    add(lines, mats.edgeLine, Y_MARK);
    add(tapeBase, mats.tapeBase, Y_MARK);
    add(tape, mats.tape, Y_STRIPE);
    add(arrows, mats.arrow, Y_MARK);
    add(centre, mats.centre, Y_MARK);

    // Stations, and the fiducials the fleet reads off the floor.
    const fiducials = new FlatBatch(), chargers = [];
    const fills = { pickup: new FlatBatch(), drop: new FlatBatch() };
    const edges = { pickup: new FlatBatch(), drop: new FlatBatch(), bay: new FlatBatch(), parking: new FlatBatch() };
    const posts = { pickup: [], drop: [] };
    const labels = [];
    for (const n of data.nodes) {
      const [x, z] = at(n);
      if (n.pick_face) {
        // A pickup in the aisle: a pad where the AMR stops beside the rack. No
        // post and no painted name: the rack is the landmark, and there are dozens.
        fills.pickup.square(x, z, PICK_PAD / 2, 0);
        edges.pickup.outline(x, z, PICK_PAD / 2, 0.06, 0);
        continue;
      }
      if (n.pickup || n.drop || n.parking || n.charger) {
        const kind = n.pickup ? "pickup" : n.drop ? "drop" : n.parking ? "parking" : "bay";
        if (fills[kind]) fills[kind].square(x, z, STATION_PAD / 2, 0);
        edges[kind].outline(x, z, STATION_PAD / 2, 0.08, 0);
        if (n.charger) chargers.push([x, z]);
        if (n.pickup || n.drop) {
          // An andon post on the pad's far corner from the aisle that feeds it.
          let ax = 0, az = 0;
          for (const o of ends.get(n.id) || []) { ax += o.x; az += o.z; }
          const sx = ax > 0.1 ? -1 : 1, sz = az > 0.1 ? -1 : az < -0.1 ? 1 : -1;
          posts[kind].push([x + sx * STATION_PAD * 0.4, z + sz * STATION_PAD * 0.4]);
          labels.push([n.name, x, z + 1.5]);
        }
      } else if (n.marker) {
        fiducials.square(x, z, 0.17, 0);
      }
    }
    add(fiducials, mats.marker, Y_MARK + 0.002);
    add(fills.pickup, mats.pickupFill, Y_PAD);
    add(fills.drop, mats.dropFill, Y_PAD);
    add(edges.pickup, mats.pickupEdge, Y_PAD_EDGE);
    add(edges.drop, mats.dropEdge, Y_PAD_EDGE);
    add(edges.bay, mats.bayEdge, Y_PAD_EDGE);
    add(edges.parking, mats.parkingEdge, Y_PAD_EDGE);
    instanced(flat(new THREE.CircleGeometry(0.26, 24)), mats.charger, chargers, Y_PAD_EDGE + 0.002);
    for (const kind of ["pickup", "drop"]) {
      instanced(new THREE.CylinderGeometry(0.035, 0.045, 1.9, 8).translate(0, 0.95, 0), mats.pole, posts[kind], 0, true);
      instanced(new THREE.SphereGeometry(0.12, 16, 12), kind === "pickup" ? mats.lampPickup : mats.lampDrop,
        posts[kind], 1.98);
    }

    // Station names painted on the floor where the plan prints them.
    for (const [name, x, z] of labels) {
      const texture = own(canvasTexture(textCanvas(name, { px: 96 })));
      const material = own(layer(6, new THREE.MeshBasicMaterial({ map: texture, ...glass(0.9) })));
      tone(material, "chalk", ["color"], mapThemed);
      const h = 0.72, w = (h * texture.image.width) / texture.image.height;
      const plate = new THREE.Mesh(own(flat(new THREE.PlaneGeometry(w, h))), material);
      plate.position.set(x, Y_PAD_EDGE, z);
      group.add(plate);
    }

    buildRacks(data, group, own, seeded(hash(data.name || "floor")));

    scene.add(group);
    staticGroup = group;
    aimSun();
    const key = `${data.name}|${data.nodes.length}|${x0},${x1},${z0},${z1}`;
    if (key !== mapKey) {
      mapKey = key;
      fit("oblique");
    }
    dirty = true;
  }

  // The racks the map payload carries -- simulator/racks.py's rule, which the
  // Webots world is built from too -- each fitted out as steel shelving stocked
  // with cartons. Nothing is cut into them: a pickup is in the aisle, and the
  // cartons on the bay of shelving facing it -- its pick face -- are what its
  // consignments are drawn from.
  function buildRacks(data, group, own, rnd) {
    const list = data.racks || [];
    if (!list.length) return;

    // Gathered by chunk of floor, so a camera riding an AMR down one aisle
    // skips the shelving behind it instead of drawing the whole hall.
    const chunks = new Map();
    const blocks = [];
    const nearFace = new Map();   // pick-face node -> [{ key, index }] of cartons round it
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), up = new THREE.Vector3(0, 1, 0);
    const box = (list, x, y, z, sx, sy, sz, yaw = 0) => {
      q.setFromAxisAngle(up, yaw);
      list.push(m.compose(new THREE.Vector3(x, y, z), q, new THREE.Vector3(sx, sy, sz)).clone());
    };
    for (const r of list) {
      const c = {
        x: ((r.x0 + r.x1) / 2) * MM, z: ((r.y0 + r.y1) / 2) * MM, sx: (r.x1 - r.x0) * MM, sz: (r.y1 - r.y0) * MM,
      };
      const faceList = (r.faces || []).map((a) => ({
        node: a.node, x0: a.x0 * MM, z0: a.y0 * MM, x1: a.x1 * MM, z1: a.y1 * MM,
      }));
      const key = `${Math.floor(c.x / RACK_CHUNK)}:${Math.floor(c.z / RACK_CHUNK)}`;
      if (!chunks.has(key)) chunks.set(key, { uprights: [], beams: [], decks: [], cartons: [], tints: [] });
      const { uprights, beams, decks, cartons, tints } = chunks.get(key);
      const nx = Math.max(1, Math.round(c.sx / SLOT)), nz = Math.max(1, Math.round(c.sz / SLOT));
      const slotX = c.sx / nx, slotZ = c.sz / nz;
      const left = c.x - c.sx / 2, back = c.z - c.sz / 2, right = left + c.sx, front = back + c.sz;
      box(blocks, c.x, RACK_HEIGHT / 2, c.z, c.sx, RACK_HEIGHT, c.sz);
      for (let i = 0; i <= nx; i += 1) {
        for (let j = 0; j <= nz; j += 1) {
          box(uprights, left + i * slotX, RACK_HEIGHT / 2, back + j * slotZ, 0.06, RACK_HEIGHT, 0.06);
        }
      }
      // Beams round the outside under every deck, and a frame across the top
      // between every bay, which is what reads as shelving from above.
      DECK_TOPS.forEach((top) => {
        const y = top - 0.04;
        box(beams, (left + right) / 2, y, back, right - left, 0.06, 0.05);
        box(beams, (left + right) / 2, y, front, right - left, 0.06, 0.05);
        box(beams, left, y, (back + front) / 2, 0.05, 0.06, front - back);
        box(beams, right, y, (back + front) / 2, 0.05, 0.06, front - back);
        box(decks, c.x, top - 0.0125, c.z, c.sx - 0.02, 0.025, c.sz - 0.02);
      });
      for (let j = 0; j <= nz; j += 1) box(beams, c.x, RACK_HEIGHT - 0.03, back + j * slotZ, c.sx, 0.06, 0.05);
      for (let i = 0; i <= nx; i += 1) box(beams, left + i * slotX, RACK_HEIGHT - 0.03, c.z, 0.05, 0.06, c.sz);
      // Two by two in every bay on every deck, not every place filled.
      for (let i = 0; i < nx; i += 1) {
        for (let j = 0; j < nz; j += 1) {
          const bx = left + (i + 0.5) * slotX, bz = back + (j + 0.5) * slotZ;
          const reach = 1.25 * Math.max(slotX, slotZ);
          const faces = faceList
            .map((a) => ({ a, dist: Math.hypot(bx - (a.x0 + a.x1) / 2, bz - (a.z0 + a.z1) / 2) }))
            .filter(({ dist }) => dist <= reach);
          DECK_TOPS.forEach((top, level) => {
            for (const [ox, oz] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
              if (rnd() > STOCKED) continue;
              const w = Math.min(slotX, slotZ) / 2 - 0.06 - rnd() * 0.05;
              const d = w * (0.9 + rnd() * 0.2), h = 0.3 + rnd() * 0.12;
              box(cartons, bx + (ox * slotX) / 4 + (rnd() - 0.5) * 0.03, top + h / 2,
                bz + (oz * slotZ) / 4 + (rnd() - 0.5) * 0.03, w, h, d, (rnd() - 0.5) * 0.1);
              const shade = 0.82 + rnd() * 0.24;
              tints.push(new THREE.Color().setRGB(shade * (0.98 + rnd() * 0.04), shade * (0.95 + rnd() * 0.06),
                shade * (0.9 + rnd() * 0.12)));
              for (const { a, dist } of faces) {
                if (!nearFace.has(a.node)) nearFace.set(a.node, []);
                nearFace.get(a.node).push({ key, index: cartons.length - 1, level, dist });
              }
            }
          });
        }
      }
    }

    const cube = own(new THREE.BoxGeometry(1, 1, 1));
    const carton = own(openBottomBox());
    const fill = (list, geometry, material, setup) => {
      if (!list.length) return null;
      const mesh = new THREE.InstancedMesh(geometry, material, list.length);
      list.forEach((matrix, i) => mesh.setMatrixAt(i, matrix));
      if (setup) setup(mesh);
      group.add(mesh);
      return mesh;
    };
    for (const k of chunks.values()) {
      fill(k.uprights, cube, mats.upright);
      fill(k.beams, cube, mats.beam);
      fill(k.decks, cube, mats.decking);
      k.mesh = fill(k.cartons, carton, mats.carton, (mesh) => k.tints.forEach((c, i) => mesh.setColorAt(i, c)));
    }
    fill(blocks, cube, mats.proxy, (mesh) => { mesh.castShadow = true; });

    // Consignments are drawn from the cartons that show from the aisle: nearest
    // the pick face first, the lower decks before the top -- within reach of a
    // robot stopped beside it.
    for (const [node, refs] of nearFace) {
      refs.sort((p, q) => (p.dist - q.dist) || (p.level - q.level));
      consign.candidates.set(node, refs.map(({ key, index }) => {
        const k = chunks.get(key);
        return { mesh: k.mesh, index, matrix: k.cartons[index], tint: k.tints[index] };
      }));
    }

    // A pick light over each pick face, on the rack's top frame -- what pick-to-light
    // shelving does -- lit while an order waits there. It is what says "collect
    // here" from across the hall, where a single carton would be too small to see.
    for (const r of list) {
      const left = r.x0 * MM, back = r.y0 * MM, right = r.x1 * MM, front = r.y1 * MM;
      for (const a of r.faces || []) {
        const ax0 = a.x0 * MM, az0 = a.y0 * MM, ax1 = a.x1 * MM, az1 = a.y1 * MM;
        const cx = (ax0 + ax1) / 2, cz = (az0 + az1) / 2, inset = 0.12;
        const [x, z] = ax0 <= left + 0.01 ? [left + inset, cz] : ax1 >= right - 0.01 ? [right - inset, cz]
          : az0 <= back + 0.01 ? [cx, back + inset] : [cx, front - inset];
        const lamp = new THREE.Group();
        const stem = new THREE.Mesh(geo.pickStem, mats.pickLight);
        stem.position.y = 0.22;
        const bulb = new THREE.Mesh(geo.pickLight, mats.pickLight);
        bulb.position.y = 0.48;
        lamp.add(stem, bulb);
        lamp.position.set(x, RACK_HEIGHT, z);
        lamp.visible = false;
        group.add(lamp);
        consign.lamps.set(a.node, lamp);
      }
    }
  }

  // ---- consignments --------------------------------------------------------------
  //
  // An order waiting at a pick face is one carton on the shelving round it: lit
  // in the pickup colour until its AMR arrives, then gone from the shelf -- it is
  // the carton the robot is now carrying. When a face's shelving runs out, a
  // carton is restocked rather than an order going without one.
  const consign = {
    candidates: new Map(),   // pick-face node -> cartons that can be its consignments
    assigned: new Map(),     // task id -> its carton
    taken: new Set(),        // cartons that have left their shelf
    glows: new Map(),        // carton -> the shell lighting it
    spare: [],               // shells not in use
    lamps: new Map(),        // pick-face node -> its pick light
  };
  const hidden = new THREE.Matrix4().makeScale(0, 0, 0);
  const shell = new THREE.Matrix4().makeScale(1.12, 1.08, 1.12);

  function resetConsignments() {
    for (const glow of consign.glows.values()) {
      glow.visible = false;
      consign.spare.push(glow);
    }
    consign.candidates.clear();
    consign.assigned.clear();
    consign.taken.clear();
    consign.glows.clear();
    consign.lamps.clear();   // they belong to the static group being rebuilt
  }

  function light(carton, on) {
    let glow = consign.glows.get(carton);
    if (on && !glow) {
      glow = consign.spare.pop() || new THREE.Mesh(geo.shell, mats.consign);
      glow.matrixAutoUpdate = false;
      glow.matrix.copy(carton.matrix).multiply(shell);
      glow.matrixWorldNeedsUpdate = true;
      glow.visible = true;
      if (!glow.parent) overlayGroup.add(glow);
      consign.glows.set(carton, glow);
      carton.mesh.setColorAt(carton.index, carton.tint.clone().lerp(new THREE.Color(palette.pickup), 0.55));
    } else if (!on && glow) {
      glow.visible = false;
      consign.glows.delete(carton);
      consign.spare.push(glow);
      carton.mesh.setColorAt(carton.index, carton.tint);
    }
    carton.mesh.instanceColor.needsUpdate = true;
  }

  function shelve(carton, onShelf) {
    carton.mesh.setMatrixAt(carton.index, onShelf ? carton.matrix : hidden);
    carton.mesh.instanceMatrix.needsUpdate = true;
    if (onShelf) consign.taken.delete(carton);
    else consign.taken.add(carton);
  }

  // ``list`` is every announced order: [{ task, node, picked }].
  function setConsignments(list) {
    const waiting = new Set(list.filter((c) => !c.picked).map((c) => c.node));
    for (const [node, lamp] of consign.lamps) lamp.visible = waiting.has(node);
    for (const { task, node, picked } of list) {
      const carton = consign.assigned.get(task);
      if (picked) {
        if (carton) {
          light(carton, false);
          shelve(carton, false);
          consign.assigned.delete(task);
        }
        continue;
      }
      if (carton) continue;
      const pool = consign.candidates.get(node);
      if (!pool || !pool.length) continue;
      const busy = new Set(consign.assigned.values());
      let pick = pool.find((c) => !busy.has(c) && !consign.taken.has(c));
      if (!pick) {
        pick = pool.find((c) => !busy.has(c));
        if (!pick) continue;
        shelve(pick, true);
      }
      consign.assigned.set(task, pick);
      light(pick, true);
    }
    dirty = true;
  }

  // Point the sun's shadow map at a disc of floor. The whole hall in overview;
  // only the floor round a ridden AMR while riding, so its shadows stay sharp.
  // The centre moves in whole shadow-map texels, measured in the light's own
  // axes, or the racks' shadows would crawl as the ride carries it along.
  const sunRight = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), SUN).normalize();
  const sunUp = new THREE.Vector3().crossVectors(SUN, sunRight);
  const sunAim = new THREE.Vector3();

  function frameShadow(x, z, radius) {
    const texel = (2 * radius) / SHADOW_MAP;
    sunAim.set(x, 0, z);
    const a = Math.round(sunAim.dot(sunRight) / texel) * texel;
    const b = Math.round(sunAim.dot(sunUp) / texel) * texel;
    const c = sunAim.dot(SUN);
    sunAim.copy(sunRight).multiplyScalar(a).addScaledVector(sunUp, b).addScaledVector(SUN, c);
    sun.target.position.copy(sunAim);
    sun.position.copy(sunAim).addScaledVector(SUN, radius * 3);
    const cam = sun.shadow.camera;
    if (cam.right !== radius) {
      cam.left = -radius; cam.right = radius; cam.top = radius; cam.bottom = -radius;
      cam.near = 0.5;
      cam.far = radius * 6;
      cam.updateProjectionMatrix();
    }
  }

  function aimSun() {
    if (bounds) frameShadow(bounds.cx, bounds.cz, Math.hypot(bounds.w, bounds.d) / 2 + 2);
  }

  // Clip planes that suit where the camera is, for depth precision: tight
  // round a ridden AMR (the fog ends the view anyway), scaled to the distance
  // otherwise.
  function updateClip() {
    let near = 0.25, far = 320;
    if (!follow) {
      const d = camera.position.distanceTo(controls.target);
      const span = bounds ? Math.max(bounds.w, bounds.d) : 100;
      near = Math.min(20, Math.max(0.1, d * 0.03));
      far = d * 4 + span * 3;
    }
    if (near !== camera.near || far !== camera.far) {
      camera.near = near;
      camera.far = far;
      camera.updateProjectionMatrix();
    }
  }

  // ---- camera ----------------------------------------------------------------

  function viewport() {
    return { w: container.clientWidth, h: container.clientHeight };
  }

  // Frame the whole floor: the closest camera from which every corner of the
  // slab, and the tops of the racks along its edge, lands inside the canvas.
  // Perspective makes the near edge the widest thing on screen, which no
  // closed-form field-of-view sum gets right, so the distance is searched for.
  function fit(mode) {
    fitMode = mode;
    const { w, h } = viewport();
    if (!bounds || !w || !h) { pendingFit = mode; return; }
    pendingFit = null;
    userMoved = false;
    camera.aspect = w / h;
    // North stays up, as on the plan, so Split reads as one floor drawn twice.
    const elevation = mode === "top" ? Math.PI / 2 - 1e-4 : THREE.MathUtils.degToRad(52);
    const toward = new THREE.Vector3(0, Math.sin(elevation), Math.cos(elevation));
    const target = new THREE.Vector3(bounds.cx, 0, bounds.cz);
    const corners = [];
    for (const y of [0, RACK_HEIGHT]) {
      for (const x of [bounds.x0, bounds.x1]) for (const z of [bounds.z0, bounds.z1]) corners.push(new THREE.Vector3(x, y, z));
    }
    const p = new THREE.Vector3();
    camera.near = 0.1;
    camera.far = 1e5;
    camera.updateProjectionMatrix();
    const fits = (dist) => {
      camera.position.copy(target).addScaledVector(toward, dist);
      camera.lookAt(target);
      camera.updateMatrixWorld();
      return corners.every((c) => {
        p.copy(c).project(camera);
        return Math.abs(p.x) <= 0.94 && Math.abs(p.y) <= 0.9 && p.z < 1;
      });
    };
    let near = 1, far = 4 * Math.hypot(bounds.w, bounds.d) + 20;
    for (let i = 0; i < 40; i += 1) {
      const mid = (near + far) / 2;
      if (fits(mid)) far = mid;
      else near = mid;
    }
    fits(far);
    controls.target.copy(target);
    controls.maxDistance = far * 3;
    controls.update();
    updateClip();
    dirty = true;
  }

  // Ride an AMR. Chase hangs back and above and swings round behind it as it
  // turns; POV sits on its front edge. Dragging looks around it, scrolling
  // moves the camera in or out, and both keep following.
  const camGoal = new THREE.Vector3();
  const lookGoal = new THREE.Vector3();

  // The ridden AMR's own tag would hang over the road ahead; the page already
  // says whose ride it is.
  function showTag(id, on) {
    const e = robots.get(id);
    if (e) e.label.visible = on;
  }

  function startFollow(id, mode) {
    const e = robots.get(id);
    if (!e) return false;
    if (follow) showTag(follow.id, true);
    showTag(id, false);
    if (follow) {
      follow.id = id;
      follow.mode = mode;
      follow.offset = 0;
    } else {
      follow = {
        id, mode, offset: 0, dist: CHASE_DIST, lift: CHASE_LIFT,
        yaw: -e.body.rotation.y, look: controls.target.clone(),
      };
    }
    controls.enabled = false;
    scene.fog = fog;
    dirty = true;
    return true;
  }

  function endFollow(notify = false) {
    if (!follow) return;
    showTag(follow.id, true);
    const e = robots.get(follow.id);
    if (e) controls.target.set(e.root.position.x, 0, e.root.position.z);
    else controls.target.copy(follow.look);
    follow = null;
    userMoved = true;       // leave the camera where the ride left it
    controls.enabled = true;
    scene.fog = null;
    aimSun();
    controls.update();
    dirty = true;
    if (notify && hooks.onFollowEnd) hooks.onFollowEnd();
  }

  function stepFollow(dt) {
    const e = robots.get(follow.id);
    if (!e) {
      endFollow(true);
      return true;
    }
    const pov = follow.mode === "pov";
    const p = e.root.position;
    const turn = 1 - Math.exp(-dt * (pov ? 10 : 3.2));
    follow.yaw += shortestTurn(follow.yaw, -e.body.rotation.y + follow.offset) * turn;
    const fx = Math.cos(follow.yaw), fz = Math.sin(follow.yaw);
    if (pov) {
      camGoal.set(p.x + fx * 0.36, 0.64, p.z + fz * 0.36);
      lookGoal.set(p.x + fx * 8, 0.45, p.z + fz * 8);
      fog.near = 8;
      fog.far = 45;
    } else {
      camGoal.set(p.x - fx * follow.dist, 0.9 + follow.dist * follow.lift, p.z - fz * follow.dist);
      lookGoal.set(p.x + fx * CHASE_AHEAD, 0.35, p.z + fz * CHASE_AHEAD);
      fog.near = follow.dist + 14;
      fog.far = follow.dist + 60;
    }
    frameShadow(p.x, p.z, RIDE_SHADOW_R);
    const k = 1 - Math.exp(-dt * (pov ? 14 : 5));
    const moved = camera.position.distanceTo(camGoal) + follow.look.distanceTo(lookGoal);
    camera.position.lerp(camGoal, k);
    follow.look.lerp(lookGoal, k);
    camera.lookAt(follow.look);
    return moved > 0.002;
  }

  function resize() {
    const { w, h } = viewport();
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    for (const e of robots.values()) sizeLabel(e.label);
    // Until the reader takes the camera, the framing follows the canvas: Split
    // halves it, a window resize reshapes it. A ride keeps riding.
    if (!follow) {
      if (pendingFit) fit(pendingFit);
      else if (!userMoved && bounds) fit(fitMode);
    }
    dirty = true;
  }

  new ResizeObserver(resize).observe(container);
  controls.addEventListener("change", () => { dirty = true; });
  controls.addEventListener("start", () => { userMoved = true; });
  new MutationObserver(applyPalette).observe(document.documentElement, {
    attributes: true, attributeFilter: ["data-theme"],
  });

  // ---- robots -------------------------------------------------------------------

  // A sprite with size attenuation off is sized in view units at unit depth;
  // this many of them make LABEL_PX pixels on the current canvas.
  function labelUnit() {
    const h = viewport().h || 600;
    return (LABEL_PX * 2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2)) / h;
  }

  function sizeLabel(sprite) {
    const k = labelUnit();
    sprite.scale.set(k * sprite.userData.aspect, k, 1);
  }

  function makeLabel(text) {
    const c = textCanvas(text, { px: 40, pad: 12, background: "rgba(11, 18, 25, 0.86)" });
    const texture = canvasTexture(c);
    texture.generateMipmaps = false;
    texture.minFilter = THREE.LinearFilter;
    const material = new THREE.SpriteMaterial({
      map: texture, sizeAttenuation: false, depthTest: false, depthWrite: false, transparent: true, fog: false,
    });
    const sprite = new THREE.Sprite(material);
    sprite.center.set(0.5, 0);
    sprite.renderOrder = 10;
    sprite.userData.aspect = c.width / c.height;
    sizeLabel(sprite);
    return sprite;
  }

  function makeRobot(id) {
    const root = new THREE.Group();
    const body = new THREE.Group();
    const look = matsFor("IDLE");
    const part = (geometry, material, x, y, z = 0) => {
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(x, y, z);
      body.add(mesh);
      return mesh;
    };
    const bumper = part(geo.bumper, mats.rubber, 0, 0.055);
    const chassis = part(geo.chassis, mats.chassis, 0, CLEARANCE + CHASSIS_H / 2);
    const strip = part(geo.strip, look.strip, 0, CLEARANCE + CHASSIS_H * 0.62);
    const lid = part(geo.lid, look.lid, 0, DECK + 0.025);
    part(geo.nose, mats.nose, 0, DECK + 0.052);
    const lidar = part(geo.lidar, mats.sensor, ROBOT_R * 0.62, DECK + 0.09);
    part(geo.lens, mats.lamp, ROBOT_R * 0.62, DECK + 0.095);
    part(geo.lamp, mats.lamp, ROBOT_R * 0.975, CLEARANCE + CHASSIS_H * 0.74);
    part(geo.wheel, mats.rubber, 0, 0.1, ROBOT_R - 0.02);
    part(geo.wheel, mats.rubber, 0, 0.1, -(ROBOT_R - 0.02));
    const cargo = part(geo.cargo, mats.carton, 0, DECK + 0.05 + 0.18);
    cargo.visible = false;
    for (const mesh of [bumper, chassis, lid, lidar, cargo]) mesh.castShadow = true;

    const track = new THREE.Mesh(geo.track, mats.track);
    track.position.y = Y_RING;
    const battery = new THREE.Mesh(new THREE.BufferGeometry(), mats.battery);
    battery.position.y = Y_RING + 0.003;
    const halo = new THREE.Mesh(geo.halo, mats.halo);
    halo.position.y = Y_RING;
    halo.visible = false;
    const label = makeLabel(String(id));
    label.position.y = DECK + 0.62;

    root.add(body, track, battery, halo, label);
    root.traverse((o) => { o.userData.robotId = id; });
    robotsGroup.add(root);
    return {
      root, body, strip, lid, cargo, battery, halo, label,
      state: null, charge: -1, low: null,
      from: { x: 0, z: 0, h: 0 }, to: { x: 0, z: 0, h: 0 }, t0: 0, dur: 1,
    };
  }

  function dropRobot(id) {
    const e = robots.get(id);
    if (!e) return;
    robotsGroup.remove(e.root);
    e.battery.geometry.dispose();
    e.label.material.map.dispose();
    e.label.material.dispose();
    robots.delete(id);
    if (follow && follow.id === id) endFollow(true);
  }

  function clearRobots() {
    for (const id of [...robots.keys()]) dropRobot(id);
    setOverlay({ selected: null, route: null, waits: [] });
  }

  function setCharge(e, pct, low) {
    const rounded = Math.round(pct);
    if (rounded === e.charge && low === e.low) return;
    e.charge = rounded;
    e.low = low;
    e.battery.geometry.dispose();
    // Starts at north and runs clockwise as seen from above, like the plan's ring.
    const sweep = Math.max(0.001, (rounded / 100) * Math.PI * 2);
    e.battery.geometry = flat(new THREE.RingGeometry(ROBOT_R + 0.1, ROBOT_R + 0.18, 56, 1,
      Math.PI / 2 - sweep, sweep));
    e.battery.material = low ? mats.batteryLow : mats.battery;
  }

  // Each frame is a target; a robot glides there over one measured frame gap,
  // so ten frames a second read as continuous motion.
  function setFrame(list, overlay) {
    const now = performance.now();
    if (lastFrameAt) frameGap = Math.min(400, Math.max(40, frameGap * 0.7 + (now - lastFrameAt) * 0.3));
    lastFrameAt = now;
    const seen = new Set();
    for (const r of list) {
      seen.add(r.id);
      const x = r.x * MM, z = r.y * MM, h = -THREE.MathUtils.degToRad(r.heading);
      let e = robots.get(r.id);
      if (!e) {
        e = makeRobot(r.id);
        robots.set(r.id, e);
        e.from = { x, z, h };
        e.to = { x, z, h };
        e.t0 = 0;
      } else {
        const cx = e.root.position.x, cz = e.root.position.z, ch = e.body.rotation.y;
        e.from = Math.hypot(x - cx, z - cz) > SNAP_M ? { x, z, h } : { x: cx, z: cz, h: ch };
        e.to = { x, z, h: e.from.h + shortestTurn(e.from.h, h) };
        e.t0 = now;
        e.dur = frameGap;
      }
      if (e.state !== r.state) {
        const look = matsFor(r.state);
        e.lid.material = look.lid;
        e.strip.material = look.strip;
        e.state = r.state;
      }
      e.cargo.visible = Boolean(r.carrying);
      setCharge(e, r.battery, r.low);
      advance(e, now);
    }
    for (const id of [...robots.keys()]) if (!seen.has(id)) dropRobot(id);
    setOverlay(overlay);
  }

  function advance(e, now) {
    const t = e.t0 ? Math.min(1, (now - e.t0) / e.dur) : 1;
    e.root.position.set(e.from.x + (e.to.x - e.from.x) * t, 0, e.from.z + (e.to.z - e.from.z) * t);
    e.body.rotation.y = e.from.h + (e.to.h - e.from.h) * t;
    return t < 1;
  }

  // ---- selection, route and right-of-way --------------------------------------

  function setOverlay({ selected: id = null, route = null, waits = [] } = {}) {
    selected = id;
    for (const [rid, e] of robots) e.halo.visible = rid === selected;

    if (routeMesh) {
      overlayGroup.remove(routeMesh);
      routeMesh.geometry.dispose();
      routeMesh = null;
    }
    goal.visible = false;
    if (route && route.length > 1) {
      const band = new FlatBatch();
      let along = 0;
      for (let i = 1; i < route.length; i += 1) {
        const p = [route[i - 1][0] * MM, route[i - 1][1] * MM];
        const q = [route[i][0] * MM, route[i][1] * MM];
        along += band.strip(p, q, 0.16, 0, DASH_M, along);
      }
      if (!band.empty) {
        routeMesh = new THREE.Mesh(band.build(), mats.route);
        routeMesh.position.y = Y_ROUTE;
        overlayGroup.add(routeMesh);
        const end = route[route.length - 1];
        goal.position.set(end[0] * MM, 0, end[1] * MM);
        goal.visible = true;
      }
    }

    // Who is holding for whom: an arc from the waiting AMR to its blocker.
    for (const m of waitMeshes) {
      overlayGroup.remove(m);
      m.geometry.dispose();
    }
    waitMeshes = [];
    for (const [a, b] of waits) {
      const ea = robots.get(a), eb = robots.get(b);
      if (!ea || !eb) continue;
      const p = new THREE.Vector3(ea.to.x, DECK + 0.5, ea.to.z);
      const q = new THREE.Vector3(eb.to.x, DECK + 0.5, eb.to.z);
      const mid = p.clone().add(q).multiplyScalar(0.5);
      mid.y += 0.4 + p.distanceTo(q) * 0.2;
      const tube = new THREE.TubeGeometry(new THREE.QuadraticBezierCurve3(p, mid, q), 24, 0.035, 6, false);
      const mesh = new THREE.Mesh(tube, mats.wait);
      overlayGroup.add(mesh);
      waitMeshes.push(mesh);
    }
    dirty = true;
  }

  // ---- pointer -------------------------------------------------------------------

  const raycaster = new THREE.Raycaster();
  raycaster.camera = camera;
  const ndc = new THREE.Vector2();

  function robotAt(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    ndc.set(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
    raycaster.setFromCamera(ndc, camera);
    const hit = raycaster.intersectObjects(robotsGroup.children, true)[0];
    return hit ? hit.object.userData.robotId ?? null : null;
  }

  // A click picks; a drag orbits, or while riding an AMR looks around it. Only
  // a press that barely moved is a click.
  let press = null;
  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) { press = null; return; }
    press = { x: event.clientX, y: event.clientY, lastX: event.clientX, lastY: event.clientY };
    if (follow) canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (follow && press && event.buttons & 1) {
      follow.offset -= (event.clientX - press.lastX) * 0.006;
      follow.lift = Math.min(1.6, Math.max(0.05, follow.lift + (event.clientY - press.lastY) * 0.004));
      press.lastX = event.clientX;
      press.lastY = event.clientY;
      dirty = true;
      return;
    }
    if (event.buttons) return;
    canvas.style.cursor = robotAt(event.clientX, event.clientY) === null ? "" : "pointer";
  });
  canvas.addEventListener("pointerup", (event) => {
    if (!press) return;
    const moved = Math.hypot(event.clientX - press.x, event.clientY - press.y);
    press = null;
    if (moved <= 5 && hooks.onPick) hooks.onPick(robotAt(event.clientX, event.clientY));
  });
  canvas.addEventListener("wheel", (event) => {
    if (!follow) return;   // otherwise the orbit controls zoom
    event.preventDefault();
    if (follow.mode === "pov") return;
    follow.dist = Math.min(CHASE_MAX, Math.max(CHASE_MIN, follow.dist * Math.exp(event.deltaY * 0.0015)));
    dirty = true;
  }, { passive: false });
  canvas.addEventListener("contextmenu", (event) => event.preventDefault());

  // ---- loop ---------------------------------------------------------------------

  // Renders only when something changed: a paused or finished run costs nothing.
  function tick(now) {
    raf = 0;
    if (!active) return;
    const dt = lastTick ? Math.min(0.1, (now - lastTick) / 1000) : 0;
    lastTick = now;
    let busy = false;
    for (const e of robots.values()) busy = advance(e, now) || busy;
    // The orbit controls would aim the camera at their own target, so they
    // only run while the camera is not riding an AMR.
    if (follow) busy = stepFollow(dt) || busy;
    else controls.update();
    if (routeMesh) {
      routeDashes.offset.x -= (dt * DASH_SPEED) / DASH_M;
      busy = true;
    }
    if (consign.glows.size) {
      // Waiting consignments breathe, so they read as waiting from across the hall.
      mats.consign.opacity = 0.35 + 0.3 * (0.5 + 0.5 * Math.sin(now / 260));
      busy = true;
    }
    if (busy || dirty) {
      dirty = false;
      updateClip();
      renderer.render(scene, camera);
      if (hooks.onRender) hooks.onRender();
    }
    raf = requestAnimationFrame(tick);
  }

  function setActive(on) {
    active = on;
    if (on) {
      resize();
      dirty = true;
      if (!raf) raf = requestAnimationFrame(tick);
    } else {
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
      lastTick = 0;
    }
  }

  // Where an AMR is drawn, in client pixels, so an HTML card can follow it.
  const probe = new THREE.Vector3();
  function screenOf(id) {
    const e = robots.get(id);
    if (!e) return null;
    probe.set(e.root.position.x, DECK, e.root.position.z).project(camera);
    if (probe.z < -1 || probe.z > 1) return null;
    const rect = canvas.getBoundingClientRect();
    return {
      x: rect.left + ((probe.x + 1) / 2) * rect.width,
      y: rect.top + ((1 - probe.y) / 2) * rect.height,
    };
  }

  // "overview" frames the floor, "top" looks straight down, "orbit" frees the
  // camera where it is; "chase" and "pov" ride the AMR named by id. Returns
  // whether the camera now does what was asked.
  function perspective(mode, id = null) {
    if (mode === "chase" || mode === "pov") return startFollow(id, mode);
    endFollow();
    if (mode === "overview") fit("oblique");
    else if (mode === "top") fit("top");
    return true;
  }

  return {
    setMap,
    setFrame,
    setOverlay,
    setConsignments,
    setActive,
    screenOf,
    perspective,
    clear: clearRobots,
  };
}
