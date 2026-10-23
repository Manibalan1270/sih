# ROBOTON — demo video narration script

Video: `ROBOTON_demo.mp4`, 3:30.2, 1920×1080, 30 fps, silent (record your voice over it).

**Pace:** about 140 words a minute, calm and confident. Each block is written to fit its slot with a breath to spare; if you run long, drop the last clause rather than rushing. `[cue: …]` marks the moment the on-screen caption changes, so start that sentence there.

**How to record:** play the video and read each block when its time code comes up, or record each block separately and drop it at its time code in any editor (Clipchamp, CapCut, DaVinci Resolve).

## 0:00.0 – 0:12.0  ·  Cold open

*On screen:* Title over the live 3D warehouse floor, camera orbiting.

> Thirty robots. One warehouse floor. And no central computer telling any of them what to do. This is ROBOTON.

<sub>19 words · 12 s slot</sub>

## 0:11.4 – 0:31.4  ·  The problem

*On screen:* Problem card; three problems appear one by one (at 0:15, 0:17, 0:19).

> Today, warehouse robots depend on a single fleet server. When it fails, the whole floor stops. In narrow single-lane aisles, robots meet head-on, and collide or deadlock. And every robot you add makes the central planner's job harder, not easier.

<sub>40 words · 20 s slot</sub>

## 0:30.8 – 0:53.8  ·  02 · The idea

*On screen:* Live 3D floor with real counters. Caption changes to "30 robots. Zero supervisors."

> ROBOTON moves every decision onto the robots themselves. Each AMR plans its own route, bids for its own work, and negotiates right of way with its neighbours over a peer-to-peer mesh. **[cue: caption changes]** This is our live floor: thirty AMRs, fifty pick points in the aisles, six drops, thirty charging bays, and no supervisor.

<sub>52 words · 23 s slot</sub>

## 0:53.2 – 1:20.2  ·  03 · Task allocation

*On screen:* Order page: a pickup and drop are chosen and the task is announced; the fleet bids.

> An order enters through the gateway, which only announces it. It never picks a robot. **[cue: "Every robot bids for itself"]** Every AMR prices the job for itself: the extra travel it adds, a low-battery penalty, minus credit for how long the order has waited. The lowest bid wins. **[cue: "Cheap on the radio"]** Only nearby zones bid, so each auction costs about thirty-four radio frames.

<sub>55 words · 27 s slot</sub>

## 1:19.6 – 1:49.6  ·  04 · Route reservation

*On screen:* 2D live floor with an AMR's route and the yield arcs; switches to Split view near the end.

> Before it moves, each AMR books its whole route as time windows on every junction and aisle, checked against the plans its peers broadcast. **[cue: "Order, not timing"]** Then it drives by order, not by clock: it enters only after everyone booked ahead has left. Cyan robots are yielding; the arcs show who waits for whom. **[cue: Split view]** When two plans clash, the earlier commitment stands. No referee.

<sub>62 words · 30 s slot</sub>

## 1:49.0 – 2:22.0  ·  05 · Single-lane aisles

*On screen:* Chase camera riding an AMR down a single-lane aisle behind another AMR carrying a carton.

> Every aisle is a single lane, driven in either direction. A robot owns an aisle from junction to junction, so two robots can never meet head-on. Here, ours waits its turn behind the robot ahead. **[cue: "No bays cut into the racks"]** Orders are picked right in the aisle; the pick light marks the shelf, and the robot holds that stretch while it loads. **[cue: "Deliver, then collect"]** And each AMR carries one carton, delivering it before it collects the next.

<sub>70 words · 33 s slot</sub>

## 2:21.4 – 2:51.4  ·  06 · Results

*On screen:* Results card: stop-and-wait vs ROBOTON, then per-seed chips and the sweep totals.

> We tested against a classic stop-and-wait baseline: same floor, same hundred and twenty orders, same seed. The baseline logged two hundred and six collisions and jammed after forty-seven orders. ROBOTON delivered all one hundred and twenty, with zero collisions, and the same held on every seed we compared. Across twelve randomized thirty-robot runs and forty benchmark runs: zero collisions.

<sub>59 words · 30 s slot</sub>

## 2:50.8 – 3:13.8  ·  07 · Under the hood

*On screen:* Six feature tiles appear one by one.

> Under the hood, it is built to survive failure. Low battery raises a robot's bid and sends it to charge. If a robot fails, its orders return to the mesh. Plans travel as compact radio frames, the dashboard only watches, and six hundred and fifty-six passing tests pin every safety rule.

<sub>51 words · 23 s slot</sub>

## 3:13.2 – 3:30.2  ·  Close

*On screen:* Back over the floor; title and tagline; fade to black.

> No central planner. No dispatcher. No single point of failure. Just a fleet that coordinates itself, safely, at scale. ROBOTON. The fleet decides.

<sub>23 words · 17 s slot</sub>
