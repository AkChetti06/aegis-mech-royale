"""Offline approximation of the mechfight/1 practice mode, for A/B testing bot logic.

This is NOT the official server. It implements the rules in protocol.md closely enough to
exercise a bot's movement, aiming, looting, zone handling and fights on the real map, against a
scripted 15-mech squad that hunts the player (one hunter early, more as the player kills them and
as time passes, all of them late), as the practice mode is described on the track page.

    python3 tools/sim.py --bot my_bot.py --matches 24 --jobs 4

The bot module is imported in-process (its `mf.run(...)` must be guarded by __main__), and every
act() call is timed. Scores are mean (16 - place), the official metric.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DT = 0.05
PI = math.pi
TAU = 2 * PI


def wrap(a):
    return (a + PI) % TAU - PI


# --------------------------------------------------------------------------------------- world
class World:
    def __init__(self, arena: str):
        with open(os.path.join(arena, "v2", "world.json"), encoding="utf-8") as fh:
            w = json.load(fh)
        t = w["terrain"]
        self.tminx, self.tminy, self.tnx, self.tny = t["minX"], t["minY"], t["nx"], t["ny"]
        raw = np.fromfile(os.path.join(arena, "v2", "terrain.i16"), dtype="<i2")
        self.H = (raw.reshape(self.tny, self.tnx).astype(np.float32) * 0.01)
        self.Hl = self.H.tolist()
        b = w["bounds"]
        self.minx, self.maxx, self.miny, self.maxy = b["minX"], b["maxX"], b["minY"], b["maxY"]
        cache = os.path.join("/tmp", "mechsim_raster_v1.npz")
        if os.path.exists(cache):
            d = np.load(cache)
            base1, top1, base2, top2 = d["b1"], d["t1"], d["b2"], d["t2"]
        else:
            base1, top1, base2, top2 = self._rasterize(w["prisms"])
            np.savez(cache, b1=base1, t1=top1, b2=base2, t2=top2)
        self.b1, self.t1, self.b2, self.t2 = base1.tolist(), top1.tolist(), base2.tolist(), top2.tolist()
        self.free = []  # open ground cells, for spawning things
        tt = top1
        js, iis = np.nonzero(np.isnan(tt))
        for k in range(0, len(js), 7):
            x, y = self.tminx + iis[k] + 0.5, self.tminy + js[k] + 0.5
            if self.minx + 30 < x < self.maxx - 30 and self.miny + 30 < y < self.maxy - 30:
                self.free.append((x, y))
        self.roofs = []
        js, iis = np.nonzero((~np.isnan(tt)) & (tt < 40) & (tt > 3))
        for k in range(0, len(js), 23):
            self.roofs.append((self.tminx + iis[k] + 0.5, self.tminy + js[k] + 0.5, float(tt[js[k], iis[k]])))

    def _rasterize(self, prisms):
        ny, nx = self.tny, self.tnx
        b1 = np.full((ny, nx), np.nan, np.float32)
        t1 = np.full((ny, nx), np.nan, np.float32)
        b2 = np.full((ny, nx), np.nan, np.float32)
        t2 = np.full((ny, nx), np.nan, np.float32)
        for p in prisms:
            rings = [p["ring"]] + list(p.get("holes", []))
            edges = []
            for r in rings:
                pts = [(r[k], r[k + 1]) for k in range(0, len(r) - 1, 2)]
                for a in range(len(pts)):
                    edges.append((pts[a - 1], pts[a]))
            ys = [q for r in [p["ring"]] for q in r[1::2]]
            xs = p["ring"][0::2]
            j0 = max(0, int(math.floor(min(ys) - self.tminy)))
            j1 = min(ny - 1, int(math.ceil(max(ys) - self.tminy)))
            base, top = p["base"], p["top"]
            for j in range(j0, j1 + 1):
                yc = self.tminy + j + 0.5
                cross = []
                for (xa, ya), (xb, yb) in edges:
                    if (ya > yc) != (yb > yc):
                        cross.append(xa + (yc - ya) * (xb - xa) / (yb - ya))
                cross.sort()
                for k in range(0, len(cross) - 1, 2):
                    i0 = max(0, int(math.ceil(cross[k] - self.tminx - 0.5)))
                    i1 = min(nx - 1, int(math.floor(cross[k + 1] - self.tminx - 0.5)))
                    if i1 < i0:
                        continue
                    seg_t = t1[j, i0:i1 + 1]
                    empty = np.isnan(seg_t)
                    # primary interval: the one reaching lowest (buildings); bridges go to slot 2
                    if p["kind"] == "bridge":
                        t2[j, i0:i1 + 1] = top
                        b2[j, i0:i1 + 1] = base
                    else:
                        cur = np.where(empty, -1e9, seg_t)
                        upd = top > cur
                        b1[j, i0:i1 + 1] = np.where(upd, np.where(empty, base, np.minimum(b1[j, i0:i1 + 1], base)), b1[j, i0:i1 + 1])
                        t1[j, i0:i1 + 1] = np.where(upd, top, seg_t)
        return b1, t1, b2, t2

    def terrain(self, x, y):
        fx = min(max(x - self.tminx, 0.0), self.tnx - 1.001)
        fy = min(max(y - self.tminy, 0.0), self.tny - 1.001)
        i, j = int(fx), int(fy)
        ax, ay = fx - i, fy - j
        H = self.Hl
        r0, r1 = H[j], H[j + 1]
        return (r0[i] * (1 - ax) + r0[i + 1] * ax) * (1 - ay) + (r1[i] * (1 - ax) + r1[i + 1] * ax) * ay

    def _cell(self, x, y):
        i, j = int(x - self.tminx), int(y - self.tminy)
        if 0 <= i < self.tnx and 0 <= j < self.tny:
            return j, i
        return None

    def solid(self, x, y, z):
        if x < self.minx or x > self.maxx or y < self.miny or y > self.maxy:
            return True
        c = self._cell(x, y)
        if c is None:
            return True
        j, i = c
        t = self.t1[j][i]
        if t == t and self.b1[j][i] <= z < t:
            return True
        t = self.t2[j][i]
        if t == t and self.b2[j][i] <= z < t:
            return True
        return False

    def support(self, x, y, z, step):
        """Highest standable surface at (x, y) not above z + step."""
        g = self.terrain(x, y)
        c = self._cell(x, y)
        if c is not None:
            j, i = c
            for t in (self.t1[j][i], self.t2[j][i]):
                if t == t and t <= z + step + 0.01 and t > g:
                    g = t
        return g

    def los(self, a, b, step=2.0):
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        n = int(d / step)
        for k in range(1, n):
            f = k / n
            x, y, z = a[0] + dx * f, a[1] + dy * f, a[2] + dz * f
            if self.solid(x, y, z) or self.terrain(x, y) > z:
                return False
        return True

    def ray(self, a, yaw, pitch, rng, step=1.5):
        """Distance along a ray until it hits geometry (or rng)."""
        cx, cy, sz = math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)
        d = step
        while d < rng:
            x, y, z = a[0] + cx * d, a[1] + cy * d, a[2] + sz * d
            if self.solid(x, y, z) or self.terrain(x, y) > z:
                return d
            d += step
        return rng


# --------------------------------------------------------------------------------------- entities
class Mech:
    def __init__(self, mid, x, y, z, cfg):
        self.id = mid
        self.pos = [x, y, z]
        self.vel = [0.0, 0.0, 0.0]
        self.aim = [random.uniform(-PI, PI), 0.0]
        self.cmd_aim = list(self.aim)
        self.stats = dict(cfg["mech"])
        self.wstats = {k: dict(v) for k, v in cfg["weapons"].items()}
        self.max_hp = self.stats["hp"]
        self.hp = self.max_hp
        self.energy = self.stats["energy"]
        self.energy_t = 0.0
        self.grounded = True
        self.cubes = 0
        self.upgrades: List[str] = []
        self.weapons: List[Optional[Dict[str, Any]]] = [self.new_weapon("gun"), None]
        self.alive = True
        self.last_combat = -99.0
        self.short_until = -1.0
        self.events: List[Dict[str, Any]] = []
        self.offers: List[List[str]] = []
        self.offer_id = 0
        self.offer_t = 0.0
        self.thresh_i = 0
        self.step_t = 0.0
        self.capture = None
        self.kills = 0
        self.died_tick = None
        # action state
        self.move = [0.0, 0.0]
        self.boost = False
        self.jump = False
        self.fire = [False, False]
        self.reload_req = False

    def new_weapon(self, kind):
        w = {"kind": kind, "cooldown": 0.0}
        if kind == "gun":
            w["ammo"] = int(self.wstats["gun"]["mag"])
            w["reload"] = 0.0
        if kind == "laser":
            w["charge"] = 0.0
            w["held"] = False
            w["full_t"] = 0.0
        return w

    def scale(self):
        return 1 + 0.05 * self.cubes

    def holds(self, kind):
        return any(w and w["kind"] == kind for w in self.weapons)


class Proj:
    __slots__ = ("kind", "pos", "vel", "owner", "dist", "rng", "ttl", "dmg")

    def __init__(self, kind, pos, vel, owner, rng, dmg):
        self.kind, self.pos, self.vel, self.owner, self.rng, self.dmg = kind, pos, vel, owner, rng, dmg
        self.dist = 0.0
        self.ttl = 7.0


# --------------------------------------------------------------------------------------- squad AI
DIFF = {  # aim noise (rad), reaction delay (s), lead skill (0..1), loot speed (s until special)
    "easy": (0.060, 0.6, 0.0, 9999),
    "medium": (0.035, 0.4, 0.6, 120),
    "hard": (0.018, 0.25, 0.9, 60),
    "expert": (0.010, 0.15, 1.0, 30),
}


class HouseBrain:
    def __init__(self, mech, diff, rng):
        self.m, self.diff, self.rng = mech, diff, rng
        self.noise, self.react, self.lead, self.loot_t = DIFF[diff]
        self.loot_t *= rng.uniform(0.6, 1.4)
        self.wander = None
        self.strafe = rng.choice((-1, 1))
        self.strafe_t = 0.0
        self.seen_t = -99.0
        self.stuck_t = 0.0
        self.last_xy = (0, 0)


# --------------------------------------------------------------------------------------- match
class Match:
    def __init__(self, world: World, cfg, bot_factory, seed, player_slot=None, log=False):
        self.w, self.cfg, self.rng = world, cfg, random.Random(seed)
        random.seed(seed)
        self.t = 0.0
        self.tick = 0
        self.log = log
        self.mechs: List[Mech] = []
        pts = []
        z0 = cfg["zone"]
        for mid in range(16):
            for _ in range(400):
                x, y = self.rng.choice(world.free)
                if math.hypot(x - z0["start_center"][0], y - z0["start_center"][1]) > z0["start_radius"] * 0.8:
                    continue
                if all(math.hypot(x - a, y - b) > 130 for a, b in pts):
                    break
            pts.append((x, y))
            self.mechs.append(Mech(mid, x, y, world.terrain(x, y), cfg))
        self.pid = player_slot if player_slot is not None else self.rng.randrange(16)
        order = ["easy"] * 4 + ["medium"] * 4 + ["hard"] * 4 + ["expert"] * 3
        self.brains: Dict[int, HouseBrain] = {}
        k = 0
        for m in self.mechs:
            if m.id != self.pid:
                self.brains[m.id] = HouseBrain(m, order[k], random.Random(self.rng.random()))
                k += 1
        self.hunt_order = [m.id for m in self.mechs if m.id != self.pid]  # weakest first
        self.projs: List[Proj] = []
        self.npcs = []
        for n in range(cfg["npc"]["count"]):
            x, y = self.rng.choice(world.free)
            self.npcs.append({"id": 100 + n, "pos": [x, y, world.terrain(x, y)], "home": (x, y), "hp": cfg["npc"]["hp"],
                              "cool": 0.0, "target": None, "vel": [0, 0, 0]})
        self.crates = []
        for c in range(cfg["crates"]["count"]):
            if self.rng.random() < cfg["crates"]["roof_fraction"] and world.roofs:
                x, y, z = self.rng.choice(world.roofs)
            else:
                x, y = self.rng.choice(world.free)
                z = world.terrain(x, y)
            self.crates.append({"id": 1000 + c, "pos": [x, y, z], "hp": cfg["crates"]["hp"]})
        self.cubes = []
        self.pickups = []
        self.next_id = 5000
        self.next_pick = 8000
        kinds = ["mortar", "laser", "lightning"]
        for _ in range(int(cfg["loot"]["floor_count"])):
            if self.rng.random() < cfg["loot"]["floor_roof_fraction"] and world.roofs:
                x, y, z = self.rng.choice(world.roofs)
            else:
                x, y = self.rng.choice(world.free)
                z = world.terrain(x, y)
            self.add_pickup(self.rng.choice(kinds), [x, y, z])
        # zone
        self.zone_phases = cfg["zone"]["phases"]
        self.zone_centers = [tuple(cfg["zone"]["start_center"])]
        self.zone_radii = [cfg["zone"]["start_radius"]]
        c = self.zone_centers[0]
        r = self.zone_radii[0]
        for ph in self.zone_phases:
            nr = ph["radius"]
            for _ in range(100):
                ang = self.rng.uniform(-PI, PI)
                d = self.rng.uniform(0, max(0.0, r - nr))
                nc = (c[0] + d * math.cos(ang), c[1] + d * math.sin(ang))
                if world.minx + 40 < nc[0] < world.maxx - 40 and world.miny + 40 < nc[1] < world.maxy - 40:
                    break
            self.zone_centers.append(nc)
            self.zone_radii.append(nr)
            c, r = nc, nr
        self.drops = []
        self.drop_times = list(cfg.get("drops", {}).get("times", []))
        self.sounds: Dict[int, List[Dict[str, Any]]] = {self.pid: []}
        self.beams: List[Dict[str, Any]] = []
        self.memory: Dict[int, float] = {}
        self.squad_seen_t = -99.0
        self.squad_seen_pos = None
        self.player_kills = 0
        self.alive_order: List[int] = []
        self.hello = {"type": "hello", "protocol": "mechfight/1", "slot": self.pid, "seed": seed & 0xFFFFFFFF,
                      "map": {"dir": ARENA,
                              "files": ["v2/world.json", "v2/terrain.i16"]},
                      "config": cfg}
        self.bot = bot_factory(json.loads(json.dumps(self.hello)))
        self.act_times: List[float] = []
        self.player = self.mechs[self.pid]
        self.crash = None

    # ------------------------------------------------------------------ helpers
    def add_pickup(self, kind, pos):
        self.pickups.append({"id": self.next_pick, "kind": kind, "pos": list(pos)})
        self.next_pick += 1

    def add_cube(self, pos):
        ang = self.rng.uniform(-PI, PI)
        self.cubes.append({"id": self.next_id, "pos": [pos[0] + 2 * math.cos(ang), pos[1] + 2 * math.sin(ang), pos[2]]})
        self.next_id += 1

    def zone_now(self):
        t = self.t
        acc = 0.0
        for k, ph in enumerate(self.zone_phases):
            hs, ss = acc + ph["hold_s"], acc + ph["hold_s"] + ph["shrink_s"]
            if t < ss:
                c0, c1 = self.zone_centers[k], self.zone_centers[k + 1]
                r0, r1 = self.zone_radii[k], self.zone_radii[k + 1]
                f = 0.0 if t < hs else (t - hs) / ph["shrink_s"]
                c = (c0[0] + (c1[0] - c0[0]) * f, c0[1] + (c1[1] - c0[1]) * f)
                return {"center": list(c), "radius": r0 + (r1 - r0) * f, "next_center": list(c1), "next_radius": r1,
                        "shrink_start": hs, "shrink_end": ss, "dps": ph["dps"]}
            acc = ss
        c = self.zone_centers[-1]
        last = self.zone_phases[-1]
        return {"center": list(c), "radius": 0.0, "next_center": list(c), "next_radius": 0.0, "shrink_start": acc,
                "shrink_end": acc, "dps": last["dps"]}

    def in_zone(self, pos, z):
        return math.hypot(pos[0] - z["center"][0], pos[1] - z["center"][1]) <= z["radius"]

    def sound(self, kind, pos):
        p = self.player
        if not p.alive:
            return
        d = math.hypot(pos[0] - p.pos[0], pos[1] - p.pos[1])
        rng = self.cfg["sound"]["range"].get(kind, 100)
        if d > rng or d < 0.5:
            return
        b = math.atan2(pos[1] - p.pos[1], pos[0] - p.pos[0]) + self.rng.gauss(0, math.radians(7) * p.stats.get("hearing", 1))
        self.sounds[self.pid].append({"kind": kind, "bearing": wrap(b), "dist": "near" if d < 60 else "mid" if d < 160 else "far",
                                      "age": 0.0, "_t": self.t})

    def damage(self, target: Mech, amount, weapon, src: Optional[Mech], from_pos, effect=None):
        if not target.alive:
            return
        rear = False
        if src is not None:
            amount *= 1 + 0.04 * min(src.cubes, 10)
            bear = math.atan2(from_pos[1] - target.pos[1], from_pos[0] - target.pos[0])
            if abs(wrap(bear - target.aim[0])) > PI / 2:
                rear = True
                amount *= target.stats["rear_mult"]
            amount *= max(0.0, 1 - target.stats.get("armor", 0))
            src.last_combat = self.t
        else:
            bear = None
        target.hp -= amount
        target.last_combat = self.t
        target.events.append({"type": "damaged", "amount": round(amount, 2), "weapon": weapon,
                              "bearing": None if bear is None else round(bear, 3), "by": None if src is None else src.id,
                              **({"rear": True} if rear else {})})
        if src is not None:
            src.events.append({"type": "hit", "target": target.id, "amount": round(amount, 2), "weapon": weapon,
                               **({"effect": effect} if effect else {}), **({"rear": True} if rear else {})})
        if target.hp <= 0:
            self.kill(target, src)

    def kill(self, m: Mech, src):
        m.alive = False
        m.hp = 0
        m.died_tick = self.tick
        alive = sum(1 for q in self.mechs if q.alive) + 1
        m.place = alive
        for w in m.weapons:
            if w and w["kind"] != "gun":
                self.add_pickup(w["kind"], m.pos)
        for _ in range(m.cubes + 2):
            self.add_cube(m.pos)
        self.sound("death", m.pos)
        if src is not None and src is not m:
            src.kills += 1
            src.events.append({"type": "kill", "target": m.id})
            if src.id == self.pid:
                self.player_kills += 1

    def damage_npc(self, n, amount, src):
        n["hp"] -= amount * (1 + 0.04 * min(src.cubes, 10))
        n["target"] = src.id
        src.last_combat = self.t
        src.events.append({"type": "hit", "target": n["id"], "amount": round(amount, 2), "weapon": "x"})
        if n["hp"] <= 0:
            self.npcs.remove(n)
            for _ in range(int(self.cfg["npc"]["cubes"])):
                self.add_cube(n["pos"])

    def damage_crate(self, c, amount, src):
        c["hp"] -= amount
        if c["hp"] <= 0:
            self.crates.remove(c)
            self.add_cube(c["pos"])
            self.sound("crate", c["pos"])
            if self.rng.random() < self.cfg["loot"]["crate_weapon_chance"]:
                self.add_pickup(self.rng.choice(["mortar", "laser", "lightning"]), c["pos"])

    def give_cube(self, m: Mech):
        m.cubes += 1
        old = m.max_hp
        m.max_hp = m.stats["hp"] * (1 + 0.04 * min(m.cubes, 10))
        m.hp += m.max_hp - old
        th = self.cfg["cubes"]["upgrade_thresholds"]
        while m.thresh_i < len(th) and m.cubes >= th[m.thresh_i]:
            m.thresh_i += 1
            m.offers.append(self.draw_offer(m))
        m.events.append({"type": "pickup", "cubes": m.cubes})

    def draw_offer(self, m):
        ok = []
        for u in self.cfg["upgrades"]:
            if m.upgrades.count(u["id"]) >= u["max"]:
                continue
            if u.get("requires") and not m.holds(u["requires"]):
                continue
            ok.append(u)
        self.rng.shuffle(ok)
        return ok[:3]

    def apply_upgrade(self, m, u):
        m.upgrades.append(u["id"])
        for e in u["effects"]:
            grp, key = e["path"].split(".", 1)
            if grp == "mech":
                if key == "hp":
                    old = m.max_hp
                    m.stats["hp"] = m.stats["hp"] * e.get("mul", 1) + e.get("add", 0)
                    m.max_hp = m.stats["hp"] * (1 + 0.04 * min(m.cubes, 10))
                    m.hp += m.max_hp - old
                else:
                    m.stats[key] = m.stats.get(key, 0) * e.get("mul", 1) + e.get("add", 0)
            else:
                kind, k2 = key.split(".")
                m.wstats[kind][k2] = m.wstats[kind][k2] * e.get("mul", 1) + e.get("add", 0)
        m.events.append({"type": "upgrade", "id": u["id"]})

    # ------------------------------------------------------------------ vision
    def visible(self, viewer: Mech, pos, height):
        e = (viewer.pos[0], viewer.pos[1], viewer.pos[2] + 8 * viewer.scale())
        dx, dy = pos[0] - e[0], pos[1] - e[1]
        d = math.hypot(dx, dy)
        near = 45 + viewer.stats.get("near_range_bonus", 0)
        far = 200 + viewer.stats.get("vision_bonus", 0)
        if d > far:
            return False
        if d > near and abs(wrap(math.atan2(dy, dx) - viewer.aim[0])) > math.radians(60):
            return False
        return self.w.los(e, (pos[0], pos[1], pos[2] + height * 0.5)) or self.w.los(e, (pos[0], pos[1], pos[2] + height))

    # ------------------------------------------------------------------ observation
    def observe(self):
        p = self.player
        mechs = []
        for m in self.mechs:
            if m is p or not m.alive:
                continue
            seen = self.visible(p, m.pos, 9 * m.scale())
            if seen:
                self.memory[m.id] = self.t
            if seen or self.t - self.memory.get(m.id, -99) < 5:
                mechs.append({"id": m.id, "kind": "mech", "pos": [round(v, 2) for v in m.pos], "vel": [round(v, 2) for v in m.vel],
                              "aim": [round(v, 3) for v in m.aim], "hp": round(m.hp, 2), "max_hp": round(m.max_hp, 2), "cubes": m.cubes,
                              "weapons": [w["kind"] for w in m.weapons if w],
                              "charging": any(w and w["kind"] == "laser" and w["charge"] > 0 for w in m.weapons)})
        for n in self.npcs:
            if math.dist(n["pos"][:2], p.pos[:2]) < 200 and self.visible(p, n["pos"], 3.5):
                mechs.append({"id": n["id"], "kind": "npc", "pos": list(n["pos"]), "vel": list(n["vel"]), "aim": [0.0, 0.0],
                              "hp": n["hp"], "max_hp": 350})

        def vis_items(items, h):
            return [it for it in items if math.dist(it["pos"][:2], p.pos[:2]) < 200 and self.visible(p, it["pos"], h)]

        z = self.zone_now()
        offer = None
        if p.offers:
            if not p.offer_t:
                p.offer_t = self.t
                p.offer_id += 1
            offer = {"id": p.offer_id, "choices": [{"id": u["id"], "name": u["name"], "desc": u["desc"]} for u in p.offers[0]],
                     "expires": p.offer_t + 5}
        snd = [{k: v for k, v in s.items() if k != "_t"} | {"age": round(self.t - s["_t"], 2)} for s in self.sounds[self.pid]]
        self.sounds[self.pid] = []
        obs = {
            "type": "obs", "tick": self.tick, "t": round(self.t, 2), "alive": sum(1 for m in self.mechs if m.alive),
            "self": {"id": p.id, "pos": [round(v, 2) for v in p.pos], "vel": [round(v, 2) for v in p.vel],
                     "aim": [round(v, 3) for v in p.aim], "hp": round(p.hp, 2), "max_hp": round(p.max_hp, 2),
                     "energy": round(p.energy, 1), "max_energy": p.stats["energy"], "grounded": p.grounded, "cubes": p.cubes,
                     "upgrades": list(p.upgrades), "in_zone": self.in_zone(p.pos, z),
                     "weapons": [None if w is None else {"kind": w["kind"], "cooldown": round(max(0, w["cooldown"]), 2),
                                                         **({"ammo": w["ammo"], "reload": round(w["reload"], 2)} if w["kind"] == "gun" else {}),
                                                         **({"charge": round(w["charge"], 2)} if w["kind"] == "laser" else {}),
                                                         "stats": dict(p.wstats[w["kind"]])} for w in p.weapons],
                     "stats": dict(p.stats, eye_height=8 * p.scale(), muzzle_height=7 * p.scale())},
            "mechs": mechs,
            "projectiles": [{"id": i, "kind": pr.kind, "pos": [round(v, 2) for v in pr.pos], "vel": [round(v, 2) for v in pr.vel]}
                            for i, pr in enumerate(self.projs) if pr.owner != p.id and math.dist(pr.pos[:2], p.pos[:2]) < 200
                            and self.visible(p, [pr.pos[0], pr.pos[1], pr.pos[2] - 0.5], 1.0)],
            "crates": [{"id": c["id"], "pos": c["pos"], "hp": c["hp"]} for c in vis_items(self.crates, 2.4)],
            "cubes": vis_items(self.cubes, 1.0),
            "pickups": vis_items(self.pickups, 1.0),
            "sounds": snd,
            "beams": [b for b in self.beams if b["by"] != p.id],
            "events": p.events,
            "zone": z,
            "offer": offer,
            "drops": [{"id": d["id"], "pos": d["pos"], "radius": 8, "lands_in": round(max(0, d["land"] - self.t), 2), "capture_s": 4,
                       "progress": round(d["progress"], 2), "holder": d["holder"]} for d in self.drops],
        }
        p.events = []
        self.beams = []
        return obs

    # ------------------------------------------------------------------ actions
    def apply_player(self, act):
        p = self.player
        mv = act.get("move", p.move)
        if isinstance(mv, (list, tuple)) and len(mv) == 2:
            l = math.hypot(mv[0], mv[1])
            p.move = [mv[0] / max(1, l), mv[1] / max(1, l)]
        p.boost = bool(act.get("boost", p.boost))
        p.jump = bool(act.get("jump", False))
        if "aim" in act:
            p.cmd_aim = [float(act["aim"][0]), float(act["aim"][1])]
        f = act.get("fire", [False, False])
        p.fire = [bool(f[0]) if len(f) > 0 else False, bool(f[1]) if len(f) > 1 else False]
        p.reload_req = bool(act.get("reload", False))
        if "pick" in act and p.offers:
            i = int(act["pick"])
            if 0 <= i < len(p.offers[0]):
                self.apply_upgrade(p, p.offers[0][i])
                p.offers.pop(0)
                p.offer_t = 0.0
        if "take" in act:
            self.do_take(p, act["take"], act.get("drop"))
        elif "drop" in act:
            s = int(act["drop"])
            if p.weapons[s]:
                self.add_pickup(p.weapons[s]["kind"], p.pos)
                p.weapons[s] = None

    def do_take(self, m: Mech, pid, drop):
        pk = next((q for q in self.pickups if q["id"] == pid), None)
        if pk is None:
            m.events.append({"type": "take_failed", "id": pid, "reason": "gone"})
            return
        if math.dist(pk["pos"][:2], m.pos[:2]) > 4 or abs(pk["pos"][2] - m.pos[2]) > 4:
            m.events.append({"type": "take_failed", "id": pid, "reason": "range"})
            return
        if m.holds(pk["kind"]):
            m.events.append({"type": "take_failed", "id": pid, "reason": "have"})
            return
        slot = next((i for i, w in enumerate(m.weapons) if w is None), None)
        if slot is None:
            if drop is None:
                m.events.append({"type": "take_failed", "id": pid, "reason": "full"})
                return
            slot = int(drop)
            old = m.weapons[slot]
            if old["kind"] == "laser" and old["charge"] > 0:
                m.events.append({"type": "take_failed", "id": pid, "reason": "busy"})
                return
            self.add_pickup(old["kind"], m.pos)
        self.pickups.remove(pk)
        w = m.new_weapon(pk["kind"])
        w["cooldown"] = max(1.0, m.wstats[pk["kind"]]["cooldown"])
        m.weapons[slot] = w
        m.events.append({"type": "took", "id": pid, "kind": pk["kind"], "slot": slot})

    # ------------------------------------------------------------------ squad brain
    def house_think(self):
        p = self.player
        n_hunt = min(15, 1 + self.player_kills + int(max(0.0, self.t - 60) / 50))
        if self.t > 300:
            n_hunt = 15
        live_ids = [i for i in self.hunt_order if self.mechs[i].alive]
        hunters = set(live_ids[:n_hunt])
        # shared vision of the player
        if p.alive:
            for i in live_ids:
                m = self.mechs[i]
                if math.dist(m.pos[:2], p.pos[:2]) < 200 and self.visible(m, p.pos, 9 * p.scale()):
                    self.squad_seen_t = self.t
                    self.squad_seen_pos = list(p.pos)
                    break
        seen_now = self.t - self.squad_seen_t < 0.15
        z = self.zone_now()
        for i in live_ids:
            m, br = self.mechs[i], self.brains[i]
            # loot emulation: specials appear in the slot over time
            if m.weapons[1] is None and self.t > br.loot_t:
                m.weapons[1] = m.new_weapon(br.rng.choice(["laser", "laser", "mortar", "lightning"]))
            if m.offers:
                u = m.offers.pop(0)
                if u:
                    self.apply_upgrade(m, br.rng.choice(u))
            hunting = i in hunters and p.alive
            zc, zr = z["next_center"], z["next_radius"]
            tgt = None
            if hunting:
                tgt = list(p.pos) if (seen_now or self.t - self.squad_seen_t > 6) else self.squad_seen_pos or list(p.pos)
            dpl = math.dist(m.pos[:2], p.pos[:2])
            # movement
            if hunting:
                special = m.weapons[1]["kind"] if m.weapons[1] else "gun"
                pref = {"laser": 110, "mortar": 120, "lightning": 28, "gun": 70}[special]
                dx, dy = tgt[0] - m.pos[0], tgt[1] - m.pos[1]
                d = math.hypot(dx, dy) or 1
                ux, uy = dx / d, dy / d
                if self.t - br.strafe_t > br.rng.uniform(1.5, 4):
                    br.strafe = -br.strafe
                    br.strafe_t = self.t
                if seen_now and d < pref * 1.3:
                    k = 1.0 if d > pref else -0.6
                    mv = (ux * k - uy * br.strafe * 0.8, uy * k + ux * br.strafe * 0.8)
                else:
                    mv = (ux, uy)
            else:
                if br.wander is None or math.dist(br.wander, m.pos[:2]) < 10 or math.dist(br.wander, zc) > zr:
                    a = br.rng.uniform(-PI, PI)
                    rr = br.rng.uniform(0, zr * 0.7)
                    br.wander = (zc[0] + rr * math.cos(a), zc[1] + rr * math.sin(a))
                dx, dy = br.wander[0] - m.pos[0], br.wander[1] - m.pos[1]
                d = math.hypot(dx, dy) or 1
                mv = (dx / d * 0.6, dy / d * 0.6)
            if not self.in_zone(m.pos, z) or math.dist(m.pos[:2], zc) > zr and self.t > z["shrink_start"] - 15:
                dx, dy = zc[0] - m.pos[0], zc[1] - m.pos[1]
                d = math.hypot(dx, dy) or 1
                mv = (dx / d, dy / d)
            m.move = list(mv)
            moved = math.dist(br.last_xy, m.pos[:2])
            br.last_xy = tuple(m.pos[:2])
            br.stuck_t = br.stuck_t + 0.1 if moved < 0.3 and math.hypot(*mv) > 0.3 else 0
            m.jump = br.stuck_t > 0.5
            m.boost = br.stuck_t > 0.5 or (hunting and dpl > 250 and m.energy > 50)
            if br.stuck_t > 1.5:
                m.move = [-mv[1], mv[0]] if br.rng.random() < 0.5 else [mv[1], -mv[0]]
            # aiming / firing at the player (house bots only fight the player, plus drones nearby)
            m.fire = [False, False]
            sees = p.alive and dpl < 230 and seen_now and self.t - self.squad_seen_t < 0.15 and self.w.los(
                (m.pos[0], m.pos[1], m.pos[2] + 8), (p.pos[0], p.pos[1], p.pos[2] + 5))
            if sees and (hunting or dpl < 90):
                if br.seen_t < 0:
                    br.seen_t = self.t
                tx, ty, tz = p.pos[0], p.pos[1], p.pos[2] + 4.5 * p.scale()
                ox, oy, oz = m.pos[0], m.pos[1], m.pos[2] + 7 * m.scale()
                tf = math.dist((tx, ty, tz), (ox, oy, oz)) / 252 * br.lead
                tx += p.vel[0] * tf
                ty += p.vel[1] * tf
                yaw = math.atan2(ty - oy, tx - ox) + br.rng.gauss(0, br.noise)
                pitch = math.atan2(tz - oz, math.hypot(tx - ox, ty - oy)) + br.rng.gauss(0, br.noise * 0.5)
                m.cmd_aim = [yaw, pitch]
                if self.t - br.seen_t > br.react:
                    err = abs(wrap(yaw - m.aim[0]))
                    w1 = m.weapons[1]
                    m.fire[0] = err < 0.08 and dpl < 225
                    if w1:
                        k = w1["kind"]
                        if k == "laser":
                            m.fire[1] = not (w1["charge"] >= m.wstats["laser"]["charge"] - 0.01 and err < 0.03)
                        elif k == "lightning":
                            m.fire[1] = dpl < 44 and err < 0.25
                        elif k == "mortar":
                            dd = math.hypot(p.pos[0] - ox, p.pos[1] - oy)
                            v = m.wstats["mortar"]["speed"]
                            s = (9.81 * 0 + 22) * dd / (v * v)
                            if s < 1 and dd > 25:
                                m.cmd_aim = [yaw, 0.5 * math.asin(s)]
                                m.fire = [False, abs(m.aim[1] - m.cmd_aim[1]) < 0.05 and err < 0.06]
            else:
                br.seen_t = -1
                if hunting and tgt:
                    m.cmd_aim = [math.atan2(tgt[1] - m.pos[1], tgt[0] - m.pos[0]), 0.0]
                elif math.hypot(*m.move) > 0.1:
                    m.cmd_aim = [math.atan2(m.move[1], m.move[0]), 0.0]
                # finish a laser charge elsewhere rather than holding forever
                if m.weapons[1] and m.weapons[1]["kind"] == "laser" and m.weapons[1]["charge"] > 0:
                    m.fire[1] = False

    # ------------------------------------------------------------------ physics
    def step_mech(self, m: Mech, z):
        st = m.stats
        sc = m.scale()
        rad = 2.5 * sc
        charging = any(w and w["kind"] == "laser" and w["charge"] > 0 for w in m.weapons)
        mv = m.move
        spd = st["walk_speed"]
        boosting = False
        if m.boost and m.grounded and math.hypot(*mv) > 0.1 and m.energy > 0:
            spd = st["boost_speed"]
            boosting = True
            m.energy -= st["boost_drain"] * DT
            m.energy_t = self.t
        if charging:
            spd *= st["charge_move_mult"]
        des = (mv[0] * spd, mv[1] * spd)
        acc = st["ground_accel"] if m.grounded else st["air_accel"]
        for k in range(2):
            dv = des[k] - m.vel[k]
            m.vel[k] += max(-acc * DT, min(acc * DT, dv))
        if m.jump and m.grounded:
            m.vel[2] = st["jump_speed"]
            m.grounded = False
        if m.boost and not m.grounded and m.energy > 0:
            m.vel[2] = min(st["max_rise_speed"], m.vel[2] + st["jet_accel"] * DT)
            m.energy -= st["jet_drain"] * DT
            m.energy_t = self.t
            boosting = True
        if not m.grounded:
            m.vel[2] -= st["gravity"] * DT
        if self.t - m.energy_t > st["energy_regen_delay"]:
            m.energy = min(st["energy"], m.energy + st["energy_regen"] * DT)
        m.energy = max(0.0, m.energy)
        # horizontal with slide
        x, y, zz = m.pos

        def blocked(nx, ny):
            for hz in (zz + st["step_up"] + 0.15, zz + 4.5 * sc, zz + 8.7 * sc):
                for ox, oy in ((0, 0), (rad, 0), (-rad, 0), (0, rad), (0, -rad), (0.7 * rad, 0.7 * rad), (-0.7 * rad, 0.7 * rad),
                               (0.7 * rad, -0.7 * rad), (-0.7 * rad, -0.7 * rad)):
                    if self.w.solid(nx + ox, ny + oy, hz):
                        return True
            # slope / terrain step
            if self.w.terrain(nx, ny) > zz + st["step_up"] + 0.3:
                return True
            return False

        nx, ny = x + m.vel[0] * DT, y + m.vel[1] * DT
        if blocked(nx, ny):
            if not blocked(nx, y):
                ny = y
                m.vel[1] = 0
            elif not blocked(x, ny):
                nx = x
                m.vel[0] = 0
            else:
                nx, ny = x, y
                m.vel[0] = m.vel[1] = 0
        nz = zz + m.vel[2] * DT
        floor = self.w.support(nx, ny, max(zz, nz), st["step_up"])
        if nz <= floor:
            nz = floor
            m.vel[2] = 0
            m.grounded = True
        else:
            m.grounded = nz - floor < 0.05
        m.pos = [nx, ny, nz]
        # aim slew
        ty = wrap(m.cmd_aim[0] - m.aim[0])
        mx = st["turn_rate"] * DT
        m.aim[0] = wrap(m.aim[0] + max(-mx, min(mx, ty)))
        tp = max(st["min_pitch"], min(st["max_pitch"], m.cmd_aim[1]))
        mp = st["pitch_rate"] * DT
        m.aim[1] += max(-mp, min(mp, tp - m.aim[1]))
        # sounds
        if m is not self.player:
            sp = math.hypot(m.vel[0], m.vel[1])
            if m.grounded and sp > 5 and self.t - m.step_t > 0.6:
                m.step_t = self.t
                self.sound("boost" if boosting else "step", m.pos)
        # zone
        if not self.in_zone(m.pos, z):
            self.damage(m, z["dps"] * DT, "zone", None, None)
            m.last_combat = self.t
        elif self.t - m.last_combat > st["regen_delay"] and m.hp < m.max_hp:
            m.hp = min(m.max_hp, m.hp + st["regen_per_s"] * DT)
        # cubes
        for c in list(self.cubes):
            if abs(c["pos"][0] - nx) < 3.5 and abs(c["pos"][1] - ny) < 3.5 and math.dist(c["pos"][:2], (nx, ny)) < 3.5 and abs(c["pos"][2] - nz) < 4:
                self.cubes.remove(c)
                self.give_cube(m)

    def fire_weapons(self, m: Mech):
        sc = m.scale()
        mz = (m.pos[0], m.pos[1], m.pos[2] + 7 * sc)
        yaw, pitch = m.aim
        for s, w in enumerate(m.weapons):
            if w is None:
                continue
            ws = m.wstats[w["kind"]]
            w["cooldown"] -= DT
            k = w["kind"]
            trig = m.fire[s]
            if k == "gun":
                if w["reload"] > 0:
                    w["reload"] -= DT
                    if w["reload"] <= 0:
                        w["reload"] = 0
                        w["ammo"] = int(ws["mag"])
                    continue
                if m.reload_req and w["ammo"] < ws["mag"]:
                    w["reload"] = ws["reload"]
                    continue
                if trig and w["cooldown"] <= 0 and w["ammo"] > 0:
                    w["cooldown"] += ws["cooldown"] if w["cooldown"] > -DT else ws["cooldown"] - w["cooldown"]
                    w["ammo"] -= 1
                    sy = yaw + random.gauss(0, math.radians(ws["spread_deg"]))
                    sp = pitch + random.gauss(0, math.radians(ws["spread_deg"]))
                    d = (math.cos(sp) * math.cos(sy), math.cos(sp) * math.sin(sy), math.sin(sp))
                    o = [mz[0] + d[0] * 3, mz[1] + d[1] * 3, mz[2] + d[2] * 3]
                    self.projs.append(Proj("gun", o, [d[0] * ws["speed"], d[1] * ws["speed"], d[2] * ws["speed"]], m.id, ws["range"], ws["damage"]))
                    self.sound("gun", m.pos) if m is not self.player else None
                    if w["ammo"] == 0:
                        w["reload"] = ws["reload"]
                if w["cooldown"] < 0:
                    w["cooldown"] = 0
            elif k == "mortar":
                if trig and w["cooldown"] <= 0:
                    w["cooldown"] = ws["cooldown"]
                    o = (m.pos[0], m.pos[1], m.pos[2] + 9.5 * sc)
                    d = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch))
                    o = [o[0] + d[0] * 3, o[1] + d[1] * 3, o[2] + d[2] * 3]
                    self.projs.append(Proj("mortar", o, [d[0] * ws["speed"], d[1] * ws["speed"], d[2] * ws["speed"]], m.id, 1e9, ws["damage"]))
                    if m is not self.player:
                        self.sound("mortar", m.pos)
            elif k == "lightning":
                if trig and w["cooldown"] <= 0 and m.energy >= ws["energy"]:
                    w["cooldown"] = ws["cooldown"]
                    m.energy -= ws["energy"]
                    m.energy_t = self.t
                    best, bd = None, ws["range"]
                    for q in self.targets(m):
                        qp = q["pos"] if isinstance(q, dict) else q.pos
                        d = math.dist(qp[:2], m.pos[:2])
                        if d < bd and abs(wrap(math.atan2(qp[1] - m.pos[1], qp[0] - m.pos[0]) - yaw)) < math.radians(ws["cone_deg"] / 2) \
                                and self.w.los(mz, (qp[0], qp[1], qp[2] + 2)):
                            best, bd = q, d
                    if m is not self.player:
                        self.sound("lightning", m.pos)
                    if best is not None:
                        self.hit_any(best, ws["damage"], "lightning", m, mz, effect="short")
            elif k == "laser":
                if w["cooldown"] > 0 and w["charge"] <= 0:
                    continue
                if m.short_until > self.t:
                    w["charge"] = 0
                    continue
                if trig or (w["charge"] > 0 and w["charge"] < ws["min_charge"]):
                    if w["charge"] < ws["charge"]:
                        w["charge"] = min(ws["charge"], w["charge"] + DT)
                        w["full_t"] = self.t
                    elif self.t - w["full_t"] > ws["hold_s"]:
                        self.fire_laser(m, w, ws, mz)
                    if w["charge"] == DT and m is not self.player:
                        self.sound("laser_charge", m.pos)
                elif w["charge"] > 0:
                    self.fire_laser(m, w, ws, mz)

    def fire_laser(self, m, w, ws, mz):
        frac = ws["min_frac"] + (1 - ws["min_frac"]) * min(1.0, w["charge"] / ws["charge"])
        w["charge"] = 0
        w["cooldown"] = ws["cooldown"]
        yaw, pitch = m.aim
        wall = self.w.ray(mz, yaw, pitch, ws["range"])
        d = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch))
        best, bt = None, wall
        for q in self.targets(m):
            qp = q["pos"] if isinstance(q, dict) else q.pos
            if isinstance(q, Mech):
                r, h = 2.5 * q.scale(), 9 * q.scale()
            elif q["id"] < 1000:
                r, h = 1.8, 3.5
            else:
                r, h = 1.4, 2.4
            t = self.ray_cyl(mz, d, qp, r, h, bt)
            if t is not None and t < bt:
                best, bt = q, t
        end = [mz[0] + d[0] * bt, mz[1] + d[1] * bt, mz[2] + d[2] * bt]
        if m is not self.player:
            self.sound("laser", m.pos)
            self.beams.append({"by": m.id, "kind": "laser", "path": [list(mz), end], "age": 0.0})
        if best is not None:
            dmg = ws["damage"] * frac
            eff = None
            if isinstance(best, Mech) and best.holds("mortar"):
                dmg *= 1 + ws["cookoff"]
                eff = "cookoff"
            self.hit_any(best, dmg, "laser", m, mz, effect=eff)

    @staticmethod
    def ray_cyl(o, d, base, r, h, tmax):
        # 2D closest approach then check z range; sampled coarse but adequate
        px, py = base[0] - o[0], base[1] - o[1]
        dl2 = d[0] * d[0] + d[1] * d[1]
        if dl2 < 1e-9:
            return None
        t = (px * d[0] + py * d[1]) / dl2
        if t < 0 or t > tmax:
            return None
        cx, cy = o[0] + d[0] * t - base[0], o[1] + d[1] * t - base[1]
        if cx * cx + cy * cy > r * r:
            return None
        z = o[2] + d[2] * t
        if base[2] - 0.5 <= z <= base[2] + h + 0.5:
            return t
        return None

    def targets(self, m):
        out = [q for q in self.mechs if q.alive and q is not m]
        if m is self.player:
            out += self.npcs + self.crates
        else:
            out += [n for n in self.npcs if n["target"] == m.id]
        return out

    def hit_any(self, q, dmg, weapon, src, from_pos, effect=None):
        if isinstance(q, Mech):
            if effect == "short" and q.holds("laser"):
                q.short_until = self.t + 2
            elif effect == "short":
                effect = None
            self.damage(q, dmg, weapon, src, from_pos, effect)
        elif q["id"] < 1000:
            self.damage_npc(q, dmg, src)
        else:
            self.damage_crate(q, dmg * (1 + 0.04 * min(src.cubes, 10)), src)

    def step_projs(self):
        keep = []
        mech_r = [(q, 2.5 * q.scale(), 9 * q.scale()) for q in self.mechs if q.alive]
        for pr in self.projs:
            g = self.cfg["weapons"]["mortar"]["gravity"] if pr.kind == "mortar" else 0
            pr.vel[2] -= g * DT
            o = list(pr.pos)
            n = [o[0] + pr.vel[0] * DT, o[1] + pr.vel[1] * DT, o[2] + pr.vel[2] * DT]
            seg = math.dist(o, n)
            pr.dist += seg
            pr.ttl -= DT
            hit = None
            ht = 2.0
            d = [(n[0] - o[0]) / max(seg, 1e-6), (n[1] - o[1]) / max(seg, 1e-6), (n[2] - o[2]) / max(seg, 1e-6)]
            for q, r, h in mech_r:
                if q.id == pr.owner:
                    continue
                if abs(q.pos[0] - o[0]) > seg + 12 or abs(q.pos[1] - o[1]) > seg + 12:
                    continue
                t = self.ray_cyl(o, d, q.pos, r + 0.3, h, seg)
                if t is not None and t < ht * seg:
                    hit, ht = q, t / seg
            owner = self.mechs[pr.owner] if pr.owner < 16 else None
            if owner is self.player or pr.kind == "npc":
                for c in self.crates if owner is self.player else []:
                    if abs(c["pos"][0] - o[0]) < seg + 3 and abs(c["pos"][1] - o[1]) < seg + 3:
                        t = self.ray_cyl(o, d, c["pos"], 1.7, 2.4, seg)
                        if t is not None and t < ht * seg:
                            hit, ht = c, t / seg
                for c in self.npcs if owner is not None else []:
                    if abs(c["pos"][0] - o[0]) < seg + 3 and abs(c["pos"][1] - o[1]) < seg + 3:
                        t = self.ray_cyl(o, d, c["pos"], 2.1, 3.5, seg)
                        if t is not None and t < ht * seg:
                            hit, ht = c, t / seg
            wall_t = None
            ns = max(1, int(seg / 2.5))
            for k in range(1, ns + 1):
                f = k / ns
                x, y, z = o[0] + (n[0] - o[0]) * f, o[1] + (n[1] - o[1]) * f, o[2] + (n[2] - o[2]) * f
                if self.w.solid(x, y, z) or self.w.terrain(x, y) > z:
                    wall_t = (k - 1) / ns
                    break
            if wall_t is not None and (hit is None or wall_t < ht):
                hit, ht = "wall", wall_t
            if hit is None:
                if pr.dist < pr.rng and pr.ttl > 0:
                    pr.pos = n
                    keep.append(pr)
                continue
            at = [o[0] + (n[0] - o[0]) * ht, o[1] + (n[1] - o[1]) * ht, o[2] + (n[2] - o[2]) * ht]
            if pr.kind == "mortar":
                self.explode(pr, at, hit if hit != "wall" else None, owner)
            elif hit != "wall":
                if pr.kind == "npc":
                    if isinstance(hit, Mech):
                        self.damage(hit, pr.dmg, "npc", None, at)
                        hit.events[-1]["by"] = 100
                elif owner is not None:
                    self.hit_any(hit, pr.dmg, "gun", owner, o)
        self.projs = keep

    def explode(self, pr, at, direct, owner):
        ws = self.cfg["weapons"]["mortar"]
        self.sound("explosion", at)
        rad = owner.wstats["mortar"]["splash_radius"] if owner else ws["splash_radius"]
        if direct is not None and owner is not None:
            self.hit_any(direct, ws["damage"], "mortar", owner, at)
        for q in self.mechs:
            if not q.alive or q is direct:
                continue
            d = math.dist(at, (q.pos[0], q.pos[1], q.pos[2] + 4.5))
            if d < rad + 2.5:
                f = 1 - (1 - ws["splash_edge"]) * min(1, max(0, d - 2.5) / rad)
                dmg = ws["splash"] * f * (0.5 if q is owner else 1)
                if owner:
                    self.damage(q, dmg, "mortar", owner if q is not owner else None, at, "concuss")
                q.energy = max(0, q.energy - ws["drain"] * f)
                ux, uy = q.pos[0] - at[0], q.pos[1] - at[1]
                ul = math.hypot(ux, uy) or 1
                q.vel[0] += ux / ul * ws["knock"] * f
                q.vel[1] += uy / ul * ws["knock"] * f
                q.vel[2] += ws["knock_up"] * f
                q.grounded = False
                q.events.append({"type": "effect", "effect": "concuss", "by": owner.id if owner else None})
        if owner is self.player:
            for c in list(self.crates):
                if math.dist(at, c["pos"]) < rad + 1.4:
                    self.damage_crate(c, ws["splash"], owner)
            for c in list(self.npcs):
                if math.dist(at, c["pos"]) < rad + 1.8:
                    self.damage_npc(c, ws["splash"], owner)

    def step_npcs(self):
        cfg = self.cfg["npc"]
        for n in self.npcs:
            n["cool"] -= DT
            tgt = None
            if n["target"] is not None:
                q = self.mechs[n["target"]] if n["target"] < 16 else None
                if q is None or not q.alive or math.dist(q.pos[:2], n["pos"][:2]) > cfg["deaggro_range"] or math.dist(n["home"], n["pos"][:2]) > cfg["chase_leash"]:
                    n["target"] = None
                else:
                    tgt = q
            if tgt is None and math.dist(n["home"], n["pos"][:2]) < cfg["leash"]:
                for q in self.mechs:
                    if q.alive and math.dist(q.pos[:2], n["pos"][:2]) < cfg["aggro_range"]:
                        tgt = q
                        break
            if tgt is not None and n["cool"] <= 0:
                o = (n["pos"][0], n["pos"][1], n["pos"][2] + 3)
                tp = (tgt.pos[0], tgt.pos[1], tgt.pos[2] + 4.5)
                if self.w.los(o, tp, 3.0):
                    n["cool"] = cfg["cooldown"]
                    yaw = math.atan2(tp[1] - o[1], tp[0] - o[0]) + random.gauss(0, math.radians(cfg["spread_deg"]))
                    pitch = math.atan2(tp[2] - o[2], math.dist(tp[:2], o[:2]))
                    v = cfg["projectile_speed"]
                    d = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch))
                    self.projs.append(Proj("npc", [o[0] + d[0] * 2.5, o[1] + d[1] * 2.5, o[2] + d[2] * 2.5],
                                           [d[0] * v, d[1] * v, d[2] * v], 99, 120, cfg["damage"]))
                    if self.player.alive:
                        self.sound("npc_gun", o)

    def step_drops(self):
        for i, tm in enumerate(self.drop_times):
            if tm is not None and self.t >= tm:
                self.drop_times[i] = None
                z = self.zone_now()
                c, r = z["next_center"], z["next_radius"]
                for _ in range(200):
                    x, y = self.rng.choice(self.w.free)
                    if math.dist((x, y), c) < r * 0.7:
                        break
                self.drops.append({"id": 20000 + i, "pos": [x, y, self.w.terrain(x, y)], "land": self.t + 25, "progress": 0.0,
                                   "holder": None})
        for d in list(self.drops):
            if self.t < d["land"]:
                continue
            inside = [m for m in self.mechs if m.alive and math.dist(m.pos[:2], d["pos"][:2]) < 8]
            if len(inside) == 1:
                if d["holder"] != inside[0].id:
                    d["holder"] = inside[0].id
                    d["progress"] = 0
                d["progress"] += DT
                if d["progress"] >= 4:
                    m = inside[0]
                    for _ in range(3):
                        self.give_cube(m)
                    m.hp = m.max_hp
                    m.energy = m.stats["energy"]
                    m.events.append({"type": "drop", "id": d["id"], "cubes": m.cubes})
                    self.drops.remove(d)
            elif not inside:
                d["holder"] = None
                d["progress"] = 0

    # ------------------------------------------------------------------ main loop
    def run(self):
        p = self.player
        end_t = self.cfg["match"]["duration_s"]
        while self.t < end_t:
            if self.tick % 2 == 0:
                if p.alive:
                    obs = self.observe()
                    t0 = time.perf_counter()
                    try:
                        act = self.bot.act(obs) or {}
                    except Exception as e:  # mimic the wrapper: coast
                        import traceback
                        if self.crash is None:
                            self.crash = traceback.format_exc()
                        act = {}
                    self.act_times.append((time.perf_counter() - t0) * 1000)
                    self.apply_player(act)
                self.house_think()
            z = self.zone_now()
            for m in self.mechs:
                if m.alive:
                    self.step_mech(m, z)
            for m in self.mechs:
                if m.alive:
                    self.fire_weapons(m)
            self.step_projs()
            self.step_npcs()
            self.step_drops()
            self.t += DT
            self.tick += 1
            alive = [m for m in self.mechs if m.alive]
            if not p.alive or len(alive) <= 1:
                break
        if p.alive:
            alive = sorted([m for m in self.mechs if m.alive], key=lambda m: -m.hp)
            place = alive.index(p) + 1
        else:
            place = p.place
        return {"place": place, "points": 16 - place, "kills": p.kills, "t": round(self.t, 1), "cubes": p.cubes,
                "act_ms_p99": float(np.percentile(self.act_times, 99)) if self.act_times else 0,
                "act_ms_max": max(self.act_times) if self.act_times else 0, "crash": self.crash,
                "weapons": [w and w["kind"] for w in p.weapons]}


ARENA = os.environ.get("MECH_ARENA", os.path.join(HERE, "..", "arena"))
CFG_PATH = os.environ.get("MECH_CONFIG", os.path.join(HERE, "config.json"))


def load_bot(path):
    spec = importlib.util.spec_from_file_location("botmod_%d" % abs(hash(path)), path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(mod)
    return getattr(mod, "BOT_CLASS", None) or getattr(mod, "AegisV2", None) or getattr(mod, "Bot")


def run_one(args):
    bot_path, seed = args
    import io
    import contextlib
    cfg = json.load(open(CFG_PATH))
    world = World(ARENA)
    factory = load_bot(bot_path)
    with contextlib.redirect_stderr(io.StringIO()):
        res = Match(world, cfg, factory, seed).run()
    res["seed"] = seed
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", action="append", required=True)
    ap.add_argument("--matches", type=int, default=16)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    from multiprocessing import Pool
    World(ARENA)  # build raster cache once
    for bot in a.bot:
        t0 = time.time()
        with Pool(a.jobs) as pool:
            res = pool.map(run_one, [(bot, a.seed * 1000 + k) for k in range(a.matches)])
        pts = [r["points"] for r in res]
        print(f"{bot}: mean(16-place) = {sum(pts) / len(pts):.2f}  kills/match {sum(r['kills'] for r in res) / len(res):.2f}  "
              f"survive_t {sum(r['t'] for r in res) / len(res):.0f}s  cubes {sum(r['cubes'] for r in res) / len(res):.1f}  "
              f"act p99 {max(r['act_ms_p99'] for r in res):.1f}ms max {max(r['act_ms_max'] for r in res):.1f}ms  ({time.time() - t0:.0f}s)")
        print("   places:", sorted(r["place"] for r in res))
        crashes = [r["crash"] for r in res if r["crash"]]
        if crashes:
            print("   CRASH:", crashes[0][-1500:])


if __name__ == "__main__":
    main()
