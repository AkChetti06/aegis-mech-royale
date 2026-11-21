"""Aegis v3: a mechfight/1 bot for Builders Mech Royale.

Practice and qualifying both reward placement, and in practice the 15 house bots act as one squad
that hunts you. Mechs only die before you if somebody kills them, so placement comes from winning
fights: taking hunters down while they arrive one or two at a time, and staying alive long enough
to do it again. Aegis v3 is built around that:

* a rasterized copy of the map (stdlib only, built at startup) for line-of-sight checks, cover
  search and wall-aware steering;
* early laser hunting and cube farming (crates and drones) to scale damage and HP;
* gun + laser fired together with predictive aim, both gated on yaw *and* pitch error;
* always facing the threat (rear hits deal 1.5x), strafing at a weapon-appropriate range;
* breaking line of sight to regenerate when hurt, and dodging visible shells and bullets;
* conservative zone handling and defensive upgrade picks.
"""
import json
import math
import os
import random
import time
from array import array

import mechfight as mf

PI = math.pi
TAU = 2 * PI


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def wrap(a):
    return (a + PI) % TAU - PI


def d2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit(x, y):
    l = math.hypot(x, y)
    return (x / l, y / l) if l > 1e-9 else (0.0, 0.0)


# ------------------------------------------------------------------------------------ map model
class Arena:
    """Terrain heights plus a 1 m raster of solid prisms (primary interval + bridge interval)."""

    def __init__(self, directory):
        self.ok = False
        try:
            with open(os.path.join(directory, "v2", "world.json"), encoding="utf-8") as fh:
                w = json.load(fh)
            t = w["terrain"]
            self.minx, self.miny = float(t["minX"]), float(t["minY"])
            self.nx, self.ny = int(t["nx"]), int(t["ny"])
            self.step = float(t.get("step", 1.0))
            scale = float(t.get("scale", 0.01))
            h = array("h")
            with open(os.path.join(directory, "v2", t.get("file", "terrain.i16")), "rb") as fh:
                h.frombytes(fh.read())
            if len(h) != self.nx * self.ny:
                return
            if array("h", [1]).tobytes()[0] != 1:
                h.byteswap()
            self.h = array("f", (v * scale for v in h))
            b = w["bounds"]
            self.bx0, self.bx1, self.by0, self.by1 = b["minX"], b["maxX"], b["minY"], b["maxY"]
            n = self.nx * self.ny
            nan = float("nan")
            self.top = array("f", [nan]) * n
            self.base = array("f", [nan]) * n
            self.top2 = array("f", [nan]) * n
            self.base2 = array("f", [nan]) * n
            for p in w.get("prisms", []):
                self._fill(p)
            self.ok = True
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            self.ok = False

    def _fill(self, p):
        rings = [p["ring"]] + list(p.get("holes") or [])
        edges = []
        for r in rings:
            pts = [(float(r[k]), float(r[k + 1])) for k in range(0, len(r) - 1, 2)]
            for a in range(len(pts)):
                edges.append((pts[a - 1], pts[a]))
        ys = p["ring"][1::2]
        base, top = float(p["base"]), float(p["top"])
        bridge = p.get("kind") == "bridge"
        T, B = (self.top2, self.base2) if bridge else (self.top, self.base)
        j0 = max(0, int(math.floor(min(ys) - self.miny)))
        j1 = min(self.ny - 1, int(math.ceil(max(ys) - self.miny)))
        for j in range(j0, j1 + 1):
            yc = self.miny + j + 0.5
            xs = []
            for (xa, ya), (xb, yb) in edges:
                if (ya > yc) != (yb > yc):
                    xs.append(xa + (yc - ya) * (xb - xa) / (yb - ya))
            xs.sort()
            row = j * self.nx
            for k in range(0, len(xs) - 1, 2):
                i0 = max(0, int(math.ceil(xs[k] - self.minx - 0.5)))
                i1 = min(self.nx - 1, int(math.floor(xs[k + 1] - self.minx - 0.5)))
                for i in range(i0, i1 + 1):
                    c = row + i
                    old = T[c]
                    if old != old or top > old:
                        T[c] = top
                        ob = B[c]
                        B[c] = base if ob != ob else min(ob, base)

    def ground(self, x, y):
        fx = clamp(x - self.minx, 0.0, self.nx - 1.001)
        fy = clamp(y - self.miny, 0.0, self.ny - 1.001)
        i, j = int(fx), int(fy)
        ax, ay = fx - i, fy - j
        k = j * self.nx + i
        h = self.h
        return (h[k] * (1 - ax) + h[k + 1] * ax) * (1 - ay) + (h[k + self.nx] * (1 - ax) + h[k + self.nx + 1] * ax) * ay

    def solid(self, x, y, z):
        if x < self.bx0 or x > self.bx1 or y < self.by0 or y > self.by1:
            return True
        i, j = int(x - self.minx), int(y - self.miny)
        if i < 0 or j < 0 or i >= self.nx or j >= self.ny:
            return True
        c = j * self.nx + i
        t = self.top[c]
        if t == t and self.base[c] <= z < t:
            return True
        t = self.top2[c]
        if t == t and self.base2[c] <= z < t:
            return True
        return False

    def los(self, a, b, step=2.0):
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        n = int(math.sqrt(dx * dx + dy * dy + dz * dz) / step)
        for k in range(1, n):
            f = k / n
            x, y, z = a[0] + dx * f, a[1] + dy * f, a[2] + dz * f
            if self.solid(x, y, z) or self.ground(x, y) > z:
                return False
        return True

    def wall(self, x, y, z, r=2.6):
        """Would a mech with feet at height z collide with something at (x, y)?"""
        for hz in (z + 1.5, z + 4.5, z + 8.5):
            for ox, oy in ((0, 0), (r, 0), (-r, 0), (0, r), (0, -r)):
                if self.solid(x + ox, y + oy, hz):
                    return True
        return self.ground(x, y) > z + 1.6


# ------------------------------------------------------------------------------------ aiming
def lead_aim(origin, target, tvel, speed, max_t=2.0):
    ox, oy, oz = origin
    tx, ty, tz = target
    vx, vy, vz = tvel
    t = 0.0
    if speed:
        dx, dy, dz = tx - ox, ty - oy, tz - oz
        a = vx * vx + vy * vy + vz * vz - speed * speed
        b = 2 * (dx * vx + dy * vy + dz * vz)
        c = dx * dx + dy * dy + dz * dz
        if abs(a) > 1e-6:
            disc = b * b - 4 * a * c
            if disc >= 0:
                s = math.sqrt(disc)
                cands = [v for v in ((-b - s) / (2 * a), (-b + s) / (2 * a)) if v > 0]
                if cands:
                    t = min(cands)
        if t <= 0:
            t = math.sqrt(c) / speed
        t = min(t, max_t)
    px, py, pz = tx + vx * t, ty + vy * t, tz + vz * t
    return math.atan2(py - oy, px - ox), math.atan2(pz - oz, max(0.01, math.hypot(px - ox, py - oy))), t


def mortar_aim(origin, target, tvel, speed, g):
    ox, oy, oz = origin
    tx, ty, tz = target
    flight = math.hypot(tx - ox, ty - oy) / max(1.0, speed * 0.8)
    res = None
    for _ in range(4):
        px, py, pz = tx + tvel[0] * flight, ty + tvel[1] * flight, tz
        hd = max(0.1, math.hypot(px - ox, py - oy))
        dz = pz - oz
        v2 = speed * speed
        disc = v2 * v2 - g * (g * hd * hd + 2 * dz * v2)
        if disc < 0:
            return None
        pitch = math.atan((v2 - math.sqrt(disc)) / (g * hd))
        flight = hd / max(1.0, speed * math.cos(pitch))
        res = (math.atan2(py - oy, px - ox), pitch)
    return res


UPGRADE_PRIORITY = {
    "armor": 100, "plating": 96, "laser_damage": 90, "laser_charge": 88, "nanites": 80, "gun_damage": 78,
    "gun_rate": 76, "lightning_efficiency": 70, "gyros": 66, "lightning_range": 64, "mortar_reload": 60,
    "gun_mag": 58, "gun_velocity": 56, "servos": 52, "capacitor": 48, "reactor": 46, "jets": 40,
    "sensors": 44, "mortar_velocity": 38, "mortar_splash": 36, "laser_pierce": 30, "lightning_chain": 34,
}


# ------------------------------------------------------------------------------------ the bot
class Aegis3:
    name = "Aegis-v3"
    WRANK = {"laser": 3, "lightning": 2, "mortar": 1}
    WVAL = {"laser": 140, "lightning": 70, "mortar": 55}

    def __init__(self, hello):
        self.cfg = hello.get("config", {}) or {}
        self.rng = random.Random(int(hello.get("seed", 7)) & 0xFFFFFFFF)
        md = (hello.get("map") or {}).get("dir", "")
        self.map = Arena(md) if md else None
        if self.map is not None and not self.map.ok:
            self.map = None
        mech = self.cfg.get("mech", {})
        self.regen_delay = float(mech.get("regen_delay", 5))
        self.enemies = {}       # id -> dict(pos, vel, t, hp, max_hp, weapons, aim, charging)
        self.dmg_from = {}      # attacker id -> (bearing, t, amount)
        self.last_dmg_t = -99.0
        self.last_dmg_bearing = None
        self.last_hit_t = -99.0
        self.strafe = 1 if self.rng.random() < 0.5 else -1
        self.strafe_t = 0.0
        self.last_pos = None
        self.stuck = 0.0
        self.unstick_dir = None
        self.unstick_until = -1.0
        self.scan_dir = 1
        self.retreating = False
        self.cover_goal = None
        self.cover_t = -99.0
        self.blacklist = {}     # pickup/cube id -> until t (unreachable)
        self.goal_id = None
        self.goal_since = 0.0
        self.goal_best = 1e9
        self.sound_alert = None
        self.focus_id = None
        self.jink_len = 1.0
        self.spot = None
        self.spot_t = -99.0
        self.spot_tid = None
        self.laser_lo, self.laser_hi = 120, 190
        self.t0 = time.perf_counter()
        self.spot_budget_ms = 14.0
        self.crowd_w, self.crowd_r, self.idle_frac = 0.6, 160.0, 0.5
        self.gun_lo, self.gun_hi = 60, 120

    # -------------------------------------------------------------- bookkeeping
    def remember(self, obs, now):
        seen = set()
        for m in obs.get("mechs", ()):
            if m.get("kind") != "mech":
                continue
            seen.add(m["id"])
            self.enemies[m["id"]] = {"pos": list(m["pos"]), "vel": list(m.get("vel", (0, 0, 0))), "t": now,
                                     "hp": m.get("hp", 1400), "max_hp": m.get("max_hp", 1400),
                                     "weapons": m.get("weapons", []), "aim": m.get("aim", [0, 0]),
                                     "charging": m.get("charging", False), "visible": True}
        for k, e in list(self.enemies.items()):
            if k not in seen:
                e["visible"] = False
                if now - e["t"] > 12:
                    del self.enemies[k]
        for ev in obs.get("events", ()):
            ty = ev.get("type")
            if ty == "damaged" and ev.get("by") is not None:
                self.last_dmg_t = now
                if ev.get("bearing") is not None:
                    self.last_dmg_bearing = float(ev["bearing"])
                    self.dmg_from[ev["by"]] = (float(ev["bearing"]), now, ev.get("amount", 0))
            elif ty == "hit":
                self.last_hit_t = now
            elif ty == "kill":
                self.enemies.pop(ev.get("target"), None)
            elif ty == "take_failed":
                self.blacklist[ev.get("id")] = now + 20
        for s in obs.get("sounds", ()):
            if s.get("kind") in ("gun", "laser", "laser_charge", "lightning", "step", "boost", "mortar", "jet") \
                    and s.get("dist") in ("near", "mid") and s.get("age", 0) < 0.5:
                self.sound_alert = (float(s["bearing"]), now, s.get("dist"))

    def pick_upgrade(self, choices):
        best, bv = 0, -1
        for i, c in enumerate(choices):
            v = UPGRADE_PRIORITY.get(c.get("id"), 20)
            if v > bv:
                best, bv = i, v
        return best

    # -------------------------------------------------------------- perception helpers
    def eye(self, me):
        st = me.get("stats", {})
        return (me["pos"][0], me["pos"][1], me["pos"][2] + st.get("eye_height", 8))

    def can_hit(self, me, pos, h=4.5):
        if self.map is None:
            return True
        st = me.get("stats", {})
        o = (me["pos"][0], me["pos"][1], me["pos"][2] + st.get("muzzle_height", 7))
        return self.map.los(o, (pos[0], pos[1], pos[2] + h), 2.5)

    def exposed(self, at, z, threats):
        """Number of threats with a line of sight to a mech standing at `at`."""
        if self.map is None:
            return len(threats)
        n = 0
        for e in threats:
            p = e["pos"]
            if self.map.los((p[0], p[1], p[2] + 8), (at[0], at[1], z + 6.5), 3.0):
                n += 1
        return n

    # -------------------------------------------------------------- weapons
    def weapons(self, me):
        ws = me.get("weapons") or []
        return {w["kind"]: (i, w) for i, w in enumerate(ws) if w}

    def plan_fire(self, me, target, now, tkind="mech"):
        """Returns (aim, fire[2], reload)."""
        ws = self.weapons(me)
        fire = [False, False]
        st = me.get("stats", {})
        pos = me["pos"]
        muzzle = (pos[0], pos[1], pos[2] + st.get("muzzle_height", 7))
        tpos = target["pos"]
        tvel = target.get("vel", (0, 0, 0)) or (0, 0, 0)
        if target.get("grounded_guess", True):
            tvel = (tvel[0], tvel[1], 0.0)
        th = {"mech": 4.5, "npc": 1.8, "crate": 1.2}.get(tkind, 4.0)
        if tkind == "mech":
            th *= target.get("scale", 1.0)
        tc = (tpos[0], tpos[1], tpos[2] + th)
        dist = math.dist(muzzle, tc)
        hd = d2(pos, tpos)
        cur = me.get("aim", [0, 0])
        reload = False
        aim = None
        # gun lead solution
        gun = ws.get("gun")
        gspeed = gun[1]["stats"].get("speed", 252) if gun else 252
        gyaw, gpitch, _ = lead_aim(muzzle, tc, tvel, gspeed)
        laser = ws.get("laser")
        light = ws.get("lightning")
        mortar = ws.get("mortar")
        visible = target.get("visible", True)
        los = visible and self.can_hit(me, tpos, th)
        # choose primary aim
        use_mortar = False
        if mortar and mortar[1]["cooldown"] <= 0.05 and (not los or (not laser and hd > 60)) and 30 < hd < 200 and tkind == "mech":
            ms = mortar[1]["stats"]
            sol = mortar_aim((pos[0], pos[1], pos[2] + st.get("eye_height", 8) + 1.5), (tpos[0], tpos[1], tpos[2] + 1.0),
                             tvel, ms.get("speed", 72), ms.get("gravity", 22))
            if sol is not None:
                aim = [sol[0], sol[1]]
                use_mortar = True
        if aim is None:
            if laser and laser[1].get("charge", 0) > 0:
                lyaw, lpitch, _ = lead_aim(muzzle, tc, tvel, 0)
                # torso needs ~0.1 s to settle, so lead the beam by that much
                aim = [math.atan2(tc[1] + tvel[1] * 0.1 - muzzle[1], tc[0] + tvel[0] * 0.1 - muzzle[0]), lpitch]
            else:
                aim = [gyaw, gpitch]
        aim[1] = clamp(aim[1], -0.69, 1.29)
        yerr = abs(wrap(aim[0] - cur[0]))
        perr = abs(aim[1] - cur[1])
        tol = math.atan2(2.0 if tkind == "mech" else 1.2, max(dist, 1.0))
        if use_mortar:
            i, w = mortar
            fire[i] = yerr < 0.05 and perr < 0.04
            return aim, fire, reload
        if gun and los:
            i, w = gun
            rng = w["stats"].get("range", 230)
            if w.get("reload", 0) <= 0 and w.get("ammo", 1) > 0 and dist < rng - 3 and yerr < tol * 1.6 and perr < tol * 1.6:
                fire[i] = True
            if w.get("ammo", 1) <= 0 and w.get("reload", 0) <= 0:
                reload = True
        if laser and tkind in ("mech", "npc"):
            i, w = laser
            ls = w["stats"]
            full = ls.get("charge", 0.75)
            ch = w.get("charge", 0.0)
            if w.get("cooldown", 0) <= 0.15 or ch > 0:
                in_range = dist < ls.get("range", 280) - 5
                if ch > 0:
                    lyaw = math.atan2(tc[1] + tvel[1] * 0.05 - muzzle[1], tc[0] + tvel[0] * 0.05 - muzzle[0])
                    on = abs(wrap(lyaw - cur[0])) < tol and perr < tol * 1.2
                    if not los or not in_range:
                        # keep the charge while it is safe to; it auto-fires hold_s after full
                        fire[i] = ch < full + ls.get("hold_s", 2) * 0.6
                    elif ch >= full - 0.03 and on:
                        fire[i] = False  # release: full-power shot on target
                    else:
                        fire[i] = True
                elif los and in_range and yerr < 0.5:
                    fire[i] = True
        if light and tkind in ("mech", "npc", "crate"):
            i, w = light
            ls = w["stats"]
            if hd < ls.get("range", 45) - 2 and yerr < math.radians(ls.get("cone_deg", 30)) * 0.4 and me.get("energy", 0) > ls.get("energy", 3) + 6 and los:
                fire[i] = True
        if mortar and not use_mortar and los and tkind == "mech" and 35 < hd < 200:
            pass
        return aim, fire, reload

    # -------------------------------------------------------------- movement helpers
    def steer(self, me, mv, zone):
        """Bend a desired move around walls, the map edge and (if inside) the zone edge."""
        if self.map is None or (mv[0] == 0 and mv[1] == 0):
            return mv
        x, y, z = me["pos"]
        base = math.atan2(mv[1], mv[0])
        mag = min(1.0, math.hypot(*mv))
        zc, zr = zone.get("center", (0, 0)), float(zone.get("radius", 1e4))
        inside = d2((x, y), zc) < zr - 4

        def ok(a):
            cx, cy = math.cos(a), math.sin(a)
            for dd in (3.0, 6.5, 11.0):
                px, py = x + cx * dd, y + cy * dd
                if self.map.wall(px, py, z):
                    return False
                if inside and d2((px, py), zc) > zr - 1:
                    return False
            return True

        if ok(base):
            return mv
        for k in (0.35, 0.7, 1.05, 1.4, 1.8, 2.3, 2.8):
            for s in (self.strafe, -self.strafe):
                a = base + s * k
                if ok(a):
                    return [math.cos(a) * mag, math.sin(a) * mag]
        return mv

    def find_cover(self, me, threats, zone, now):
        if self.map is None or not threats:
            return None
        if self.cover_goal and now - self.cover_t < 1.5:
            return self.cover_goal
        x, y, z = me["pos"]
        zc, zr = zone.get("center", (0, 0)), float(zone.get("radius", 1e4))
        best, bs = None, 1e9
        tx = sum(e["pos"][0] for e in threats) / len(threats)
        ty = sum(e["pos"][1] for e in threats) / len(threats)
        away = math.atan2(y - ty, x - tx)
        for r in (8, 15, 24, 34):
            for k in range(10):
                if self.over_budget(self.spot_budget_ms):
                    break
                a = away + (k - 4.5) * 0.45
                px, py = x + r * math.cos(a), y + r * math.sin(a)
                if self.map.wall(px, py, z) or d2((px, py), zc) > zr - 6:
                    continue
                gz = self.map.ground(px, py)
                if abs(gz - z) > 3:
                    continue
                n = self.exposed((px, py), gz, threats)
                # path should not be blocked outright
                score = n * 100 + r * 0.8 - 15 * math.cos(a - away)
                if score < bs:
                    best, bs = (px, py), score
            if best is not None and bs < 50:
                break
        self.cover_goal = best
        self.cover_t = now
        return best

    def fight_spot(self, me, target, vis, zone, now, pref):
        """Pick a nearby standing spot: line of sight to the focus target, hidden from the others."""
        if self.map is None:
            return None
        if self.spot is not None and now - self.spot_t < 0.5 and self.spot_tid == target["id"]:
            return self.spot
        x, y, z = me["pos"]
        zc, zr = zone.get("center", (0, 0)), float(zone.get("radius", 1e4))
        tp = target["pos"]
        others = sorted((e for e in vis if e["id"] != target["id"]), key=lambda e: d2(e["pos"], me["pos"]))[:3]
        tb = math.atan2(tp[1] - y, tp[0] - x)
        best, bs = None, 1e9
        cands = [(x, y)]
        for r in (7.0, 14.0, 22.0):
            for k in range(8):
                a = tb + PI / 2 + k * PI / 4
                cands.append((x + r * math.cos(a), y + r * math.sin(a)))
        for c in cands:
            if self.over_budget(self.spot_budget_ms):
                break
            if c is not cands[0] and self.map.wall(c[0], c[1], z):
                continue
            if d2(c, zc) > zr - 5:
                continue
            gz = self.map.ground(c[0], c[1])
            if abs(gz - z) > 2.5 and c is not cands[0]:
                continue
            gz = z if c is cands[0] else gz
            eye = (c[0], c[1], gz + 7.0)
            dt = d2(c, tp)
            sc = abs(dt - pref) * 0.35 + d2(c, (x, y)) * 0.6
            if not self.map.los(eye, (tp[0], tp[1], tp[2] + 4.5), 3.0):
                sc += 70
            for e in others:
                ep = e["pos"]
                de = d2(c, ep)
                if de < 90:
                    sc += (90 - de) * 0.5
                if self.map.los((ep[0], ep[1], ep[2] + 8), (c[0], c[1], gz + 6.5), 3.0):
                    sc += 55
            if sc < bs:
                best, bs = c, sc
        self.spot, self.spot_t, self.spot_tid = best, now, target["id"]
        return best

    def dodge(self, obs, me):
        x, y, z = me["pos"]
        vx0, vy0 = me.get("vel", (0, 0, 0))[:2]
        ex = ey = 0.0
        danger = 0.0
        g = self.cfg.get("weapons", {}).get("mortar", {}).get("gravity", 22)
        for p in obs.get("projectiles", ()):
            pp, pv = p["pos"], p["vel"]
            if p.get("kind") == "mortar":
                # predict landing point and step out of the splash
                vz = pv[2]
                hz = pp[2] - (z + 1)
                disc = vz * vz + 2 * g * hz
                if disc < 0:
                    continue
                tl = (vz + math.sqrt(disc)) / g
                lx, ly = pp[0] + pv[0] * tl, pp[1] + pv[1] * tl
                dd = math.hypot(lx - x, ly - y)
                if dd < 13 and tl < 3.0:
                    ux, uy = unit(x - lx, y - ly)
                    if ux == uy == 0:
                        ux, uy = unit(-pv[1], pv[0])
                    w = (13 - dd) / 13 * 2.5
                    ex += ux * w
                    ey += uy * w
                    danger += w
                continue
            rx, ry = pp[0] - x, pp[1] - y
            rvx, rvy = pv[0] - vx0, pv[1] - vy0
            s2 = rvx * rvx + rvy * rvy
            if s2 < 1:
                continue
            tau = -(rx * rvx + ry * rvy) / s2
            if tau < 0 or tau > 0.6:
                continue
            cx, cy = rx + rvx * tau, ry + rvy * tau
            miss = math.hypot(cx, cy)
            if miss > 4.5:
                continue
            ux, uy = unit(-rvy, rvx)
            if ux * cx + uy * cy > 0:
                ux, uy = -ux, -uy
            w = (4.5 - miss) / 4.5 * 0.6
            ex += ux * w
            ey += uy * w
            danger += w
        return ex, ey, danger

    # -------------------------------------------------------------- main decision
    def over_budget(self, ms):
        return (time.perf_counter() - self.t0) * 1000.0 > ms

    def act(self, obs):
        self.t0 = time.perf_counter()
        me = obs["self"]
        now = float(obs.get("t", obs.get("tick", 0) / 20.0))
        pos = me["pos"]
        x, y, z = pos
        zone = obs.get("zone") or {}
        act = {"tick": obs["tick"]}
        if obs.get("offer"):
            act["pick"] = self.pick_upgrade(obs["offer"].get("choices", []))
        self.remember(obs, now)
        hp, mhp = float(me.get("hp", 1)), float(me.get("max_hp", 1400))
        hpf = hp / max(1.0, mhp)
        energy = float(me.get("energy", 0))
        ws = self.weapons(me)
        has_laser = "laser" in ws
        alive = int(obs.get("alive", 16))

        # ---------------- threats
        threats = [e for e in self.enemies.values() if now - e["t"] < 6]
        vis = [dict(e, id=k) for k, e in self.enemies.items() if e["visible"]]
        for e in vis:
            e["scale"] = 1 + 0.05 * 0  # size from cubes is unknown for others; hit box margin covers it
        near_threats = [e for e in threats if d2(e["pos"], pos) < 230]
        target = None
        if vis:
            def tscore(e):
                d = d2(e["pos"], pos)
                s = d * 0.6 + e["hp"] * 0.12                      # focus the weakest, nearest mech
                if e.get("charging"):
                    s -= 25
                if not self.can_hit(me, e["pos"]):
                    s += 200
                if e["id"] == self.focus_id:
                    s -= 90                                       # stick with it until it dies
                return s
            target = min(vis, key=tscore)
            self.focus_id = target["id"]

        # ---------------- zone geometry
        zc, zr = zone.get("center", (0, 0)), float(zone.get("radius", 1e4))
        nc, nr = zone.get("next_center", zc), float(zone.get("next_radius", zr))
        t_shrink = float(zone.get("shrink_start", 1e9)) - now
        cur_margin = zr - d2(pos, zc)
        next_margin = nr - d2(pos, nc)
        travel = max(0.0, -next_margin) / 11.0
        zone_goal = None
        zone_urg = 0.0
        if cur_margin < 6:
            zone_goal, zone_urg = zc, 3.0
        elif next_margin < 8 and (t_shrink < travel + 25 or now >= zone.get("shrink_start", 1e9)):
            zone_goal, zone_urg = nc, 2.0 if cur_margin > 25 else 3.0
        elif cur_margin < 25:
            zone_goal, zone_urg = zc, 1.0

        # ---------------- fighting state
        hurt_recent = now - self.last_dmg_t < 1.5
        n_vis = len(vis)
        n_close = sum(1 for e in vis if d2(e["pos"], pos) < 160)
        if self.retreating:
            if hpf > 0.82 or (not threats and hpf > 0.6) or (target and target["hp"] < 150 and hpf > 0.25):
                self.retreating = False
        else:
            if hpf < 0.38 or (hpf < 0.55 and n_close >= 2) or (hpf < 0.5 and target and target["hp"] > hp * 1.6):
                self.retreating = True

        # ---------------- aim and fire
        aim = None
        fire = [False, False]
        reload = False
        tgt_kind = None
        farm = None
        if target is not None:
            aim, fire, reload = self.plan_fire(me, target, now, "mech")
            tgt_kind = "mech"
        else:
            # farm drones and crates for cubes when nothing hunts us
            cands = []
            for m in obs.get("mechs", ()):
                if m.get("kind") == "npc":
                    dd = d2(m["pos"], pos)
                    if dd < 150 and (hpf > 0.6 or dd > 60):
                        cands.append((dd * 0.8, m, "npc"))
            for c in obs.get("crates", ()):
                dd = d2(c["pos"], pos)
                if dd < 120 and abs(c["pos"][2] - z) < 25:
                    cands.append((dd, c, "crate"))
            cands = [c for c in cands if self.can_hit(me, c[1]["pos"], 1.5)]
            if cands and not (threats and now - max(e["t"] for e in threats) < 2):
                _, farm, tgt_kind = min(cands, key=lambda c: c[0])
                aim, fire, reload = self.plan_fire(me, farm, now, tgt_kind)
        if aim is None:
            # look where danger comes from: recent damage, a remembered hunter, a sound, else sweep
            look = None
            if self.last_dmg_bearing is not None and now - self.last_dmg_t < 4:
                look = self.last_dmg_bearing
            elif threats:
                e = min(threats, key=lambda e: d2(e["pos"], pos))
                look = math.atan2(e["pos"][1] - y, e["pos"][0] - x)
            elif self.sound_alert and now - self.sound_alert[1] < 3:
                look = self.sound_alert[0]
            if look is None:
                cur = me.get("aim", [0, 0])[0]
                look = cur + 0.35 * self.scan_dir
                if self.rng.random() < 0.01:
                    self.scan_dir = -self.scan_dir
            aim = [look, 0.0]
            # keep a laser charge going toward a likely threat but never waste it
            if "laser" in ws and ws["laser"][1].get("charge", 0) > 0:
                fire[ws["laser"][0]] = True
            gun = ws.get("gun")
            if gun and gun[1].get("ammo", 12) < gun[1]["stats"].get("mag", 12) * 0.5 and gun[1].get("reload", 0) <= 0:
                reload = True
        # don't let a damaging shooter sit behind us while we farm or chase
        if target is None and self.last_dmg_bearing is not None and now - self.last_dmg_t < 1.0:
            aim = [self.last_dmg_bearing, 0.0]
            fire = [False, False]
            if "laser" in ws and ws["laser"][1].get("charge", 0) > 0:
                fire[ws["laser"][0]] = True
        act["aim"] = [wrap(aim[0]), aim[1]]
        act["fire"] = fire
        if reload:
            act["reload"] = True

        # ---------------- movement
        mvx = mvy = 0.0
        boost = False
        mode = "idle"
        loot_goal = None
        take = None
        # standing on a useful pickup? take it
        for p in obs.get("pickups", ()):
            if d2(p["pos"], pos) <= 3.7 and abs(p["pos"][2] - z) < 3.8:
                k = p.get("kind")
                if k in ws:
                    continue
                slots = me.get("weapons") or [None, None]
                empty = next((i for i, w in enumerate(slots) if w is None), None)
                val = self.WRANK.get(k, 0)
                if empty is not None:
                    take = (p["id"], None)
                    break
                # swap the worse special for a better one (never drop the gun, never a charging laser)
                worst = None
                for i, w in enumerate(slots):
                    if w and w["kind"] != "gun" and not (w["kind"] == "laser" and w.get("charge", 0) > 0):
                        wv = self.WRANK.get(w["kind"], 0)
                        if wv < val and (worst is None or wv < worst[1]):
                            worst = (i, wv)
                if worst is not None:
                    take = (p["id"], worst[0])
                    break
        if take:
            act["take"] = take[0]
            if take[1] is not None:
                act["drop"] = take[1]

        if self.retreating and (threats or hurt_recent):
            mode = "retreat"
            src = threats if threats else []
            if not src and self.last_dmg_bearing is not None:
                b = self.last_dmg_bearing
                src = [{"pos": [x + 80 * math.cos(b), y + 80 * math.sin(b), z]}]
            cov = self.find_cover(me, src, zone, now)
            if cov is not None and d2(cov, pos) > 2.5:
                mvx, mvy = unit(cov[0] - x, cov[1] - y)
                boost = energy > 25 and hurt_recent
            elif cov is None and src:
                cx = sum(e["pos"][0] for e in src) / len(src)
                cy = sum(e["pos"][1] for e in src) / len(src)
                ax, ay = unit(x - cx, y - cy)
                mvx, mvy = ax * 0.8 - ay * self.strafe * 0.6, ay * 0.8 + ax * self.strafe * 0.6
                boost = energy > 30 and hurt_recent
        elif target is not None:
            mode = "fight"
            d = d2(target["pos"], pos)
            tx, ty = target["pos"][0], target["pos"][1]
            ux, uy = unit(tx - x, ty - y)
            # irregular jinks spoil the enemy's lead on our velocity
            if now - self.strafe_t > self.jink_len:
                self.strafe = -self.strafe
                self.strafe_t = now
                self.jink_len = self.rng.uniform(0.5, 1.6)
            # preferred distance: long with a laser (hitscan; their bullets fly longer), mid with gun, close with lightning
            if "lightning" in ws and (not has_laser or self.WRANK["lightning"] > self.WRANK["laser"]):
                lo, hi = 12, 32
            elif has_laser:
                lo, hi = self.laser_lo, self.laser_hi
            else:
                lo, hi = self.gun_lo, self.gun_hi
            spot = self.fight_spot(me, target, vis, zone, now, (lo + hi) / 2)
            if spot is not None and d2(spot, pos) > 2.0:
                gx, gy = unit(spot[0] - x, spot[1] - y)
                sx, sy = -uy * self.strafe, ux * self.strafe
                mvx, mvy = gx + sx * 0.45, gy + sy * 0.45
            else:
                radial = 1.0 if d > hi else -1.0 if d < lo else 0.0
                sx, sy = -uy * self.strafe, ux * self.strafe
                mvx = ux * radial + sx * 0.9
                mvy = uy * radial + sy * 0.9
            charging = "laser" in ws and ws["laser"][1].get("charge", 0) > 0
            boost = energy > 55 and d > hi + 50 and not charging
        elif near_threats and now - max(e["t"] for e in near_threats) < 5 and not obs.get("drops"):
            # a hunter just slipped out of view: hold, face it, keep the laser primed, let it come to us
            mode = "hold"
            e = min(near_threats, key=lambda e: d2(e["pos"], pos))
            ex_, ey_ = e["pos"][0], e["pos"][1]
            de = d2(e["pos"], pos)
            ux, uy = unit(ex_ - x, ey_ - y)
            if de > 170:
                mvx, mvy = ux * 0.6, uy * 0.6
            elif de < 50:
                mvx, mvy = -ux * 0.6, -uy * 0.6
            act["aim"] = [math.atan2(ey_ - y, ex_ - x), 0.0]
            if "laser" in ws and ws["laser"][1].get("cooldown", 0) <= 0:
                f = list(act["fire"])
                f[ws["laser"][0]] = ws["laser"][1].get("charge", 0) < ws["laser"][1]["stats"].get("charge", 0.75) + 1.0
                act["fire"] = f
        else:
            # looting / farming
            mode = "loot"
            best = None
            want_laser = not has_laser
            for p in obs.get("pickups", ()):
                if self.blacklist.get(p["id"], -1) > now:
                    continue
                k = p.get("kind")
                if k in ws:
                    continue
                if abs(p["pos"][2] - z) > 6:
                    continue
                dd = d2(p["pos"], pos)
                if d2(p["pos"], zc) > zr - 5:
                    continue
                slots = me.get("weapons") or []
                full = all(slots) and len(slots) >= 2
                special = next((w["kind"] for w in slots if w and w["kind"] != "gun"), None)
                val = self.WVAL.get(k, 0)
                if full:
                    if special is not None and self.WRANK.get(k, 0) <= self.WRANK.get(special, 0):
                        continue
                if want_laser and k == "laser":
                    val += 60
                sc = val - dd * 0.9
                if dd < 160 and (best is None or sc > best[0]):
                    best = (sc, p["pos"], p["id"])
            for c in obs.get("cubes", ()):
                if self.blacklist.get(c["id"], -1) > now:
                    continue
                if abs(c["pos"][2] - z) > 4:
                    continue
                dd = d2(c["pos"], pos)
                if d2(c["pos"], zc) > zr - 3:
                    continue
                sc = 90 - dd * 0.9
                if dd < 130 and (best is None or sc > best[0]):
                    best = (sc, c["pos"], c["id"])
            for dp in obs.get("drops", ()):
                if dp.get("holder") not in (None, me.get("id")):
                    continue
                dd = d2(dp["pos"], pos)
                if dd > 260 or dp.get("lands_in", 99) > dd / 11 + 6:
                    continue
                sc = 160 + (60 if hpf < 0.8 else 0) - dd * 0.6
                if best is None or sc > best[0]:
                    best = (sc, dp["pos"], dp["id"])
            if farm is not None and tgt_kind == "npc":
                dd = d2(farm["pos"], pos)
                if dd < 60:
                    # stay outside drone gun range while shooting
                    ux, uy = unit(x - farm["pos"][0], y - farm["pos"][1])
                    mvx, mvy = ux, uy
            if best is not None and best[0] > -40 and not (mvx or mvy):
                gid = best[2]
                if gid != self.goal_id:
                    self.goal_id, self.goal_since, self.goal_best = gid, now, d2(best[1], pos)
                else:
                    dcur = d2(best[1], pos)
                    if dcur < self.goal_best - 3:
                        self.goal_best, self.goal_since = dcur, now
                    elif now - self.goal_since > 8:
                        self.blacklist[gid] = now + 30
                loot_goal = best[1]
                mvx, mvy = unit(loot_goal[0] - x, loot_goal[1] - y)
                if d2(loot_goal, pos) < 2.5 and gid >= 20000:
                    mvx = mvy = 0.0
                boost = energy > 70 and d2(loot_goal, pos) > 40 and gid < 20000 and False
            elif not (mvx or mvy) and farm is None:
                # drift toward the next zone's inner area; keep moving slowly (silent under 5 m/s)
                gx, gy = nc
                if d2(pos, nc) > nr * self.idle_frac:
                    mvx, mvy = unit(gx - x, gy - y)
                    mvx, mvy = mvx * 0.6, mvy * 0.6
            if farm is not None and tgt_kind == "crate" and not (mvx or mvy):
                dd = d2(farm["pos"], pos)
                if dd > 60:
                    mvx, mvy = unit(farm["pos"][0] - x, farm["pos"][1] - y)

        # stay away from crowds: every remembered mech we are not fighting pushes us away
        if self.crowd_w > 0 and mode in ("loot", "idle", "fight", "retreat"):
            cx = cy = 0.0
            for k, e in self.enemies.items():
                if target is not None and k == target["id"]:
                    continue
                if now - e["t"] > 8:
                    continue
                de = d2(e["pos"], pos)
                if de < self.crowd_r:
                    w = (1 - de / self.crowd_r) ** 2
                    ux, uy = unit(x - e["pos"][0], y - e["pos"][1])
                    cx += ux * w
                    cy += uy * w
            mvx += cx * self.crowd_w
            mvy += cy * self.crowd_w

        # zone has priority over everything except a close dodge
        if zone_goal is not None and zone_urg > 0:
            zx, zy = unit(zone_goal[0] - x, zone_goal[1] - y)
            if zone_urg >= 3:
                mvx, mvy = zx, zy
                boost = energy > 15 and cur_margin < 0
            elif zone_urg >= 2:
                mvx, mvy = mvx * 0.3 + zx, mvy * 0.3 + zy
                boost = boost or (energy > 40 and next_margin < -40 and t_shrink < travel + 10)
            else:
                dot = mvx * zx + mvy * zy
                if dot < 0:
                    mvx, mvy = mvx - dot * zx, mvy - dot * zy
                mvx, mvy = mvx + 0.4 * zx, mvy + 0.4 * zy
        if not me.get("in_zone", True):
            boost = energy > 10

        # projectile dodge
        ex, ey, danger = self.dodge(obs, me)
        if danger > 0.05:
            mvx, mvy = mvx + ex * 1.6, mvy + ey * 1.6
            if danger > 1.0 and energy > 30 and me.get("grounded", True):
                boost = True

        # stuck handling: jump/jet and sidestep
        if self.last_pos is not None and obs["tick"] % 10 == 0:
            moved = d2(self.last_pos, pos)
            want = math.hypot(mvx, mvy) > 0.3
            self.stuck = self.stuck + 1 if (want and moved < 2.0) else max(0, self.stuck - 1)
            self.last_pos = (x, y)
        elif self.last_pos is None:
            self.last_pos = (x, y)
        jump = False
        if self.stuck >= 2:
            if self.unstick_until < now:
                a = math.atan2(mvy, mvx) + self.strafe * self.rng.uniform(0.9, 1.8)
                self.unstick_dir = (math.cos(a), math.sin(a))
                self.unstick_until = now + 1.2
                self.stuck = 0
            jump = me.get("grounded", True) and energy > 25
        if self.unstick_until > now and self.unstick_dir:
            mvx, mvy = self.unstick_dir

        mv = [mvx, mvy]
        l = math.hypot(*mv)
        if l > 1:
            mv = [mv[0] / l, mv[1] / l]
        if l > 0.05:
            mv = self.steer(me, mv, zone)
        if not me.get("grounded", True) and self.unstick_until > now and energy > 15:
            boost = True
        act["move"] = [round(mv[0], 4), round(mv[1], 4)]
        act["boost"] = bool(boost)
        if jump:
            act["jump"] = True
        mf.draw.label("%s hp%.0f e%.0f %s" % (mode, hp, energy, "R" if self.retreating else ""))
        return act


BOT_CLASS = Aegis3

if __name__ == "__main__":
    mf.run(Aegis3)
