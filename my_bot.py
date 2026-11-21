"""Aegis v1: defensive survival bot for Builders Mech Royale / mechfight/1.

Standard-library strategy built on the organizer's mechfight.py transport wrapper.
The bot deliberately favors placement and survival over risky eliminations.
"""
import math
import random
import os
import json
from navigation import GroundRoutes
import mechfight as mf

PI = math.pi
TAU = 2 * PI
EPS = 1e-8


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def length(x, y):
    return math.hypot(x, y)


def unit(x, y):
    d = math.hypot(x, y)
    return (x / d, y / d) if d > EPS else (0.0, 0.0)


def yaw_diff(a, b):
    return (a - b + PI) % TAU - PI


def dist2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def within_zone(p, center, radius, margin=0.0):
    return dist2(p, center) < radius - margin


def aim_at(origin, target, speed=None, velocity=None):
    """Predictive yaw/pitch for straight projectiles or hitscan."""
    ox, oy, oz = origin
    tx, ty, tz = target
    vx, vy, vz = velocity or (0.0, 0.0, 0.0)
    dx, dy, dz = tx - ox, ty - oy, tz - oz
    t = 0.0
    if speed and speed > 1:
        aa = vx * vx + vy * vy + vz * vz - speed * speed
        bb = 2 * (dx * vx + dy * vy + dz * vz)
        cc = dx * dx + dy * dy + dz * dz
        if abs(aa) > EPS:
            disc = bb * bb - 4 * aa * cc
            if disc >= 0:
                sq = math.sqrt(disc)
                candidates = [v for v in ((-bb - sq) / (2 * aa), (-bb + sq) / (2 * aa)) if v > 0]
                if candidates:
                    t = min(candidates)
        if t <= 0:
            t = math.sqrt(cc) / speed
        t = min(t, 2.5)
    dx += vx * t
    dy += vy * t
    dz += vz * t
    return math.atan2(dy, dx), math.atan2(dz, max(0.01, math.hypot(dx, dy)))


def mortar_aim(origin, target, velocity, speed, gravity):
    """Iteratively solve low ballistic arc against a moving target."""
    ox, oy, oz = origin
    tx, ty, tz = target
    vx, vy, vz = velocity
    flight = max(0.05, math.hypot(tx - ox, ty - oy) / max(speed * 0.77, 1))
    pitch = 0.3
    yaw = math.atan2(ty - oy, tx - ox)
    for _ in range(4):
        xx, yy = tx + vx * flight - ox, ty + vy * flight - oy
        zz = tz + vz * flight - oz
        horiz = max(0.01, math.hypot(xx, yy))
        v2 = speed * speed
        delta = v2 * v2 - gravity * (gravity * horiz * horiz + 2 * zz * v2)
        if delta < 0:
            return None
        # lower trajectory; shallower aim makes hit time and exposure smaller
        pitch = math.atan((v2 - math.sqrt(delta)) / max(gravity * horiz, 0.01))
        yaw = math.atan2(yy, xx)
        flight = horiz / max(1.0, speed * math.cos(pitch))
        if flight > 4.0:
            return None
    return yaw, pitch


class Bot:
    name = "Aegis-Survivor-v1"

    def __init__(self, hello: mf.Hello):
        self.config = hello.get("config", {})
        self.rng = random.Random(int(hello.get("seed", 17)))
        self.pickup_radius = self.config.get("loot", {}).get("pickup_radius", 4)
        self.last_pos = None
        self.last_move = (0.0, 0.0)
        self.last_check = -999
        self.stuck = 0
        self.last_jump = -999
        self.preferred_side = 1 if self.rng.random() < 0.5 else -1
        self.last_threat = None
        self.last_threat_t = -999.0
        self.last_damage_yaw = None
        self.last_damage_t = -999.0

    def pick_upgrade(self, choices):
        """Prefer upgrades that help us escape, survive, and regenerate."""
        def utility(choice):
            key = (str(choice.get("id", "")) + " " + str(choice.get("name", "")) +
                   " " + str(choice.get("desc", ""))).lower()
            scores = [
                ("armor", 11), ("damage taken", 11), ("regen", 10),
                ("heal", 9), ("walk", 9), ("speed", 8.5), ("boost", 8.5),
                ("energy", 8), ("health", 7.5), ("max hp", 7.5),
                ("sprint", 7.5), ("turn", 6), ("laser", 6.5),
                ("gun_rate", 5.5), ("gun", 5), ("lightning", 5),
                ("mortar", 4.5), ("damage", 5), ("sensor", 3),
            ]
            return max((score for term, score in scores if term in key), default=2.0)
        return max(range(len(choices)), key=lambda i: utility(choices[i])) if choices else 0

    def projectile_evasion(self, obs, me):
        """Steer perpendicular to incoming shots that approach within ~0.8 s."""
        sx, sy, sz = me["pos"]
        mvx, mvy = me.get("vel", (0, 0, 0))[:2]
        evx = evy = danger = 0.0
        for p in obs.get("projectiles", []):
            pp, pv = p.get("pos", (0, 0, 0)), p.get("vel", (0, 0, 0))
            if len(pp) < 3 or len(pv) < 3:
                continue
            rx, ry = pp[0] - sx, pp[1] - sy
            vx, vy = pv[0] - mvx, pv[1] - mvy
            speed_sq = vx * vx + vy * vy
            if speed_sq < 1:
                continue
            tau = clamp(-(rx * vx + ry * vy) / speed_sq, 0, 0.85)
            nearx, neary = rx + vx * tau, ry + vy * tau
            d = math.hypot(nearx, neary)
            z = pp[2] + pv[2] * tau - (11.0 * tau * tau if p.get("kind") == "mortar" else 0)
            if d > 18 or abs(z - (sz + 4.5)) > 15:
                continue
            severity = (18 - d) / 18 * (0.9 - tau) / 0.9
            if p.get("kind") == "mortar":
                severity *= 1.7
            ux, uy = unit(-vy, vx)
            # Dodge away from the projected closest approach to choose side.
            if ux * nearx + uy * neary > 0:
                ux, uy = -ux, -uy
            evx += ux * severity
            evy += uy * severity
            danger += severity
        return evx, evy, danger

    def zone_navigation(self, me, zone, now):
        """Move well into upcoming safe area; do not get pushed out by kiting."""
        x, y = me["pos"][:2]
        cur_c = zone.get("center", [0, 0])
        nxt_c = zone.get("next_center", cur_c)
        cur_r = max(1.0, float(zone.get("radius", 500)))
        nxt_r = max(1.0, float(zone.get("next_radius", cur_r)))
        deadline = float(zone.get("shrink_start", 1e6)) - now
        shrink_end = float(zone.get("shrink_end", now + 30))
        # The circle shown as "center" and radius is the instantaneous safe area.
        cur_margin = cur_r - dist2((x, y), cur_c)
        next_margin = nxt_r - dist2((x, y), nxt_c)
        urgently_outside = cur_margin < 18
        # Move into next zone increasingly early as the shrink approaches.
        early = deadline < 65 or now >= zone.get("shrink_start", 1e6)
        if urgently_outside:
            gx, gy = cur_c[0], cur_c[1]
            urgency = 6.0 if cur_margin < 0 else 3.0
        elif early and next_margin < max(22, nxt_r * 0.20):
            gx, gy = nxt_c[0], nxt_c[1]
            urgency = 2.5 if next_margin > 0 else 4.5
        elif cur_margin < 55:
            gx, gy = cur_c[0], cur_c[1]
            urgency = 2.0
        else:
            # Stay in the current safe area until the next phase matters.
            gx, gy, urgency = x, y, 0.0
        if early and nxt_r < 75:
            # In small circles a fixed 22 m buffer can exceed the entire radius.
            safe_depth = max(1.5, nxt_r * 0.45)
            if next_margin >= safe_depth and cur_margin > 2:
                urgency = 0.0
            else:
                gx, gy = nxt_c
                urgency = 6.0
        # Any route into the next circle also leads back inside the current one.
        if urgency and self.map.routes and dist2((x, y), nxt_c) > 12:
            gx, gy = self.map.routes.waypoint(me["pos"], nxt_c, now)
        ux, uy = unit(gx - x, gy - y)
        return ux * urgency, uy * urgency, urgency, cur_margin, next_margin

    def choose_loot(self, obs, me, enemies, zone):
        sx, sy, sz = me["pos"]
        nearest_enemy = min((dist2(m["pos"], me["pos"]) for m in enemies), default=1e9)
        weapons = me.get("weapons", [])
        owned = [w.get("kind") for w in weapons if w]
        priorities = {"laser": 10, "lightning": 8, "mortar": 7, "gun": 1}
        pickups = sorted(obs.get("pickups", []), key=lambda p: dist2(p["pos"], me["pos"]))
        goal = None
        action = {}
        for p in pickups:
            kind = p.get("kind")
            if kind in owned or kind not in priorities:
                continue
            dd = dist2(p["pos"], me["pos"])
            if abs(p["pos"][2] - sz) > 4 or dd > 80:
                continue
            if not within_zone(p["pos"], zone.get("center", [0, 0]), max(10, zone.get("radius", 500))):
                continue
            if nearest_enemy < 50 and dd > 8:
                continue
            empty = next((i for i, w in enumerate(weapons) if w is None), None)
            replace = None
            if empty is None and weapons:
                # Never drop a charging laser. Avoid sacrificing a good special weapon.
                slot = min(range(len(weapons)), key=lambda i: priorities.get((weapons[i] or {}).get("kind"), 0))
                old = weapons[slot] or {}
                if priorities[kind] <= priorities.get(old.get("kind"), 0) + 2 or old.get("charge", 0) > 0:
                    continue
                replace = slot
            goal = p["pos"][:2]
            if dd <= self.pickup_radius - 0.35:
                action["take"] = p["id"]
                if replace is not None:
                    action["drop"] = replace
            break
        if goal is None and nearest_enemy > 75:
            cubes = [c for c in obs.get("cubes", [])
                     if dist2(c["pos"], me["pos"]) <= 75
                     and abs(c["pos"][2] - sz) < 5
                     and within_zone(c["pos"], zone.get("center", [0, 0]), max(12, zone.get("radius", 500)))]
            if cubes:
                goal = min(cubes, key=lambda c: dist2(c["pos"], me["pos"]))["pos"][:2]
        # Supply drops can heal to full, but are risky in an active fight.
        if nearest_enemy > 115 and me.get("hp", 0) < 0.8 * me.get("max_hp", 1):
            for p in obs.get("drops", []):
                if p.get("lands_in", 999) < 8 and dist2(p["pos"], me["pos"]) < 110:
                    if within_zone(p["pos"], zone.get("center", [0, 0]), zone.get("radius", 500), 12):
                        goal = p["pos"][:2]
                        break
        return goal, action

    def choose_target(self, obs, me):
        pos = me["pos"]
        enemies = [m for m in obs.get("mechs", []) if m.get("kind") == "mech"]
        if enemies:
            def threat(m):
                dd = max(1, dist2(m["pos"], pos))
                # Nearby laser chargers and low-hp threats take priority.
                charging = 65 if m.get("charging", False) else 0
                low_hp = 30 if m.get("hp", 1e4) < 250 else 0
                return dd - charging - low_hp
            return min(enemies, key=threat), enemies
        # Farm cubes from crates when no competing mechs are visible.
        crates = [c for c in obs.get("crates", []) if dist2(c["pos"], pos) < 135]
        if crates:
            c = min(crates, key=lambda c: dist2(c["pos"], pos))
            return {"kind": "crate", "pos": c["pos"], "vel": [0, 0, 0]}, enemies
        return None, enemies

    def combat(self, obs, me, target):
        sx, sy, sz = me["pos"]
        weapons = me.get("weapons", [])
        fire = [False, False]
        if not target:
            # Slowly scan for targets when not engaged, without wasting ammo.
            desired = me.get("aim", [0, 0])
            return [desired[0] + 0.24, 0.0], fire, False
        kind = target.get("kind", "mech")
        tx, ty, tz = target["pos"]
        v = target.get("vel", [0, 0, 0])
        d = math.hypot(tx - sx, ty - sy)
        muzzle = sz + me.get("stats", {}).get("muzzle_height", 7.0)
        target_z = tz + (4.5 * (1 + self.config.get("cubes", {}).get("size_per_cube", 0.05) * target.get("cubes", 0)) if kind == "mech" else 1.75)
        origin = (sx, sy, muzzle)
        dest = (tx, ty, target_z)
        actual_yaw = me.get("aim", [0.0, 0.0])[0]
        # Favor usable special weapons at short range; gun otherwise.
        special_slot = next((i for i, w in enumerate(weapons) if w and w.get("kind") == "laser" and w.get("charge", 0) > 0), None)
        if special_slot is None:
            special_slot = next((i for i, w in enumerate(weapons) if w and w.get("kind") == "lightning" and d < 40), None)
        if special_slot is None:
            special_slot = next((i for i, w in enumerate(weapons) if w and w.get("kind") == "laser" and w.get("cooldown", 0) <= 0.1 and d < 275), None)
        if special_slot is None:
            special_slot = next((i for i, w in enumerate(weapons) if w and w.get("kind") == "mortar" and w.get("cooldown", 0) <= 0.1 and 30 < d < 190), None)
        slot = special_slot if special_slot is not None else 0
        selected = weapons[slot] if slot < len(weapons) and weapons[slot] else None
        weapon_kind = selected.get("kind", "gun") if selected else "gun"
        stats = selected.get("stats", {}) if selected else {}
        speed = stats.get("speed", 252.0 if weapon_kind == "gun" else 72.0)
        # Laser is hitscan, but aim slightly ahead because the torso takes time to turn.
        if weapon_kind == "laser":
            aim = aim_at(origin, (tx + v[0] * 0.10, ty + v[1] * 0.10, target_z + v[2] * 0.10))
        elif weapon_kind == "lightning":
            aim = aim_at(origin, dest)
        elif weapon_kind == "mortar":
            aim = mortar_aim((sx, sy, sz + me.get("stats", {}).get("height", 9) + 0.5), dest, v, speed, stats.get("gravity", 22.0))
            if aim is None:
                weapon_kind = "gun"
                slot = next((i for i, w in enumerate(weapons) if w and w.get("kind") == "gun"), 0)
                selected = weapons[slot] if slot < len(weapons) else None
                aim = aim_at(origin, dest, 252.0, v)
        else:
            aim = aim_at(origin, dest, speed, v)
        angle = abs(yaw_diff(aim[0], actual_yaw))
        in_arc = angle < (0.11 if weapon_kind == "laser" else 0.20)
        reload = False
        for i, w in enumerate(weapons[:2]):
            if not w:
                continue
            k = w.get("kind")
            if k == "gun":
                if w.get("ammo", 1) <= 0 and w.get("reload", 0) <= 0:
                    reload = True
                # Don't spend ammo if target is beyond its range or aim is far off.
                gun_aim = aim_at(origin, dest, w.get("stats", {}).get("speed", 252), v)
                # Both triggers share one torso. Save bullets during steep mortar lobs.
                compatible = abs(aim[1] - gun_aim[1]) * max(1, d) < 5
                if compatible and d < w.get("stats", {}).get("range", 230) - 4 and angle < 0.20:
                    fire[i] = True
            elif i == slot and k == "laser":
                charge = w.get("charge", 0.0)
                if d < w.get("stats", {}).get("range", 280) - 8:
                    # Charge to full power, then release on target. Always avoid auto release.
                    charge_limit = w.get("stats", {}).get("charge", 0.75)
                    if charge < charge_limit - 0.02 or not in_arc:
                        fire[i] = True
                    else:
                        fire[i] = False
                elif charge > 0:
                    fire[i] = False
            elif i == slot and k == "lightning":
                if d < w.get("stats", {}).get("range", 45) - 3 and angle < 0.28 and me.get("energy", 0) > 8:
                    fire[i] = True
            elif i == slot and k == "mortar":
                if d > 30 and angle < 0.14:
                    fire[i] = True
        return [aim[0], clamp(aim[1], -1.35, 1.35)], fire, reload

    def act(self, obs: mf.Observation) -> mf.Action:
        me, zone = obs["self"], obs["zone"]
        x, y, _ = me["pos"]
        now = obs.get("t", obs.get("tick", 0) / 20)
        act = {"tick": obs["tick"]}
        if obs.get("offer"):
            act["pick"] = self.pick_upgrade(obs["offer"].get("choices", []))
        # Damage bearings reveal attackers even when the enemy is behind cover.
        for event in obs.get("events", []):
            if event.get("type") == "damaged" and event.get("bearing") is not None:
                self.last_damage_yaw = float(event["bearing"])
                self.last_damage_t = now
        target, enemies = self.choose_target(obs, me)
        aim, fire, reload = self.combat(obs, me, target)
        if not enemies and target is None and self.last_damage_yaw is not None and now - self.last_damage_t < 3.0:
            aim = [self.last_damage_yaw, 0.0]
        act["aim"] = aim
        act["fire"] = fire
        if reload:
            act["reload"] = True

        # Movement weighs avoiding enemy teams over dealing damage.
        navx, navy, zone_urg, cur_margin, next_margin = self.zone_navigation(me, zone, now)
        stex, stey = navx, navy
        closest_enemy = 1e9
        if enemies:
            nearest = min(enemies, key=lambda m: dist2(m["pos"], me["pos"]))
            closest_enemy = dist2(nearest["pos"], me["pos"])
            self.last_threat = tuple(nearest["pos"][:2])
            self.last_threat_t = now
            for m in enemies:
                d = max(1.0, dist2(m["pos"], me["pos"]))
                if d > 170:
                    continue
                ux, uy = unit(x - m["pos"][0], y - m["pos"][1])
                weight = 3.5 * (1 - d / 180) ** 2 + (2.0 if d < 38 else 0.0)
                stex += weight * ux
                stey += weight * uy
            # A perpendicular escape makes leading projectile aim harder.
            vx, vy = unit(x - nearest["pos"][0], y - nearest["pos"][1])
            side = self.preferred_side
            if closest_enemy < 130:
                side_factor = 0.9 if closest_enemy > 50 else 1.2
                stex += -vy * side * side_factor
                stey += vx * side * side_factor
        elif self.last_threat and now - self.last_threat_t < 3:
            ux, uy = unit(x - self.last_threat[0], y - self.last_threat[1])
            stex += 0.5 * ux
            stey += 0.5 * uy

        # Sounds provide useful danger bearings outside the direct line of sight.
        sound_danger = 0.0
        for sound in obs.get("sounds", []):
            if sound.get("kind") in ("death", "drop", "crate", "reload", "step"):
                continue
            if float(sound.get("age", 0)) > 0.7:
                continue
            distance_class = sound.get("dist", "far")
            strength = {"near": 1.25, "mid": 0.35}.get(distance_class, 0.0)
            if strength and sound.get("bearing") is not None:
                bearing = float(sound["bearing"])
                stex -= math.cos(bearing) * strength
                stey -= math.sin(bearing) * strength
                sound_danger += strength
        evx, evy, projectile_danger = self.projectile_evasion(obs, me)
        stex += 4 * evx
        stey += 4 * evy
        goal, loot_act = self.choose_loot(obs, me, enemies, zone)
        act.update(loot_act)
        if goal and closest_enemy > 65 and zone_urg < 3:
            gx, gy = unit(goal[0] - x, goal[1] - y)
            # Loot is opportunistic; never prioritize it over the zone.
            stex += gx * 1.6
            stey += gy * 1.6
        if not enemies and not goal and zone_urg == 0 and projectile_danger < 0.2 and sound_danger < 0.3:
            # Camp quietly to regenerate HP and avoid revealing ourselves by footsteps.
            stex, stey = 0.0, 0.0

        # Detect when buildings or terrain prevent progress. Try to skirt or vault.
        if obs["tick"] - self.last_check >= 16:
            if self.last_pos is not None and self.last_move != (0, 0):
                if math.hypot(x - self.last_pos[0], y - self.last_pos[1]) < 2:
                    self.stuck = min(30, self.stuck + 1)
                else:
                    self.stuck = max(0, self.stuck - 2)
            self.last_pos = (x, y)
            self.last_check = obs["tick"]
        if self.stuck > 2:
            ux, uy = unit(stex, stey)
            if ux == uy == 0:
                ux = 1.0
            side = self.preferred_side if self.stuck < 12 else -self.preferred_side
            stex = 0.25 * ux - 2.5 * side * uy
            stey = 0.25 * uy + 2.5 * side * ux

        if zone_urg >= 4 and (cur_margin < 8 or zone.get("dps", 0) >= 100):
            # Commit to the escape waypoint even when multiple pursuers push outward.
            stex, stey = navx, navy
        mx, my = unit(stex, stey)
        moving = abs(stex) + abs(stey) > 0.08
        act["move"] = [mx, my] if moving else [0.0, 0.0]
        self.last_move = (mx, my) if moving else (0.0, 0.0)
        energy = me.get("energy", 0)
        # Boost only when escaping serious threats or zone damage; conserve energy.
        danger = closest_enemy < 55 or projectile_danger > 0.55
        act["boost"] = bool(moving and energy > (15 if cur_margin < 0 else 42)
                            and (cur_margin < 0 or (zone_urg >= 4 and next_margin < 0)
                                 or (danger and me.get("grounded", True))))
        need_jump = (self.stuck > 3 or (closest_enemy < 85 and projectile_danger > 0.2))
        if (need_jump and me.get("grounded", True) and energy > 28
                and obs["tick"] - self.last_jump > 35):
            act["jump"] = True
            self.last_jump = obs["tick"]
        if not me.get("grounded", True) and self.stuck > 4 and energy > 45:
            act["boost"] = True
        return act



# ---- Aegis V2: map-aware survival, non-player objectives, and recovery ----


class CoverMap:
    """Compact, optional 2-D obstacle index over grounded building footprints.

    If the competition map is unavailable or differs from the documented
    structure, navigation falls back to the v1 reactive movement policy.
    """
    def __init__(self, directory):
        self.grid = {}
        self.routes = None
        self.features = []
        if not directory:
            return
        try:
            with open(os.path.join(directory, "v2", "world.json"), encoding="utf-8") as fh:
                raw = json.load(fh)
            for p in raw.get("prisms", []):
                ring = p.get("ring", [])
                if not isinstance(ring, list) or len(ring) < 6:
                    continue
                pts = tuple((float(ring[j]), float(ring[j+1])) for j in range(0, len(ring)-1, 2))
                lo_x = min(t[0] for t in pts)
                hi_x = max(t[0] for t in pts)
                lo_y = min(t[1] for t in pts)
                hi_y = max(t[1] for t in pts)
                feat = (pts, float(p.get("base", -100.0)),
                        float(p.get("top", 100.0)), (lo_x, lo_y, hi_x, hi_y),
                        p.get("holes", []))
                idx = len(self.features)
                self.features.append(feat)
                # Massive stadium prisms occupy many cells; still acceptable at initialization.
                for gx in range(int((lo_x-5)//35), int((hi_x+5)//35)+1):
                    for gy in range(int((lo_y-5)//35), int((hi_y+5)//35)+1):
                        self.grid.setdefault((gx, gy), []).append(idx)
            self.routes = GroundRoutes(self, raw, directory)
        except (OSError, EOFError, ValueError, TypeError, KeyError, IndexError):
            self.grid = {}
            self.features = []

    @staticmethod
    def inside(poly, x, y):
        state = False
        x0, y0 = poly[-1]
        for x1, y1 in poly:
            if (y0 > y) != (y1 > y):
                cross = (x1-x0) * (y-y0) / (y1-y0) + x0
                if x < cross:
                    state = not state
            x0, y0 = x1, y1
        return state

    @staticmethod
    def to_points(pts):
        if not isinstance(pts, list) or len(pts) < 6:
            return ()
        try:
            return tuple((float(pts[j]),float(pts[j+1])) for j in range(0,len(pts)-1,2))
        except (ValueError, TypeError):
            return ()

    @staticmethod
    def near_wall(poly, x, y, radius):
        # Also detect a cylindrical mech touching a wall even when its center
        # is just outside the actual polygon.
        x0, y0 = poly[-1]
        rr = radius * radius
        for x1, y1 in poly:
            vx, vy = x1-x0, y1-y0
            seg = vx*vx+vy*vy
            t = clamp(((x-x0)*vx+(y-y0)*vy)/max(1e-8, seg),0.0,1.0)
            if (x-x0-t*vx)**2+(y-y0-t*vy)**2 < rr:
                return True
            x0,y0=x1,y1
        return False

    def blocked(self, x, y, z, radius=2.75):
        if not self.features:
            return False
        gx, gy = int(x//35), int(y//35)
        for idx in self.grid.get((gx,gy), ()):
            poly,base,top,bb,holes = self.features[idx]
            # Ignore overhead bridges and prisms fully beneath our mech.
            if z + 8.5 <= base or z + 1.2 >= top:
                continue
            if x < bb[0]-radius or x > bb[2]+radius or y < bb[1]-radius or y > bb[3]+radius:
                continue
            body = self.inside(poly,x,y)
            touch = self.near_wall(poly,x,y,radius)
            if not body and not touch:
                continue
            # Courtyards and gaps in stadium footprints must remain passable.
            if holes and body:
                in_hole = False
                for h in holes:
                    points = self.to_points(h)
                    if points and self.inside(points,x,y) and not self.near_wall(points,x,y,radius):
                        in_hole=True
                        break
                if in_hole:
                    continue
            return True
        return False

    def safe_vector(self, pos, vector, zone):
        """Avoid driving into known obstacles; short horizon + zone barrier."""
        if not self.features:
            return vector
        x,y,z=pos
        dx,dy=unit(*vector)
        if dx==dy==0:
            return vector
        # Don't interfere with valid directions; solves the common "stuck at wall" problem.
        def possible(ax, ay):
            for d in (5.0, 10.0, 16.0):
                px,py=x+ax*d,y+ay*d
                if self.blocked(px,py,z):
                    return False
                if zone and d>5:
                    c=zone.get('center',(0,0))
                    r=float(zone.get('radius',500))
                    if math.hypot(px-c[0],py-c[1]) > r + 3 and math.hypot(x-c[0],y-c[1]) < r-8:
                        return False
            return True
        if possible(dx,dy):
            return vector
        # Prefer a nearby tangent, retaining the original steering's intention.
        for angle in (0.27,-0.27,0.55,-0.55,0.85,-0.85,1.15,-1.15,1.55,-1.55,2.0,-2.0):
            a=math.atan2(dy,dx)+angle
            vx,vy=math.cos(a),math.sin(a)
            if possible(vx,vy):
                return vx,vy
        return vector


class Aegis(Bot):
    name = "Aegis-Survivor-v3"

    def __init__(self, hello):
        super().__init__(hello)
        self.map = CoverMap(hello.get("map",{}).get("dir", ""))
        self.last_aggressor = None
        self.last_aggressor_time = -999.0
        self.focus = None

    def choose_target(self, obs, me):
        pos=me['pos']
        enemies=[m for m in obs.get('mechs',()) if m.get('kind')=='mech']
        nearest_enemy=min((dist2(m['pos'],pos) for m in enemies), default=1e6)
        hp_pct=me.get('hp',1400)/max(1,me.get('max_hp',1400))
        if enemies:
            def priority(enemy):
                distance = dist2(enemy["pos"], pos)
                # Finish wounded pursuers before splitting damage onto another.
                remaining = enemy.get("hp", 1400) / max(1, enemy.get("max_hp", 1400))
                return (distance + 85 * remaining - 18 * (enemy["id"] == self.focus)
                        - 20 * bool(enemy.get("charging")))
            target = min(enemies, key=priority)
            self.focus = target["id"]
            return target, enemies
        # Drones drop two cubes: a very good low-risk early upgrade source when
        # we can remain outside the 45 m drone engagement range.
        if hp_pct>0.70 and obs.get('alive',16)>7 and nearest_enemy>145:
            drones=[m for m in obs.get('mechs',()) if m.get('kind')=='npc'
                    and dist2(m['pos'],pos)<175]
            if drones:
                m=min(drones,key=lambda m:dist2(m['pos'],pos)+m.get('hp',350)*0.015)
                return m,enemies
            crates=[m for m in obs.get('crates',()) if dist2(m['pos'],pos)<160]
            if crates:
                m=min(crates,key=lambda m: dist2(m['pos'],pos) + m.get('hp',300)*0.01)
                return {'kind':'crate', 'pos':m['pos'], 'vel':[0,0,0]},enemies
        return None,enemies

    def choose_loot(self, obs, me, enemies, zone):
        sx,sy,sz=me['pos']
        pos=me['pos']
        hp_pct=me.get('hp',1)/max(1,me.get('max_hp',1))
        nearest_enemy=min((dist2(m['pos'],pos) for m in enemies),default=1e6)
        current_center=zone.get('center',[0,0])
        next_center=zone.get('next_center',current_center)
        next_radius=float(zone.get('next_radius',zone.get('radius',500)))
        now=float(obs.get('t',0.0))
        urgency=next_radius-dist2(pos,next_center)
        safe_to_loot = nearest_enemy > (95 if hp_pct > 0.80 else 145) and urgency>-35
        weapons=me.get('weapons',[])
        owned=[w.get('kind') for w in weapons if w]
        action={}
        # Pick up items if we are already standing on them, even during combat.
        for p in sorted(obs.get('pickups',()),key=lambda p:dist2(p['pos'],pos)):
            dd=dist2(p['pos'],pos)
            if dd > self.pickup_radius:
                break
            if abs(p["pos"][2] - sz) > 4 or p["kind"] in owned:
                continue
            if None in weapons:
                action["take"] = p["id"]
                break
            if p["kind"] == "laser" and "mortar" in owned:
                action.update(take=p["id"], drop=owned.index("mortar"))
                break
        options=[]
        if safe_to_loot:
            for cube in obs.get('cubes',()):
                dd=dist2(cube['pos'],pos)
                if dd>100 or abs(cube['pos'][2]-sz)>4:continue
                if not within_zone(cube['pos'], next_center, max(12,next_radius+10)):
                    continue
                # Pure power cubes have large immediate value, especially before 3 cubes.
                value=78.0 if me.get('cubes',0)<3 else 55.0
                options.append((value - 0.47*dd, cube['pos'][:2]))
            # Capture supply drops for 3 cubes, full health and energy; approach
            # early enough for the 4 s capture timer once the drop lands.
            for drop in obs.get('drops',()):
                if drop.get('holder') not in (None,me.get('id')):continue
                dd=dist2(drop['pos'],pos)
                lands=float(drop.get('lands_in',999))
                if lands>25 or dd>170 or not within_zone(drop['pos'],next_center,max(12,next_radius+10)):
                    continue
                value=120 + (55 if hp_pct<0.9 else 0) - 0.45*dd - max(0,lands-8)*0.6
                options.append((value,drop['pos'][:2]))
            if len(weapons)<2 or any(w is None for w in weapons):
                for p in obs.get('pickups',()):
                    dd=dist2(p['pos'],pos)
                    if dd>120 or abs(p['pos'][2]-sz)>4 or p.get('kind') in owned:continue
                    if not within_zone(p['pos'],next_center,max(12,next_radius+10)):continue
                    values={'laser':88,'mortar':78,'lightning':56}
                    options.append((values.get(p.get('kind'), 0)-0.49*dd,p['pos'][:2]))
            # Farm a crate or NPC if it can be finished without advancing into danger.
            if hp_pct>0.78 and obs.get('alive',16)>8 and me.get('cubes',0)<6:
                for m in obs.get('mechs',()):
                    if m.get('kind')!='npc':continue
                    dd=dist2(m['pos'],pos)
                    if 60<dd<135 and within_zone(m['pos'],next_center,max(12,next_radius+10)):
                        options.append((35-0.24*dd,m['pos'][:2]))
                for p in obs.get('crates',()):
                    dd=dist2(p['pos'],pos)
                    if 55<dd<120 and within_zone(p['pos'],next_center,max(12,next_radius+10)):
                        options.append((27-0.22*dd,p['pos'][:2]))
        if options:
            best=max(options,key=lambda a:a[0])
            if best[0]>2:
                return best[1],action
        return None,action

    def combat(self, obs, me, target):
        aim,fire,reload=super().combat(obs,me,target)
        weapons=me.get('weapons',())
        # Never let a gun run empty during an unavoidable close encounter.
        gun=next((w for w in weapons if w and w.get('kind')=='gun'),None)
        if gun and gun.get('ammo',12)<=3 and gun.get('reload',0)<=0:
            nearest_enemy=min((dist2(m['pos'],me['pos']) for m in obs.get('mechs',())
                               if m.get('kind')=='mech'),default=1e6)
            if nearest_enemy>80 or gun.get('ammo',12)<=0:
                reload=True
        if target and target["kind"] != "mech" and me["hp"] < 0.8 * me["max_hp"]:
            fire = [False, False]
        return aim,fire,reload

    def act(self, obs):
        act=super().act(obs)
        me=obs['self']
        pos=me['pos']
        now=float(obs.get('t',0.0))
        enemies=[m for m in obs.get('mechs',()) if m.get('kind')=='mech']
        closest=min((dist2(m['pos'],pos) for m in enemies),default=1e6)
        # Rear damage is especially painful (1.5x). Don't keep staring at a
        # distant farming target when a closer attacker is shooting our back.
        for ev in obs.get('events',()):
            if ev.get('type')=='damaged' and ev.get('bearing') is not None and ev.get('rear'):
                self.last_aggressor=float(ev['bearing'])
                self.last_aggressor_time=now
        if self.last_aggressor is not None and now-self.last_aggressor_time<0.9:
            angle=yaw_diff(self.last_aggressor, act.get('aim',(0,0))[0])
            if abs(angle)>1.0 and closest>38:
                act['aim']=[self.last_aggressor,0.0]
                act['fire']=[False,False]
        # Move around solid obstructions rather than spending match seconds
        # running face first into building footprints.
        move=act.get('move',[0.0,0.0])
        if math.hypot(*move)>0.1:
            mx,my=self.map.safe_vector(pos,move,obs.get('zone'))
            act['move']=list(unit(mx,my))
        # If the zone has already engulfed us, aggressively use remaining energy.
        if not me.get('in_zone', True) and me.get('energy',0)>10 and math.hypot(*act['move'])>0.5:
            act['boost']=True
        sound = next((s for s in obs.get("sounds", ()) if s.get("dist") == "near"
                      and s.get("kind") in ("gun", "laser_charge", "boost", "step")), None)
        if not enemies and sound:
            # Turn sensors toward the most recent close threat behind us.
            if not any(act["fire"]):
                act["aim"] = [sound["bearing"], 0.0]
        return act


if __name__ == "__main__":
    mf.run(Aegis)
