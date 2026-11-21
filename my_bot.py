"""Aegis v1: defensive survival bot for Builders Mech Royale / mechfight/1.

Standard-library strategy built on the organizer's mechfight.py transport wrapper.
The bot deliberately favors placement and survival over risky eliminations.
"""
import math
import random
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
        # For a closing radius 0, treat the zone as fully urgent.
        if cur_r < 75 and deadline < 50:
            gx, gy = nxt_c[0], nxt_c[1]
            urgency = max(urgency, 4.0)
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
            return [desired[0] + 0.045, 0.0], fire, False
        kind = target.get("kind", "mech")
        tx, ty, tz = target["pos"]
        v = target.get("vel", [0, 0, 0])
        d = math.hypot(tx - sx, ty - sy)
        muzzle = sz + me.get("stats", {}).get("muzzle_height", 7.0)
        target_z = tz + (4.5 if kind == "mech" else 2.5)
        origin = (sx, sy, muzzle)
        dest = (tx, ty, target_z)
        actual_yaw = me.get("aim", [0.0, 0.0])[0]
        # Favor usable special weapons at short range; gun otherwise.
        special_slot = next((i for i, w in enumerate(weapons) if w and w.get("kind") == "laser" and w.get("charge", 0) > 0), None)
        if special_slot is None:
            special_slot = next((i for i, w in enumerate(weapons) if w and w.get("kind") == "lightning" and d < 40), None)
        if special_slot is None:
            special_slot = next((i for i, w in enumerate(weapons) if w and w.get("kind") == "laser" and d < 260), None)
        if special_slot is None:
            special_slot = next((i for i, w in enumerate(weapons) if w and w.get("kind") == "mortar" and 35 < d < 175), None)
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
            aim = mortar_aim(origin, dest, v, speed, stats.get("gravity", 22.0))
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
                if d < w.get("stats", {}).get("range", 230) - 4 and angle < 0.20:
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


mf.run(Bot)
