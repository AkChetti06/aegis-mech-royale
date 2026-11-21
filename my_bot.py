"""Aegis for the mechfight/1 arena. All decisions use the current match observation and map."""

import math
import random

import mechfight as mf
from evasion import anticipated, incoming, choose_motion
from combat import FireControl, angle, distance
from navigation import Campus, RouteFollower


def unit(x, y):
    norm = math.hypot(x, y)
    return (x / norm, y / norm) if norm > 0.001 else (0.0, 0.0)


def point_direction(p, origin):
    return math.atan2(p[1] - origin[1], p[0] - origin[0])


def clamp(v, low, high):
    return max(low, min(high, v))


HEADINGS = tuple((math.cos(i * math.tau / 16), math.sin(i * math.tau / 16)) for i in range(16))
WEAPON_VALUE = {"gun": 0, "lightning": 90, "mortar": 114, "laser": 134}


class Aegis:
    name = "Aegis-Sentinel"

    def __init__(self, hello, campus=None):
        self.config = hello["config"]
        self.campus = campus
        if campus is None:
            try:
                self.campus = Campus(hello["map"]["dir"])
            except (OSError, EOFError, ValueError, KeyError, IndexError):
                pass
        self.route = RouteFollower(self.campus)
        self.control = FireControl(self.campus, 1.0 / self.config["match"]["tick_hz"])
        self.rng = random.Random(hello.get("seed", 0))
        self.items = {}
        self.attackers = {}
        self.repairing = False
        self.round_history = {}
        self.observed_lead = 0.45
        self.aggressor = None
        self.damage_yaw = None
        self.last_damage = -100.0
        self.goal = None
        self.goal_key = None
        self.plan_at = 0
        self.last_circle = None
        self.side = self.rng.choice((-1, 1))
        self.juke_at = self.next_jump = 0.0
        self.jet_until = -1.0
        self.last_move = (0.0, 0.0)
        self.check_pos = None
        self.check_time = -0.8
        self.stuck = 0
        self.bad_goals = {}
        self.deposit = None
        self.vault = None
        self.ledge_reserve = 0.0
        self.withdraw_until = -1.0
        self.hurt_rate = 0.0
        self.last_hp_time = 0.0
        self.sprinting = False

    def upgrade(self, choices, me):
        # Mag size and reload speed both improve sustained arm-gun damage.
        scores = {
            "gun_mag": 93,
            "plating": 91,
            "gun_damage": 89,
            "armor": 84,
            "nanites": 86,
            "servos": 81,
            "gun_rate": 75,
            "gun_velocity": 72,
            "laser_damage": 77,
            "laser_charge": 70,
            "mortar_reload": 70,
            "mortar_splash": 64,
            "jets": 67,
            "reactor": 60,
            "lightning_range": 64,
            "capacitor": 55,
            "sensors": 48,
            "gyros": 40,
        }
        if me["hp"] < 0.65 * me["max_hp"]:
            scores["plating"] += 10
            scores["nanites"] += 6
        if me["upgrades"].count("gun_mag") > 0:
            scores["gun_mag"] -= 15
        return max(range(len(choices)), key=lambda i: scores.get(choices[i]["id"], 30))

    def remember(self, obs, me, now):
        self.hurt_rate *= math.exp(-(now - self.last_hp_time) / 2.0)
        self.last_hp_time = now
        fresh = set()
        for group in ("crates", "pickups", "cubes"):
            for obj in obs[group]:
                key = (group, obj["id"])
                fresh.add(key)
                self.items[key] = (obj, now)
        for key, (obj, seen) in list(self.items.items()):
            # Recheck a nearby item in the omnidirectional sensor radius.
            if key in fresh:
                continue
            nearby = distance(me["pos"], obj["pos"]) < 16 and abs(me["pos"][2] - obj["pos"][2]) < 6
            if nearby or now - seen > (70 if key[0] == "pickups" else 24):
                del self.items[key]
        for event in obs["events"]:
            if event["type"] == "damaged":
                if event.get("bearing") is not None:
                    self.aggressor, self.damage_yaw = event.get("by"), event["bearing"]
                    self.last_damage = now
                    self.attackers[self.aggressor] = now
                    self.hurt_rate += event.get("amount", 0) / me["max_hp"]
            elif event["type"] == "kill":
                if self.deposit and self.deposit[0] == event.get("target"):
                    self.deposit = (None, self.deposit[1], now)
            elif event["type"] in ("pickup", "took", "drop", "take_failed"):
                self.plan_at = now

    def dangerous(self, me, enemies):
        pos = me["pos"]
        muzzle = (pos[0], pos[1], pos[2] + me["stats"]["height"] * 0.65)
        threats = []
        for e in enemies:
            d = distance(pos, e["pos"])
            ey = (*e["pos"][:2], e["pos"][2] + e["height"] * 0.8)
            exposed = self.campus is None or self.campus.clear(ey, muzzle)
            to_me = math.atan2(pos[1] - ey[1], pos[0] - ey[0])
            watching = abs(angle(e.get("aim", (0, 0))[0] - to_me)) < 0.6
            weight = (1.2 if watching else 0.68) * clamp(1.45 - d / 180, 0.12, 1.35)
            if e.get("charging"):
                weight *= 1.5
            threats.append((weight * (1 if exposed else 0.18), e, exposed))
        threats.sort(key=lambda v: -v[0])
        return threats

    def pickup(self, me, pickups):
        kinds = [w["kind"] if w else None for w in me["weapons"]]
        for p in sorted(pickups, key=lambda item: distance(item["pos"], me["pos"])):
            if distance(me["pos"], p["pos"]) > self.config["loot"]["pickup_radius"]:
                break
            if abs(p["pos"][2] - me["pos"][2]) > 4 or p["kind"] in kinds:
                continue
            if None in kinds:
                return {"take": p["id"]}
            for i, k in enumerate(kinds):
                if k != "gun" and WEAPON_VALUE[p["kind"]] > WEAPON_VALUE[k] + 12:
                    return {"take": p["id"], "drop": i}
        return {}

    def objective(self, obs, me, threats, now):
        pos, zone = me["pos"], obs["zone"]
        c, r = zone["center"], zone["radius"]
        nc, nr = zone["next_center"], zone["next_radius"]
        health = me["hp"] / me["max_hp"]
        cur_margin = r - distance(pos, c)
        next_margin = nr - distance(pos, nc)
        remaining = max(0, zone["shrink_end"] - now)
        speed = me["stats"]["walk_speed"]
        # Leave time for a detour and a jet climb. Near the end, meters matter.
        urgent = cur_margin < min(11, r * 0.24) or next_margin < -max(0, (remaining - 9) * speed * 0.47)
        phase = (tuple(nc), nr)
        if self.last_circle != phase:
            self.last_circle, self.plan_at = phase, now
        if (
            self.goal_key
            and self.goal_key[0] in ("cubes", "pickups", "crates")
            and self.goal_key not in self.items
        ):
            self.plan_at = now
        if self.goal and now < self.plan_at:
            return self.goal, self.goal_key, urgent
        self.plan_at = now + (0.45 if threats or urgent else 0.72)
        self.bad_goals = {k: t for k, t in self.bad_goals.items() if t > now}
        best = -math.inf
        goal, key = None, None
        engage = sum(t[0] for t in threats if t[2])
        close = min((distance(pos, e["pos"]) for _, e, line in threats if line), default=999)
        kinds = [w["kind"] for w in me["weapons"] if w]
        safe_depth = min(37, max(1, nr * 0.32))
        urgency = clamp(1.35 - remaining / 90, 0.20, 1.35)
        if urgent:
            urgency = 3.0
        withdrawing = now < self.withdraw_until

        def consider(p, name, benefit, risk=True):
            nonlocal best, goal, key
            if name in self.bad_goals:
                return
            d = distance(p, pos)
            height = max(0.0, p[2] - pos[2])
            if height > 54 and d < 35:
                return
            score = benefit - 0.34 * d - 0.44 * height
            score -= 2.7 * max(0, distance(c, p) - r + min(8, r * 0.18))
            score -= urgency * max(0, distance(nc, p) - nr + safe_depth)
            if name == self.goal_key:
                score += 9
            if risk:
                for weight, foe, exposed in threats[:4]:
                    fd = distance(foe["pos"], p)
                    if fd < 75 and d > 7:
                        score -= weight * min(55, (75 - fd) * (0.63 if health > 0.6 else 1.0))
            if score > best:
                best, goal, key = score, p, name

        def surface(x, y):
            return (x, y, self.campus.height(x, y) if self.campus else pos[2])

        if nr < 9 and remaining < 13:
            # Walk onto the exact end point, including when it lies on a roof.
            goal, key = surface(*nc), ("zone", 0)
        else:
            ux, uy = unit(pos[0] - nc[0], pos[1] - nc[1])
            drift = min(distance(pos, nc), max(0, nr - safe_depth - 5))
            inside = surface(nc[0] + ux * drift, nc[1] + uy * drift)
            if self.campus:
                ix = clamp(inside[0], self.campus.x0 + 4, self.campus.xmax - 4)
                iy = clamp(inside[1], self.campus.y0 + 4, self.campus.ymax - 4)
                inside = surface(ix, iy)
            consider(inside, ("zone", 0), 29 + (33 if urgent else 0), False)

            if self.campus:
                high = pos[2] - self.campus.ground[self.campus.index(*pos[:2])]
                # Broad roofs give room to step back out of an attacker's sight.
                for anchor, depth, rise in self.campus.anchors:
                    d = distance(anchor, pos)
                    if d > (330 if urgent else 230) or (urgent and distance(anchor, nc) > nr + 4):
                        continue
                    value = (
                        49
                        + min(29, rise) * 0.55
                        + depth * 0.65
                        + min(18, now * 0.10)
                        + (11 if high >= 8 else 0)
                    )
                    if health < 0.60 or engage > 1.5:
                        value += 15
                    if withdrawing:
                        value += 22
                    if high > 8 and abs(pos[2] - anchor[2]) < 2.5 and d < 35 and engage > 0.6:
                        value += 15
                    consider(anchor, ("roof", anchor[:2]), value)

            for name, (obj, seen) in self.items.items():
                p = obj["pos"]
                d = distance(pos, p)
                if d > 235:
                    continue
                if name[0] == "pickups":
                    k = obj["kind"]
                    if k in kinds or (
                        len(kinds) >= 2 and WEAPON_VALUE[k] <= max(WEAPON_VALUE[v] for v in kinds) + 12
                    ):
                        continue
                    value = WEAPON_VALUE[k] + (17 if len(kinds) == 1 else -12)
                elif name[0] == "cubes":
                    value = 133 if me["cubes"] < 3 else 118 if me["cubes"] < 10 else 78
                    if d < 21:
                        value += 19
                else:
                    if health < 0.78 or engage > 0.75 or d > 85:
                        continue
                    value = 78 if len(kinds) < 2 else 58
                if withdrawing and d > 15:
                    value -= 46 if name[0] == "cubes" else 32
                consider(p, name, value - min(20, (now - seen) * 0.55))

            if self.deposit and self.deposit[0] is None and now - self.deposit[2] < 6:
                consider(self.deposit[1], ("salvage", 0), 85 if withdrawing else 125)
            for drop in obs["drops"]:
                if drop.get("holder") not in (None, me["id"]):
                    continue
                d = distance(pos, drop["pos"])
                if d > 360:
                    continue
                early_wait = max(0.0, drop["lands_in"] - d / speed - 2.0)
                value = 143 + 90 * (1 - health) - early_wait * 1.6 + drop.get("progress", 0) * 12
                consider(drop["pos"], ("drop", drop["id"]), value)
            # Select a nearby, reachable hiding spot when several guns have a shot.
            if self.campus and threats and (withdrawing or engage > 1.7):
                sources = [(*e["pos"][:2], e["pos"][2] + e["height"] * 0.8) for _, e, _ in threats[:3]]
                for dx, dy in HEADINGS[::2]:
                    for travel in (10, 24, 40):
                        x, y = pos[0] + travel * dx, pos[1] + travel * dy
                        if not (
                            self.campus.x0 + 4 < x < self.campus.xmax - 4
                            and self.campus.y0 + 4 < y < self.campus.ymax - 4
                        ):
                            continue
                        p = surface(x, y)
                        if abs(p[2] - pos[2]) > 4:
                            continue
                        eye = (x, y, p[2] + me["stats"]["height"] * 0.8)
                        hidden = sum(not self.campus.clear(source, eye) for source in sources)
                        if hidden == len(sources):
                            consider(
                                p,
                                ("cover", (round(x / 10), round(y / 10))),
                                116 + (1 - health) * 50,
                                False,
                            )
            if health < 0.85 and engage < 0.32 and close > 75 and now - self.last_damage < 14:
                consider(tuple(pos), ("heal", 0), 115, False)

        self.goal, self.goal_key = goal, key
        return goal, key, urgent

    def update_shot_estimate(self, obs, me, enemy):
        """Estimate whether this opponent is leading our strafe or aiming at our current position."""
        x, y = me["pos"][:2]
        vx, vy = me["vel"][:2]
        ex, ey = enemy["pos"][:2]
        dist = distance(me["pos"], enemy["pos"])
        if not (28 < dist < 200):
            return False
        self.round_history = {i: t for i, t in self.round_history.items() if t > obs["t"] - 1.2}
        lead_detected = False
        for p in obs["projectiles"]:
            if p["kind"] != "gun" or p["id"] in self.round_history:
                continue
            px, py = p["pos"][:2]
            sx, sy = p["vel"][:2]
            speed = math.hypot(sx, sy)
            if speed < 20 or distance(p["pos"], enemy["pos"]) > 0.63 * dist:
                continue
            # Project the round and the moving mech onto the same lateral axis.
            ux, uy = sx / speed, sy / speed
            toward = (x - px) * ux + (y - py) * uy
            if toward <= 0 or abs((px - ex) * uy - (py - ey) * ux) > 9:
                continue
            self.round_history[p["id"]] = obs["t"]
            side = (px - x) * (-uy) + (py - y) * ux
            travel = (-vx * uy + vy * ux) * toward / speed
            if abs(travel) < 1.7:
                continue
            if abs(side - travel) < abs(side) - 0.65:
                self.observed_lead = 0.77 * self.observed_lead + 0.23
                lead_detected |= abs(side - travel) < me["stats"]["radius"] + 1.0
            elif abs(side) < abs(side - travel) - 0.65:
                self.observed_lead *= 0.77
        return lead_detected

    def steer(self, obs, goal, key, threats, urgent):
        me, z, now = obs["self"], obs["zone"], obs["t"]
        pos = me["pos"]
        x, y, feet = pos
        energy = me["energy"]
        hp = me["hp"] / me["max_hp"]
        waypoint = self.route.next(pos, goal, now)
        gx, gy = unit(waypoint[0] - x, waypoint[1] - y)
        d_goal = distance(pos, goal)
        current_margin = z["radius"] - distance(pos, z["center"])
        safe = current_margin >= min(6, z["radius"] * 0.14)
        projected = [t for t in threats if t[2]]
        pressure = sum(t[0] for t in projected)
        danger = projected[0][1] if projected else None
        if danger:
            ex, ey = danger["pos"][:2]
            ax, ay = unit(x - ex, y - ey)
        else:
            ax, ay = 0.0, 0.0
        if danger and self.update_shot_estimate(obs, me, danger):
            self.juke_at = min(self.juke_at, now + 0.25)
        if now >= self.juke_at:
            self.side = -self.side
            interval = 0.80 if self.observed_lead > 0.68 else 3.4 if self.observed_lead < 0.31 else 1.75
            self.juke_at = now + interval * self.rng.uniform(0.75, 1.3)

        if now - self.check_time >= 0.85:
            progress = distance(pos, self.check_pos) if self.check_pos else 10
            if math.hypot(*self.last_move) > 0.5 and progress < 1.25:
                self.stuck += 1
            else:
                self.stuck = 0
            self.check_pos, self.check_time = tuple(pos), now
            if self.stuck >= 5 and key[0] not in ("zone", "cover"):
                self.bad_goals[key] = now + 10
                self.plan_at = now
                self.route.path = []

        # Persist the height of a blocked ledge across observations. Center-only
        # checks miss a wall brushing a shoulder and can cut the jets mid-climb.
        radius, body = me["stats"]["radius"], me["stats"]["height"]
        path_height = waypoint[2]
        if self.campus and (gx or gy):
            path_height = max(path_height, self.campus.lip_ahead(pos, (gx, gy), radius, body))
        if self.ledge_reserve:
            if energy < self.ledge_reserve:
                return (0.0, 0.0), False, False
            self.ledge_reserve = 0.0
            self.stuck = 3
        if self.campus and self.stuck >= 3 and not self.vault and me["grounded"]:
            # A bigger mech can snag a wall on its rear shoulder. Inspect the
            # whole footprint and save enough energy for one continuous climb.
            ring = [
                self.campus.support(x + dx * (radius + 2), y + dy * (radius + 2), feet, body)
                for dx, dy in HEADINGS
            ]
            lip = max(path_height, max(ring), feet + 9.0)
            energy_needed = (lip - feet) * me["stats"]["jet_drain"] / me["stats"]["max_rise_speed"] + 9.0
            if energy_needed <= me["max_energy"] - 1:
                if energy < energy_needed:
                    self.ledge_reserve = energy_needed
                    return (0.0, 0.0), False, False
                self.vault = (
                    lip + 1.0,
                    x + gx * (radius + 7),
                    y + gy * (radius + 7),
                    now + (lip - feet) / 5.0 + 3.0,
                )
            else:
                index = min(range(len(ring)), key=ring.__getitem__)
                return HEADINGS[index], False, False
        if self.vault:
            roof_z, end_x, end_y, expires = self.vault
            if now > expires or (feet >= roof_z + 0.45 and distance(pos, (end_x, end_y)) < 3):
                self.vault = None
            else:
                vx, vy = unit(end_x - x, end_y - y)
                return (
                    (vx, vy),
                    feet < roof_z + 0.8 and energy > 1,
                    bool(me["grounded"] and feet < roof_z - 0.7),
                )
        climb = path_height > feet + 1.25 and distance(pos, waypoint) < 23
        if climb and (
            not danger or urgent or self.stuck or key[0] in ("roof", "cover", "drop", "pickups")
        ):
            required = max(
                6.0,
                (path_height - feet - 2)
                * me["stats"]["jet_drain"]
                / max(3.0, me["stats"]["max_rise_speed"])
                + 5,
            )
            if required <= me["max_energy"] and energy >= required:
                reach = clamp(distance(pos, waypoint), radius + 3, 11)
                self.vault = (
                    path_height,
                    x + gx * reach,
                    y + gy * reach,
                    now + (path_height - feet) / 6 + 2,
                )
                return (gx, gy), True, bool(me["grounded"])
            if required <= me["max_energy"] and (self.stuck or not danger):
                return (0.0, 0.0), False, False

        collecting = key[0] in ("cubes", "pickups", "salvage") and d_goal < 6
        if collecting and feet > goal[2] + 3 and pressure < 1.7:
            # The pickup radius is measured in both XY and Z. Land over the item.
            return ((0.0, 0.0) if d_goal < 2.5 else unit(goal[0] - x, goal[1] - y)), False, False
        capturing = key[0] == "drop" and d_goal < 5 and abs(feet - goal[2]) < 4
        settled = key[0] in ("roof", "heal", "cover", "zone") and d_goal < 3 and abs(feet - goal[2]) < 4
        tight = z["radius"] < 6 and current_margin >= 0
        if (capturing and pressure < 1.7) or (settled and (not danger or tight)):
            return (0.0, 0.0), False, False

        nearby = incoming(obs, me)
        if energy > 75:
            self.sprinting = True
        elif energy < 30:
            self.sprinting = False
        if (
            key[0] in ("cubes", "pickups", "drop", "salvage")
            and d_goal < 12
            and abs(feet - goal[2]) < 4
            and not climb
        ):
            if d_goal < 2.0:
                return (0.0, 0.0), False, False
            return unit(goal[0] - x, goal[1] - y), bool(d_goal > 8 and energy > 42), False
        if not danger and not nearby and not self.stuck:
            return (
                (gx, gy),
                bool(me["grounded"] and d_goal > 22 and (energy > 68 or (not safe and energy > 8))),
                False,
            )
        # Score each direction using where it would put the cylinder in the next second.
        # Keep lateral velocity in a duel, then turn when a leading shot approaches.
        spacing = (
            33
            if any(w and w["kind"] == "lightning" for w in me["weapons"]) and len(projected) == 1
            else 84
        )
        retreat = now < self.withdraw_until or pressure > 1.65
        if retreat:
            spacing = 139
        if danger:
            d_enemy = distance(pos, danger["pos"])
            radial = clamp((spacing - d_enemy) / 38, -0.85, 1.65)
        else:
            radial = 0.0
        priority = 1.8
        if urgent:
            priority = 6.8 if not safe or z["dps"] >= 100 else 4.1
        elif key[0] in ("cubes", "pickups", "drop", "cover"):
            priority = 2.7 if d_goal > 3 else 0.5
        elif settled:
            priority = 0.0
        if danger and key[0] in ("zone", "roof") and not urgent:
            priority *= 1.65 if retreat else 0.82
        speed = me["stats"]["walk_speed"]
        candidates = []
        headings = ((gx, gy), *HEADINGS)
        for dx, dy in headings:
            if not (dx or dy):
                continue
            score = priority * (dx * gx + dy * gy) + 0.38 * (
                dx * self.last_move[0] + dy * self.last_move[1]
            )
            px, py = x + dx * speed, y + dy * speed
            if self.campus:
                if not (
                    self.campus.x0 + 3 < px < self.campus.xmax - 3
                    and self.campus.y0 + 3 < py < self.campus.ymax - 3
                ):
                    score -= 8.0
                floor = self.campus.support(px, py, feet, me["stats"]["height"])
                middle = self.campus.support(x + dx * 5, y + dy * 5, feet, me["stats"]["height"])
                if max(floor, middle) > feet + 2:
                    score -= 4.8
                if key[0] == "roof" and feet - goal[2] >= -2 and feet - floor > 4:
                    score -= 4.8
            else:
                floor = feet
            if danger:
                lateral = -ay * dx + ax * dy
                score += (
                    radial * (dx * ax + dy * ay)
                    + (2.5 if self.observed_lead < 0.45 else 1.8) * abs(lateral)
                    + 1.5 * self.side * lateral
                )
                if self.campus and retreat:
                    source = (*danger["pos"][:2], danger["pos"][2] + danger["height"] * 0.8)
                    eye = (px, py, max(floor, feet - 5) + me["stats"]["height"] * 0.65)
                    if not self.campus.clear(source, eye):
                        score += 3.5
            m = z["radius"] - distance((px, py), z["center"])
            score -= 0.67 * max(0, min(8, z["radius"] * 0.2) - m) if m < current_margin else 0
            if capturing and distance((px, py), goal) > 6:
                score -= 2.5
            candidates.append((score, dx, dy))
        shots = nearby + anticipated(me, projected)
        can_jump = not (capturing or collecting or tight) and now >= self.next_jump
        urgent_shot = any(
            t < 0.70 and abs(side) < me["stats"]["radius"] + 4.5 for t, _, _, side, _, _, _ in nearby
        )
        single_duel = len(projected) == 1 and d_enemy < 155 and not urgent_shot and key[0] != "cover"
        if single_duel and can_jump and me["grounded"] and energy > 40:
            self.jet_until = now + self.rng.uniform(0.55, 1.0)
        burst = single_duel and now < self.jet_until and energy > 16
        move, boost, jump = choose_motion(me, candidates, shots, self.sprinting, can_jump, burst)
        if jump:
            self.next_jump = now + self.rng.uniform(1.9, 2.9)
        return move, boost, jump

    def select_encounters(self, me, entities, threats, now):
        health = me["hp"] / me["max_hp"]
        # Keep the front armor toward an active attacker. If another mech is
        # looking away, a second duel can wait until the first is resolved.
        engaged = set(self.attackers)
        for _, e, line in threats:
            yaw = math.atan2(me["pos"][1] - e["pos"][1], me["pos"][0] - e["pos"][0])
            if line and abs(angle(e["aim"][0] - yaw)) < 0.20:
                engaged.add(e["id"])
        engaged_contacts = [e for e in entities if e["id"] in engaged]
        if engaged_contacts:
            targets = engaged_contacts + [
                e
                for e in entities
                if e["id"] not in engaged and e["kind"] == "mech" and e.get("hp", 1400) < 160
            ]
        else:
            # Keep an unthreatening mech outside the fight until equipment and
            # health give us a reasonable chance of winning an exchange.
            stocked = all(w is not None for w in me["weapons"])
            targets = []
            for e in entities:
                d = distance(me["pos"], e["pos"])
                if e["kind"] != "mech":
                    if d < 85 and (health > 0.83 or d < 32):
                        targets.append(e)
                elif (
                    d < 55
                    or e.get("hp", 1400) < 450
                    or health > 0.91
                    and (stocked or me["cubes"] >= 2)
                    and d < 185
                ):
                    targets.append(e)
        if health < 0.86 and not any(line and e["id"] in engaged for _, e, line in threats):
            self.repairing = True
        if health > 0.98 or engaged_contacts and now - self.last_damage < 1:
            self.repairing = False
        if self.repairing:
            targets = [
                e
                for e in targets
                if e["id"] in engaged or e["kind"] == "mech" and distance(me["pos"], e["pos"]) < 38
            ]
        return targets, engaged_contacts

    def act(self, obs):
        me, now = obs["self"], obs["t"]
        self.remember(obs, me, now)
        entities = []
        for raw in obs["mechs"]:
            npc = raw["kind"] == "npc"
            scale = 1 + self.config["cubes"].get("size_per_cube", 0.05) * raw.get("cubes", 0)
            radius = self.config["npc" if npc else "mech"]["radius"] * scale
            height = self.config["npc" if npc else "mech"]["height"] * scale
            entities.append(dict(raw, radius=radius, height=height))
        self.attackers = {i: since for i, since in self.attackers.items() if now - since < 5}
        threats = self.dangerous(me, [e for e in entities if e["kind"] == "mech"])
        health = me["hp"] / me["max_hp"]
        if (self.hurt_rate > 0.18 and health < 0.78) or (health < 0.50 and now - self.last_damage < 2):
            self.withdraw_until = now + 6
        goal, key, urgent = self.objective(obs, me, threats, now)
        covering = key[0] in ("heal", "cover", "roof", "drop")
        # Once concealed, give regen time to work instead of resetting it on a crate.
        quiet = (
            health < 0.84
            and covering
            and not any(t[2] for t in threats)
            or health < 0.43
            and key[0] == "cover"
            and now - self.last_damage > 1.0
        )
        crates = [
            dict(
                c,
                kind="crate",
                radius=self.config["crates"]["radius"],
                height=self.config["crates"]["height"],
            )
            for c in obs["crates"]
        ]
        targets, engaged_contacts = self.select_encounters(me, entities, threats, now)
        chosen = self.control.choose(
            me,
            targets,
            [] if engaged_contacts or self.repairing else crates,
            quiet,
            self.aggressor if now - self.last_damage < 3 else None,
        )
        if (
            chosen
            and chosen[0]["kind"] == "mech"
            and not (self.deposit and self.deposit[0] is None and now - self.deposit[2] < 3)
        ):
            self.deposit = (chosen[0]["id"], tuple(chosen[0]["pos"]), now)
        scan = me["aim"][0] + 0.35
        if self.damage_yaw is not None and now - self.last_damage < 1.4:
            scan = self.damage_yaw
        elif threats:
            e = threats[0][1]
            scan = math.atan2(e["pos"][1] - me["pos"][1], e["pos"][0] - me["pos"][0])
        elif obs["sounds"]:
            cues = [
                s
                for s in obs["sounds"]
                if s["dist"] != "far"
                and s["kind"] in ("gun", "boost", "laser_charge", "step", "lightning")
            ]
            if cues:
                scan = min(cues, key=lambda s: (s["dist"] != "near", s.get("age", 0)))["bearing"]
        if self.damage_yaw is not None and now - self.last_damage < 0.85 and not engaged_contacts:
            if not chosen or (
                chosen[0]["id"] != self.aggressor
                and abs(angle(point_direction(chosen[1], me["pos"]) - self.damage_yaw)) > 1.6
            ):
                chosen, scan = None, self.damage_yaw
        aim, fire, reload = self.control.command(me, chosen, scan, bool(threats))
        self.control.support_arc(me, aim, fire, engaged_contacts)
        move, boost, jump = self.steer(obs, goal, key, threats, urgent)
        self.last_move = move
        act = dict(
            tick=obs["tick"], aim=aim, fire=fire, reload=reload, move=list(move), boost=boost, jump=jump
        )
        act.update(self.pickup(me, obs["pickups"]))
        if obs.get("offer") and obs["offer"]["choices"]:
            act["pick"] = self.upgrade(obs["offer"]["choices"], me)
        return act


if __name__ == "__main__":
    mf.run(Aegis)
