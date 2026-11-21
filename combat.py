"""Select an observable target and coordinate the two triggers on one torso."""

import math


def angle(a):
    return (a + math.pi) % math.tau - math.pi


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_aim(origin, dest):
    return (
        math.atan2(dest[1] - origin[1], dest[0] - origin[0]),
        math.atan2(dest[2] - origin[2], max(0.001, distance(origin, dest))),
    )


def intercept_time(delta, velocity, speed):
    a = sum(v * v for v in velocity) - speed * speed
    b = 2 * sum(v * d for v, d in zip(velocity, delta))
    c = sum(d * d for d in delta)
    disc = b * b - 4 * a * c
    if disc >= 0 and abs(a) > 1e-6:
        root = math.sqrt(disc)
        times = [t for t in ((-b - root) / (2 * a), (-b + root) / (2 * a)) if t > 0]
        if times:
            return min(3.5, min(times))
    return math.sqrt(c) / max(1, speed)


class FireControl:
    def __init__(self, campus):
        self.campus = campus
        self.focus = None

    def exposed(self, origin, target):
        p = target["pos"]
        h = target["height"]
        for f in (0.53, 0.83, 0.24):
            dest = (p[0], p[1], p[2] + f * h)
            if self.campus is None or self.campus.clear(origin, dest):
                return dest
        return None

    def choose(self, me, contacts, crates, recovery, aggressor):
        pos = me["pos"]
        origin = (pos[0], pos[1], pos[2] + me["stats"]["muzzle_height"])
        best = None
        blocked = []
        best_score = -math.inf
        for target in (*contacts, *crates):
            kind = target["kind"]
            d = distance(pos, target["pos"])
            if d > 265 or (kind == "crate" and d > 110):
                continue
            dest = self.exposed(origin, target)
            if dest is None:
                if kind == "mech" and 20 < d < 170:
                    blocked.append((d, target))
                continue
            pitch = point_aim(origin, dest)[1]
            if not (me["stats"]["min_pitch"] - 0.06 <= pitch <= me["stats"]["max_pitch"] + 0.06):
                continue
            hp = target.get("hp", 1400)
            bearing = math.atan2(dest[1] - pos[1], dest[0] - pos[0])
            if kind == "mech":
                # Count how much damage we still have to land to remove this gun.
                rear = abs(angle(bearing + math.pi - target.get("aim", (0, 0))[0])) > math.pi / 2
                score = 175 - min(2200, hp / (1.5 if rear else 1)) * 0.038 - 0.29 * d
                score += 21 * (target["id"] == aggressor) + 13 * target.get("charging", False)
                if recovery and d > 62 and hp > 270:
                    continue
            else:
                if recovery and target["id"] != aggressor:
                    continue
                score = (55 if kind == "npc" else 43) - d * 0.37 - hp * 0.038
                if kind == "npc" and d < 48:
                    score += 25
                if target["id"] == aggressor:
                    score += 50
            score += 12 * (target["id"] == self.focus)
            score -= 6 * abs(angle(bearing - me["aim"][0]))
            if score > best_score:
                best, best_score = (target, dest), score
        if blocked and (not best or best[0]["kind"] != "mech") and not recovery:
            mortar = next(
                (w for w in me["weapons"] if w and w["kind"] == "mortar" and w["cooldown"] < 0.16), None
            )
            if mortar:
                tube = (*pos[:2], pos[2] + me["stats"]["height"] + 0.5)
                for _, enemy in sorted(blocked, key=lambda item: item[0])[:2]:
                    feet = enemy["pos"]
                    dest = (*feet[:2], feet[2] + enemy["height"] * 0.18)
                    lob = self.mortar(tube, enemy, dest, mortar["stats"])
                    if lob and me["stats"]["min_pitch"] < lob[1] < me["stats"]["max_pitch"]:
                        best = (dict(enemy, lob=lob), dest)
                        break
        self.focus = best[0]["id"] if best else None
        return best

    def predict(self, target, dest, t):
        vx, vy, vz = target.get("vel", (0, 0, 0))
        x, y = dest[0] + vx * t, dest[1] + vy * t
        z = dest[2] + vz * t
        if self.campus and target["kind"] != "crate" and t > 0.08:
            foot = target["pos"][2]
            floor = self.campus.support(x, y, foot, target["height"])
            if floor > foot + 2.4 and vz <= 0.5:
                # A wall stops a walking target's lateral lead.
                x, y = dest[:2]
            elif abs(vz) > 1 or foot > floor + 2:
                z = max(floor + dest[2] - foot, z - 11 * t * t)
            else:
                z = floor + dest[2] - foot
        return x, y, z

    def mortar(self, origin, target, dest, stats):
        speed, gravity = stats["speed"], stats["gravity"]
        t = distance(origin, dest) / speed
        for _ in range(4):
            lead = self.predict(target, dest, t)
            # Uncertainty grows over a shell's long flight. Favor the ground
            # near the likely endpoint so a slight miss can still splash.
            if t > 0.55:
                factor = 1 / (1 + 0.27 * (t - 0.55))
                lead = (
                    dest[0] + (lead[0] - dest[0]) * factor,
                    dest[1] + (lead[1] - dest[1]) * factor,
                    lead[2],
                )
            d = max(0.1, distance(origin, lead))
            height = lead[2] - origin[2]
            v2 = speed**2
            disc = v2**2 - gravity * (gravity * d**2 + 2 * height * v2)
            if disc < 0:
                return None
            pitch = math.atan2(v2 - math.sqrt(disc), gravity * d)
            yaw = math.atan2(lead[1] - origin[1], lead[0] - origin[0])
            t = d / max(1, speed * math.cos(pitch))
        if self.campus:
            last = origin
            for f in (0.25, 0.5, 0.75, 1.0):
                pt = (
                    origin[0] + (lead[0] - origin[0]) * f,
                    origin[1] + (lead[1] - origin[1]) * f,
                    origin[2] + speed * math.sin(pitch) * t * f - 0.5 * gravity * (t * f) ** 2,
                )
                if not self.campus.clear(last, pt):
                    return None
                last = pt
        return yaw, pitch

    def command(self, me, chosen, scan_yaw, dangerous):
        slots = {w["kind"]: (i, w) for i, w in enumerate(me["weapons"]) if w}
        gun = slots.get("gun", (None, None))[1]
        reload = bool(gun and gun.get("reload", 0) == 0 and gun.get("ammo", 0) == 0)
        fire = [False, False]
        if not chosen:
            if gun and gun["ammo"] < gun["stats"]["mag"] - (4 if dangerous else 1):
                reload = True
            return [angle(scan_yaw), 0], fire, reload
        target, dest = chosen
        pos = me["pos"]
        origin = (pos[0], pos[1], pos[2] + me["stats"]["muzzle_height"])
        vel = target.get("vel", (0, 0, 0))
        d = math.dist(origin, dest)
        yaw, pitch = me["aim"]
        hitscan = point_aim(origin, self.predict(target, dest, 0.035))
        gun_aim = hitscan
        if gun:
            t = intercept_time(tuple(q - p for p, q in zip(origin, dest)), vel, gun["stats"]["speed"])
            gun_aim = point_aim(origin, self.predict(target, dest, t + 0.025))
        aim = gun_aim
        if "mortar" in slots:
            i, w = slots["mortar"]
            if w["cooldown"] <= 0.15 and d > max(13, w["stats"]["splash_radius"] + 6):
                tube = (pos[0], pos[1], pos[2] + me["stats"]["height"] + 0.5)
                splash_point = (*dest[:2], target["pos"][2] + target["height"] * 0.17)
                lob = target.get("lob") or self.mortar(tube, target, splash_point, w["stats"])
                if lob and me["stats"]["min_pitch"] <= lob[1] <= me["stats"]["max_pitch"]:
                    aim = lob
                    fire[i] = abs(angle(yaw - lob[0])) < 0.12 and abs(pitch - lob[1]) < 0.115
        if "laser" in slots:
            i, w = slots["laser"]
            s = w["stats"]
            if d < s["range"] and w["cooldown"] < 0.1:
                charged = w["charge"]
                power = (0.35 + 0.65 * min(1, charged / s["charge"])) * s["damage"]
                lethal = power * (1 + 0.04 * min(me["cubes"], 10)) >= target.get("hp", 9999)
                release = charged >= s["charge"] - 0.011 or (charged >= s["min_charge"] and lethal)
                if release or charged >= s["charge"] - 0.15:
                    aim = hitscan
                # Release only when the torso can slew onto the observed cylinder.
                tolerance = math.atan2(target["radius"] * 0.85, max(1, d))
                lined = (
                    abs(angle(yaw - hitscan[0])) < 0.06 + tolerance
                    and abs(pitch - hitscan[1]) < 0.05 + tolerance
                )
                fire[i] = not (release and lined)
        if gun:
            i = slots["gun"][0]
            # Simultaneous shots are useful when both trajectories fit inside the target.
            half_h = target["height"] * 0.48
            compatible = abs(aim[1] - gun_aim[1]) * d < half_h + 1.3
            compatible &= abs(angle(aim[0] - gun_aim[0])) * d < target["radius"] + 2
            fire[i] = bool(
                not target.get("lob")
                and compatible
                and d <= gun["stats"]["range"] + 2
                and abs(angle(yaw - gun_aim[0])) < 0.15
                and abs(pitch - gun_aim[1]) < 0.13
            )
            if d > gun["stats"]["range"] and gun["ammo"] < gun["stats"]["mag"] - 2:
                reload = True
        if "lightning" in slots:
            i, w = slots["lightning"]
            s = w["stats"]
            fire[i] = (
                d < s["range"]
                and abs(angle(yaw - hitscan[0])) < math.radians(s["cone_deg"]) - 0.03
                and me["energy"] >= s["energy"] + 1
            )
        return (
            [angle(aim[0]), max(me["stats"]["min_pitch"], min(me["stats"]["max_pitch"], aim[1]))],
            fire,
            reload,
        )
