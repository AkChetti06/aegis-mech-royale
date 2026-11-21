"""One-decision movement search using the arena's acceleration and projectile rules."""

import math


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def incoming(obs, me):
    """Project nearby missiles onto a plane perpendicular to their ground track."""
    x, y, z = me["pos"]
    mx, my, _ = me["vel"]
    shots = []
    for p in obs["projectiles"]:
        px, py, pz = p["pos"]
        vx, vy, vz = p["vel"]
        rx, ry = px - x, py - y
        ux, uy = vx - mx, vy - my
        t = -(rx * ux + ry * uy) / max(1, ux * ux + uy * uy)
        if not (0 < t < 1.05):
            continue
        speed = math.hypot(vx, vy)
        if speed < 2:
            continue
        nx, ny = -vy / speed, vx / speed
        offset = rx * nx + ry * ny
        if abs(offset) > 25:
            continue
        mortar = p["kind"] == "mortar"
        height = pz - z + vz * t - (11 * t * t if mortar else 0)
        if not (-10 < height < me["stats"]["height"] + 14):
            continue
        shots.append((t, nx, ny, offset, height, 5.2 if mortar else 6.7, 5.0 if mortar else 0.35))
    return sorted(shots)[:10]


def anticipated(me, threats):
    shots = []
    x, y, _ = me["pos"]
    vx, vy, vz = me["vel"]
    for weight, e, visible in threats[:3]:
        if not visible:
            continue
        ex, ey, _ = e["pos"]
        dx, dy = x - ex, y - ey
        dist = max(1.0, math.hypot(dx, dy))
        nx, ny = -dy / dist, dx / dist
        t = clamp(dist / 252 + 0.055, 0.12, 0.92)
        sideways = (vx * nx + vy * ny) * t
        dz = me["stats"]["height"] * 0.49
        # Account for both a center-aimed round and one leading our velocity.
        # Use the opposing torso to estimate which aim solution is more likely.
        yaw = e.get("aim", (math.atan2(dy, dx), 0))[0]
        aim_side = dist * math.sin(yaw - math.atan2(dy, dx))
        leading = abs(aim_side - sideways) + 0.6 < abs(aim_side)
        gain = clamp(weight, 0.25, 1.45)
        shots.append((t, nx, ny, 0.0, dz, gain * (1.05 if leading else 2.65), 0.35))
        shots.append((t, nx, ny, sideways, dz + vz * t, gain * (3.45 if leading else 1.75), 0.35))
    return shots


def choose_motion(me, headings, shots, sprint, can_jump, jet_burst):
    """Compare flight paths against the same shot set for every legal control."""
    st = me["stats"]
    vx, vy, vz = me["vel"]
    radius, height = st["radius"], st["height"]
    grounded = me["grounded"]
    energy = me["energy"]
    slow = 0.5 if any(w and w.get("charge", 0) > 0 for w in me["weapons"]) else 1.0
    modes = [(False, False)]
    if grounded and sprint and energy > 5:
        modes.append((True, False))
    if can_jump and grounded:
        modes.append((energy > 28, True))
    elif not grounded and energy > 12:
        modes.append((True, False))
    if jet_burst:
        modes = [(True, grounded)]
    best = -math.inf
    result = ((0.0, 0.0), False, False)
    for base, dx, dy in headings:
        for boost, jump in modes:
            air = not grounded or jump
            scale = st["boost_speed"] if boost and not air else st["walk_speed"]
            wx, wy = slow * scale * dx, slow * scale * dy
            ax, ay = wx - vx, wy - vy
            accel = st["air_accel"] if air else st["ground_accel"]
            transition = max(0.001, math.hypot(ax, ay) / accel)
            score = base
            if boost:
                score += 0.27 if not air else -0.16
                if energy < 26:
                    score -= 0.45
            if jump:
                score -= 0.19
            v0 = st["jump_speed"] if jump else vz
            az = st["jet_accel"] - st["gravity"] if boost else -st["gravity"]
            for t, nx, ny, offset, shot_z, damage, pad in shots:
                # Integral of the acceleration-limited velocity toward the command.
                ramp = t * t / (2 * transition) if t < transition else t - transition * 0.5
                cx, cy = vx * t + ax * ramp, vy * t + ay * ramp
                if air:
                    if boost and az > 0:
                        cap_time = clamp((st["max_rise_speed"] - v0) / az, 0, t)
                        foot = (
                            v0 * cap_time
                            + 0.5 * az * cap_time**2
                            + st["max_rise_speed"] * (t - cap_time)
                        )
                    else:
                        foot = v0 * t + 0.5 * az * t * t
                    if grounded:
                        foot = max(0.0, foot)
                else:
                    foot = 0.0
                side_miss = abs(cx * nx + cy * ny - offset) - radius - pad
                high_miss = max(foot - shot_z, shot_z - (foot + height)) - pad
                clearance = max(side_miss, high_miss)
                # A smooth shoulder avoids ties between two barely survivable moves.
                score -= damage * clamp(1.0 - clearance / 2.3, 0.0, 1.8)
            if score > best:
                best, result = score, ((dx, dy), boost, jump)
    return result
