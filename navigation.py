"""Height-aware routes and line-of-sight queries on the published arena map."""

import heapq
import json
import math
import os
import sys
from array import array


class Campus:
    cell = 2
    stride = 4

    def __init__(self, directory):
        folder = os.path.join(directory, "v2")
        with open(os.path.join(folder, "world.json"), encoding="utf-8") as stream:
            world = json.load(stream)
        terrain = world["terrain"]
        with open(os.path.join(folder, terrain["file"]), "rb") as stream:
            heights = array("h")
            heights.fromfile(stream, terrain["nx"] * terrain["ny"])
        if sys.byteorder != "little":
            heights.byteswap()

        bounds = world["bounds"]
        self.x0 = math.ceil(bounds["minX"])
        self.y0 = math.ceil(bounds["minY"])
        self.nx = int((bounds["maxX"] - self.x0) / self.cell) + 1
        self.ny = int((bounds["maxY"] - self.y0) / self.cell) + 1
        self.xmax = self.x0 + (self.nx - 1) * self.cell
        self.ymax = self.y0 + (self.ny - 1) * self.cell
        self.ground = array("f")
        columns = [
            round((self.x0 + i * self.cell - terrain["minX"]) / terrain["step"]) for i in range(self.nx)
        ]
        for j in range(self.ny):
            y = self.y0 + j * self.cell
            row = round((y - terrain["minY"]) / terrain["step"]) * terrain["nx"]
            self.ground.extend(heights[row + i] * terrain["scale"] for i in columns)
        self.top = self.ground[:]
        self.bridge_base = array("f", [math.inf]) * len(self.ground)
        self.bridge_top = array("f", [-math.inf]) * len(self.ground)
        for prism in world["prisms"]:
            self._rasterize(prism)
        self.deck = array("f", map(max, self.top, self.bridge_top))
        self._build_routes()

    def _rasterize(self, prism):
        # Even/odd scanline fill handles exterior rings and their courtyards together.
        rings = [prism["ring"], *prism.get("holes", ())]
        edges = []
        for ring in rings:
            points = list(zip(ring[::2], ring[1::2]))
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
                if y1 != y2:
                    edges.append((min(y1, y2), max(y1, y2), x1, y1, (x2 - x1) / (y2 - y1)))
        if not edges:
            return
        lo = max(0, math.ceil((min(e[0] for e in edges) - self.y0) / self.cell))
        hi = min(self.ny, math.ceil((max(e[1] for e in edges) - self.y0) / self.cell))
        bridge = prism["kind"] == "bridge"
        for j in range(lo, hi):
            y = self.y0 + j * self.cell
            crosses = sorted(x + (y - y1) * slope for yl, yh, x, y1, slope in edges if yl <= y < yh)
            for left, right in zip(crosses[::2], crosses[1::2]):
                first = max(0, math.ceil((left - self.x0) / self.cell))
                last = min(self.nx, math.ceil((right - self.x0) / self.cell))
                for k in range(j * self.nx + first, j * self.nx + last):
                    if bridge:
                        self.bridge_base[k] = min(self.bridge_base[k], prism["base"])
                        self.bridge_top[k] = max(self.bridge_top[k], prism["top"])
                    else:
                        self.top[k] = max(self.top[k], prism["top"])

    def index(self, x, y):
        i = max(0, min(self.nx - 1, round((x - self.x0) / self.cell)))
        j = max(0, min(self.ny - 1, round((y - self.y0) / self.cell)))
        return j * self.nx + i

    def height(self, x, y):
        return self.deck[self.index(x, y)]

    def support(self, x, y, feet, body=9):
        k = self.index(x, y)
        if feet + body < self.bridge_base[k]:
            return self.top[k]
        return self.deck[k]

    def lip_ahead(self, pos, direction, radius, body):
        x, y, feet = pos
        dx, dy = direction
        highest = feet
        for length in (1.5, 4.0, 7.0):
            px, py = x + dx * length, y + dy * length
            for side in (-radius, 0.0, radius):
                highest = max(highest, self.support(px - dy * side, py + dx * side, feet, body))
        return highest

    def clear(self, a, b):
        ax, ay, az = a
        bx, by, bz = b
        steps = max(1, math.ceil(math.hypot(bx - ax, by - ay) / 3))
        dx, dy, dz = (bx - ax) / steps, (by - ay) / steps, (bz - az) / steps
        # Both endpoints come from observations or bounded goals on the arena.
        ix = (ax - self.x0) / self.cell + 0.5
        iy = (ay - self.y0) / self.cell + 0.5
        dx /= self.cell
        dy /= self.cell
        nx, top, base, bridge = self.nx, self.top, self.bridge_base, self.bridge_top
        for _ in range(1, steps):
            ix += dx
            iy += dy
            az += dz
            k = int(iy) * nx + int(ix)
            if k < 0 or k >= len(top) or az < top[k] - 0.2 or base[k] < az < bridge[k]:
                return False
        return True

    def _build_routes(self):
        s = self.stride
        # Nodes sit 4 m inside the map boundary, with a nominal spacing of 8 m.
        self.cols = len(range(2, self.nx - 2, s))
        self.rows = len(range(2, self.ny - 2, s))
        self.points = []
        self.anchors = []
        for j in range(self.rows):
            for i in range(self.cols):
                u, v = i * s + 2, j * s + 2
                k = v * self.nx + u
                x, y = self.x0 + u * self.cell, self.y0 + v * self.cell
                z = self.deck[k]
                self.points.append((x, y, z))
                rise = z - self.ground[k]
                if not (8 < rise < 51) or (i + j) % 2:
                    continue
                depth = 0
                for d in (2, 4, 7, 10):
                    if min(u, v, self.nx - u - 1, self.ny - v - 1) < d:
                        break
                    rim = (
                        k - d,
                        k + d,
                        k - d * self.nx,
                        k + d * self.nx,
                        k - d * self.nx - d,
                        k + d * self.nx + d,
                        k - d * self.nx + d,
                        k + d * self.nx - d,
                    )
                    if any(abs(self.deck[n] - z) > 2 for n in rim):
                        break
                    depth = d * self.cell
                if depth >= 8:
                    self.anchors.append(((x, y, z), depth, rise))
        self.graph = [[] for _ in self.points]
        for a, (x, y, z) in enumerate(self.points):
            i, j = a % self.cols, a // self.cols
            for di, dj in ((1, 0), (0, 1), (1, 1), (-1, 1)):
                if not (0 <= i + di < self.cols and j + dj < self.rows):
                    continue
                b = (j + dj) * self.cols + i + di
                bx, by, bz = self.points[b]
                # Sample both sides of the mech as well as its center.
                dx, dy = bx - x, by - y
                distance = math.hypot(dx, dy)
                ox, oy = round(-dy * 2 / distance / self.cell), round(dx * 2 / distance / self.cell)
                offset = ox + oy * self.nx
                start = (j * s + 2) * self.nx + i * s + 2
                step = dj * self.nx + di
                peak = max(
                    self.deck[start + n * step + w] for n in (1, 2, 3) for w in (0, offset, -offset)
                )
                peak = max(peak, z, bz)
                for src, dst, climb in ((a, b, peak - z), (b, a, peak - bz)):
                    if climb > 49:
                        continue
                    extra = 1.65 * climb + 9 if climb > 1.2 else 0
                    self.graph[src].append((dst, distance + extra))

    def node(self, pos):
        step = self.cell * self.stride
        x = max(0, min(self.cols - 1, round((pos[0] - self.x0 - 2 * self.cell) / step)))
        y = max(0, min(self.rows - 1, round((pos[1] - self.y0 - 2 * self.cell) / step)))
        node = y * self.cols + x
        # At a facade a ground mech can round to a roof node or vice versa.
        if abs(self.points[node][2] - pos[2]) > 4:
            candidates = (
                (y + j) * self.cols + x + i
                for i in (-1, 0, 1)
                for j in (-1, 0, 1)
                if 0 <= x + i < self.cols and 0 <= y + j < self.rows
            )
            node = min(candidates, key=lambda n: math.dist(self.points[n], pos))
        return node

    def route(self, start, goal):
        queue = [(0, 0, start)]
        costs, parent = {start: 0}, {start: None}
        gx, gy, _ = self.points[goal]
        nearest, closest = start, math.inf
        for _ in range(2500):
            if not queue:
                break
            _, cost, here = heapq.heappop(queue)
            if cost != costs[here]:
                continue
            x, y, _ = self.points[here]
            remaining = math.hypot(gx - x, gy - y)
            if remaining < closest:
                nearest, closest = here, remaining
            if here == goal:
                break
            for other, edge in self.graph[here]:
                new_cost = cost + edge
                if new_cost >= costs.get(other, math.inf):
                    continue
                costs[other], parent[other] = new_cost, here
                ox, oy, _ = self.points[other]
                estimate = 1.15 * math.hypot(ox - gx, oy - gy)
                heapq.heappush(queue, (new_cost + estimate, new_cost, other))
        path = []
        while nearest != start:
            path.append(self.points[nearest])
            nearest = parent[nearest]
        return path


class RouteFollower:
    def __init__(self, campus):
        self.campus = campus
        self.goal = None
        self.path = []
        self.retry = 0.0

    def next(self, pos, goal, now):
        if self.campus is None:
            return goal
        if math.dist(pos[:2], goal[:2]) < 9:
            return goal
        node = self.campus.node(goal)
        while self.path:
            point = self.path[-1]
            if math.dist(pos[:2], point[:2]) > 3.4 or pos[2] < point[2] - 2:
                break
            self.path.pop()
        if (
            self.goal != node or not self.path or math.dist(pos[:2], self.path[-1][:2]) > 15
        ) and now >= self.retry:
            self.path = self.campus.route(self.campus.node(pos), node)
            self.goal, self.retry = node, now + 0.55
        return self.path[-1] if self.path else goal
