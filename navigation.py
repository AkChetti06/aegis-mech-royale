"""Ground routes through the campus, shared across observations in a match."""
import heapq
import math
import os
import sys
from array import array


class GroundRoutes:
    step = 8

    def __init__(self, cover, world, directory):
        terrain = world["terrain"]
        heights = array("h")
        with open(os.path.join(directory, "v2", terrain["file"]), "rb") as stream:
            heights.fromfile(stream, terrain["nx"] * terrain["ny"])
        if sys.byteorder != "little":
            heights.byteswap()

        def ground(x, y):
            i = round((x - terrain["minX"]) / terrain["step"])
            j = round((y - terrain["minY"]) / terrain["step"])
            return heights[j * terrain["nx"] + i] * terrain["scale"]

        bounds = world["bounds"]
        spacing = self.step
        self.graph = {}
        for i in range(math.ceil((bounds["minX"] + 3) / spacing),
                       math.floor((bounds["maxX"] - 3) / spacing) + 1):
            for j in range(math.ceil((bounds["minY"] + 3) / spacing),
                           math.floor((bounds["maxY"] - 3) / spacing) + 1):
                x, y = i * spacing, j * spacing
                if not cover.blocked(x, y, ground(x, y), 3.0):
                    self.graph[i, j] = []
        for node, neighbors in self.graph.items():
            i, j = node
            for a, b in ((1, 0), (0, 1), (1, 1), (1, -1)):
                other = i + a, j + b
                if other not in self.graph:
                    continue
                if a and b and ((i + a, j) not in self.graph or (i, j + b) not in self.graph):
                    continue
                x, y = (i + a / 2) * spacing, (j + b / 2) * spacing
                if cover.blocked(x, y, ground(x, y), 3.0):
                    continue
                cost = math.hypot(a, b)
                neighbors.append((other, cost))
                self.graph[other].append((node, cost))
        self.path = []
        self.destination = None
        self.retry = 0

    def nearest(self, point):
        i, j = (round(n / self.step) for n in point[:2])
        if (i, j) in self.graph:
            return i, j
        nearby = ((a, b) for a in range(i - 8, i + 9) for b in range(j - 8, j + 9)
                  if (a, b) in self.graph)
        return min(nearby, key=lambda p: math.hypot(p[0] * self.step - point[0],
                                                   p[1] * self.step - point[1]), default=None)

    def route(self, start, goal):
        frontier = [(0, 0, start)]
        parent = {start: None}
        costs = {start: 0}
        # Limit work on disconnected components under the per-observation deadline.
        for _ in range(1700):
            if not frontier:
                break
            _, cost, here = heapq.heappop(frontier)
            if here == goal:
                path = []
                while parent[here] is not None:
                    path.append((here[0] * self.step, here[1] * self.step))
                    here = parent[here]
                return path
            if cost != costs[here]:
                continue
            for other, edge in self.graph[here]:
                new_cost = cost + edge
                if new_cost < costs.get(other, math.inf):
                    costs[other] = new_cost
                    parent[other] = here
                    estimate = math.hypot(other[0] - goal[0], other[1] - goal[1])
                    heapq.heappush(frontier, (new_cost + 1.2 * estimate, new_cost, other))
        return []

    def waypoint(self, pos, destination, now):
        while self.path and math.dist(pos[:2], self.path[-1]) < 3:
            self.path.pop()
        stale = not self.path or math.dist(pos[:2], self.path[-1]) > self.step * 2
        if self.destination != tuple(destination) or (stale and now >= self.retry):
            self.destination = tuple(destination)
            start, goal = self.nearest(pos), self.nearest(destination)
            self.path = self.route(start, goal) if start and goal else []
            self.retry = now + 1.0
        return self.path[-1] if self.path else destination
