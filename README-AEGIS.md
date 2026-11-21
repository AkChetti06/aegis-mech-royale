# Aegis v1 — survival-focused Mech Royale bot

Aegis is a Python `mechfight/1` bot built to replace the example strategy. It uses only Python's standard library plus the official `mechfight.py` wrapper already in the downloaded kit.

## Features

- Current/next safe-zone navigation, prioritizing escape from zone damage
- Escape steering from visible enemy squads, projectile closest-approach prediction and sound bearings
- Opportunistic pickups, cubes and supply drops, plus stuck recovery
- Predictive straight-shot aiming for the gun; low-arc ballistic solution for mortar
- Laser hold-to-charge and release logic; close-range lightning gating
- Defensive upgrade selection and energy-aware boosting
- Adaptive attack direction when damage arrives from outside line of sight

## Installing onto the original starter kit (on your Mac)

From inside the original `mechfight-kit/` folder:

```sh
unzip -o ~/Downloads/aegis-mech-bot.zip -d python/
python3 check/check.py -- python3 -u python/my_bot.py
```

If the kit is somewhere else, change directory to that folder first. The zip overwrites `python/Dockerfile` but leaves the original `example_bot.py` and `mechfight.py` unchanged.

## Test the actual Docker runtime (requires Docker Desktop)

```sh
docker build --platform linux/amd64 -t aegis-bot ./python
python3 check/check.py --map-dir /arena -- \
  docker run --rm -i --platform linux/amd64 \
  --network none --cpus 1 --memory 512m --read-only \
  --tmpfs /tmp:rw,size=64m --user 65534:65534 \
  -v "$PWD/arena:/arena:ro" aegis-bot
```

If successful, copy the *contents* of `python/` to a public GitHub repository root with `Dockerfile` at its root, and submit the repository URL to the official Builders Mech Royale track. Keep the submitted commit accessible.

## Important validation notes

- The author environment did **not** have the starter kit's `mechfight.py` wrapper, `check/check.py` or map. The official conformance checker and real-game score have not yet been run on this build.
- The Python module compiled and its actions passed 1,500 *synthetic* observations against the JSON action schema supplied in the uploaded zip, with no malformed actions. This is not a match simulation and does not prove competitiveness or official conformance.
- The bot has reactive rather than map-based navigation and may struggle with complex obstacles, roofs, bridges, moving targets near hard cover and team tactics. It should be improved based on real practice feedback.
- Do not send any model API requests at runtime: the match container has no network and a 50 ms decision budget.
