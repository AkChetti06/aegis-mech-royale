# Aegis v3: a Builders Mech Royale bot

A `mechfight/1` bot for the [Builders Mech Royale track](https://www.joinbuilders.org/buildathon/tracks/builders). It is pure Python standard library, built on the kit's `mechfight.py` wrapper.

## Strategy

The score is the mean of `16 - place`. In practice the 15 house bots act as one squad that hunts you, so a house bot only dies before you if somebody kills it. Placement therefore comes from winning fights while the hunters arrive one or two at a time. Running away doesn't help.

- **Map model**: at startup the bot rasterizes `/arena/v2/world.json` and `terrain.i16` (about 0.15 s) to check line of sight, find cover and steer around walls. Without the map it falls back to plain steering.
- **Firing**: gun and laser fire together. The gun uses a lead solution, the laser fires on a full charge, and every trigger waits until both yaw and pitch are on target. The gun never fires along a mortar's lobbed arc.
- **Focus fire**: the bot sticks to the weakest, nearest target until it dies, so enemies don't escape and regenerate.
- **Fight positioning**: every 0.5 s it picks a nearby spot that has line of sight to the target and is hidden from the other visible mechs. It keeps a weapon-appropriate range (long with a laser) and jinks irregularly to spoil the enemy's lead.
- **Facing**: it always faces the threat, because hits from behind deal 1.5× damage. With no visible target it turns toward the last damage bearing or a remembered hunter.
- **Retreat**: when hurt or outnumbered it moves to cover that breaks line of sight and regenerates. It stops retreating early if the target is nearly dead.
- **Economy**: it hunts for a laser first, farms crates and drones for cubes, captures supply drops, and picks defensive upgrades (armor, plating, laser damage and charge).
- **Safety**: zone handling plans travel time, it dodges mortar landing points and bullets, it jumps and jets when stuck, and it avoids crowds.
- **Timing**: decisions take about 2 ms (p99) on the checker. The expensive spot searches stop at 14 ms, well inside the 50 ms budget.

## Repository layout

| Path | What |
| --- | --- |
| `my_bot.py` | The bot (entrypoint) |
| `mechfight.py` | The kit's protocol wrapper (unchanged) |
| `Dockerfile` | The kit's Python image, entrypoint `my_bot.py` |
| `tools/sim.py` | An offline approximation of practice mode for A/B tests (not shipped in the image) |
| `tools/config.json` | The match config from the kit's recorded `hello` |

## Testing

Conformance (from the kit folder):

```sh
python3 check/check.py --map-dir "$PWD/arena" -- python3 -u /path/to/my_bot.py
```

Offline A/B against a scripted hunter squad on the real map (needs `numpy`; point `MECH_ARENA` at the kit's `arena/`):

```sh
MECH_ARENA=/path/to/mechfight-kit/arena python3 tools/sim.py --bot my_bot.py --matches 48 --jobs 4
```

`tools/sim.py` is **not** the official server. Its house bots are a guess at the described behavior: one hunter early, more as you kill them, all of them late. Use it to compare versions against each other, not to predict the leaderboard score. On 48 simulated matches, the kit's example bot scores 1.4 (official baseline 1.15), Aegis v2 scores 1.9 and Aegis v3 scores 2.9–3.1.
