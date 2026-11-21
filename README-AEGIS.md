# Aegis Skyline

A Python pilot for the Builders Mech Royale track. The root Dockerfile starts `my_bot.py` with the organizer's `mechfight.py` transport. The pilot uses the Python standard library and the arena supplied at match time.

## Strategy

- `navigation.py` reads the terrain and building footprints at startup. It finds walking and jet routes across roofs and reserves enough energy to clear a ledge.
- `my_bot.py` chooses among the next circle, equipment, cubes, supply drops and shelter. It remembers loot while its torso turns away.
- `combat.py` selects exposed targets, leads arm-gun rounds, coordinates both triggers and tests mortar arcs against terrain.
- `evasion.py` compares movement commands against incoming projectiles, taking acceleration, torso aim and the mech's current size into account.

## Run the official check

From this repository's root, set `KIT` to the extracted organizer kit. Keep the Python modules together when copying the bot.

```sh
KIT="$HOME/mech-royale/mechfight-kit"
docker build --platform linux/amd64 -t aegis-skyline .
python3 "$KIT/check/check.py" --map-dir /arena -- \
  docker run --rm -i --platform linux/amd64 --network none --cpus 1 \
    --memory 512m --read-only --pids-limit 128 --cap-drop ALL \
    --tmpfs /tmp:rw,size=64m --user 65534:65534 \
    -v "$KIT/arena:/arena:ro" aegis-skyline
```

The check sends 300 recorded observations. The official practice score comes from a separate run of 48 matches. [Track and scoring rules](https://www.joinbuilders.org/buildathon/tracks/builders)
