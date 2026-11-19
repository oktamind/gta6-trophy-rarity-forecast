# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/oktamind/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/oktamind/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Refresh-aware earners-over-players forecaster for GTA VI PS5 trophy earn rates on Exophase. A rate is earners / players, so the program forecasts the two counts separately and divides, instead of extrapolating the rate: players grow as a power law in log-time since release, each trophy's earners follow their own shrunk log-time slope (rare trophies accumulate slowly and dilute as the player base multiplies), and the spread is a logit-normal fan that widens with horizon, staleness (a flat history means a pending refresh of unknown size), and small-count noise. TimesFM was checked first and cannot run on the build VM (~1 GB RAM free vs its 2 GB threshold), so the entry is a statistical forecaster, validated on the practice cases and on a synthetic power-law case where it recovers the true rates within ~6%.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=rrts6br6&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
