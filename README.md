# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/KuntoSadewo/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/KuntoSadewo/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI PlayStation 5 trophy's earn rate on Exophase at 2026-11-26 00:00 UTC, as five quantiles (5/25/50/75/95%) per trophy, from the house's recorded reads up to the case checkpoint. Entry for make.fun competition `gta6-trophy-rarity-forecast-01`.

## Method

Per trophy, in logit space:

1. **Earners vs. audience, separately.** Earners E = rate × players tracked / 100 and the tracked audience P are each fitted as straight lines of ln E and ln P against ln(1 + hours since release) over the last 24 h of reads, then read at the target. The rate forecast is E*/P* — so a growing audience dilutes the rate while progressing players lift it. A second fit, logit(rate) against ln(1 + hours), is blended in (60/40), and the blend is shrunk toward the last observed rate with a weight that grows with the horizon.
2. **Honest spread.** Residual scatter of the logit fit plus a random-walk term growing with the square root of the horizon in days, so a 4.5-day target admits a wider band than a 90-minute one.
3. **Stale page.** When every read in the history is identical, Exophase has not refreshed since before the first read, so a refresh may land before the target. The forecast becomes a 50/50 mixture of "no refresh" and "refresh dilutes the rate" (the one refresh observed so far multiplied the tracked audience ~6x; newcomers earn an easy trophy quickly and a rare one barely at all, so the diluted rate lets their earn share grow with the rate itself). Quantiles come from the mixture CDF.

Deterministic: no randomness, no network, no clock. Node 22, standard library only.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=5m9k5b20&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
