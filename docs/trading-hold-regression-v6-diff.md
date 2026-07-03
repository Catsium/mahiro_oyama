# Trading HOLD Regression: V6 Comparison

Date: 2026-06-24

This patch compared the current trading paths against the supplied V6 archive:

`C:\Users\euris\Documents\Projects\Mahiro Oyama\Mahiro Oyama Ver\mahiro_oyama V6 6.6.26 latest working.zip`

## Findings

- V6 hid volatility failures by returning `NORMAL`/`mult=1.0` when VIX/SPY-vol data was unavailable.
- Current code already made some failures explicit, but still allowed missing SPY data to downsize rather than block new buys.
- The main HOLD regression was not one single gate. It was a combination of hidden data-health ambiguity, weak raw buy labels rendering as `BUY`, old non-dip volume hard rejection at `<1.2`, and insufficient pre-candidate rejection diagnostics.
- PythonAnywhere decisions still apply: do not restore direct Stooq as a PA primary path. The data-manager/Finnhub adapter remains the PA-safe history path.

## Implemented Rules

- Missing SPY regime data, missing volatility data, or stale held quotes block new buys and appear in `data_health_blocks`.
- Raw buy labels below 40% display as `BULLISH_LEAN`; 40-69% displays as `BUY_CANDIDATE`; 70%+ displays as `STRONG_BUY_CANDIDATE`.
- Non-dip `vol_ratio < 0.70` hard rejects with `VERY_LOW_VOLUME_CONFIRMATION`.
- Non-dip `0.70 <= vol_ratio < 1.20` applies `LOW_VOLUME_PENALTY_ONLY` with size multiplier `0.75`.
- `/api/bot/status` exposes raw/display counts, rejection counts, top pre-candidate rejects, top ranked rejects, provider health, stale/risk-unmanaged positions, buy caps, exposure, and tick runtime.
- `/bot/tick` is the admin-only synchronous execution endpoint; `/health` remains lightweight.

## Verification

- `python -m compileall trading market routes utils tests`
- `python -m unittest tests.test_trading_v1 -v`
