# How the mahiro_oyama bot makes decisions

A map of the full decision path, the data behind each input, the learning loops, and what
the 2026-06-03 fixes changed. Paper-trading bot, $10k start, $0.99 flat commission/trade,
runs every ~2–10 min while NYSE is open.

---

## 1. Data sources (and what's dead in production)

| Input | Source | Status on PythonAnywhere |
|---|---|---|
| Live quote | Finnhub `quote` → yfinance → last recorded snapshot | Finnhub works |
| Daily bars (indicators) | PA: **Finnhub candles** + live Finnhub bar → recorded; off-PA: yfinance → Stooq | Finnhub candles carry PA free tier |
| News sentiment | Finnhub `company_news` → yfinance, scored by VADER + finance lexicon | Finnhub works |
| Analyst / insider | Finnhub `recommendation_trends` / `stock_insider_sentiment` | works |
| Earnings calendar | Finnhub `earnings_calendar` | works |
| Market regime | SPY daily trend + **intraday tilt** + credit spread (HYG:IEI) | **intraday tilt dead** (yfinance 15m blocked) |
| Volatility gate | SPY 20d realized vol from daily bars (Finnhub candles on PA) | works via Finnhub candles |

**2026-06-07 PA hardening:** PythonAnywhere free accounts use Finnhub daily candles instead of
Stooq, because Stooq is not on the free outbound allowlist and may also return an API-key/captcha
wall. `_stooq_daily` now preserves the old interface but delegates to Finnhub candles in
`PYTHONANYWHERE_MODE`.

Stale quotes are explicitly flagged (`stale=True`); the bot **halts all buy/sell on a stale
ticker** and values it at avg-cost so the equity chart never draws a $0 spike.

---

## 2. The per-tick pipeline

```
run_bot()  (lock: one pass at a time)
  ├─ market closed? → queue a resume, return
  ├─ interval cooldown (2 min, bypassed by force/resume)
  ├─ fetch in parallel for watchlist ∪ holdings: news, daily ctx, intraday, earnings, analyst, insider, quote
  ├─ get_market_regime() + get_vix()            → regime params, size multipliers, buy halt
  ├─ SELL pass   (every holding)                → §6 exit ladder
  ├─ BUY pass    (watchlist BUY/STRONG-BUY)     → §4 gates, §5 sizing
  ├─ OUTSIDE BUY (market-wide scan leaders)     → same gates, ≥$400
  ├─ record HOLD/SKIP reasons, cap history
  └─ equity snapshot, save state
```

---

## 3. Signal generation — `get_recommendation` (signals.py)

A **categorized, weight-voted aggregator**. Each category casts integer votes in −2..+2:

- **TREND** — price vs 30-day MA, MA7/MA30 cross (only votes when it *diverges* from the
  price-vs-MA read, else it's double-counting), ADX strength, weekly higher-timeframe posture.
- **MOMENTUM** — RSI + MACD are primary; Bollinger %position and Stochastic only confirm or
  flag disagreement (an "oscillators split" uncertainty score shaves confidence).
- **VOLUME** — volume ratio × trend direction, Money Flow Index.
- **DIP** — distance below the 20-day high (a real dip votes +2; a −20%+ "falling knife" votes −1);
  trend-exhaustion (5+ green days while >5% above MA30) votes the category **down**.
- **REL_STR** — 20-day return minus the sector ETF's (XLK/XLF/…).
- **NEWS** — VADER sentiment with a **6-hour half-life decay** (epoch-accurate).
- **ANALYST** (14-day decay), **INSIDER** (21-day decay, halved), **REGIME** (SPY direction).

Each category vote = `mean(its votes) × learned_weight(category)` (see §7). Then:

- `tot` = sum of category votes; `cats_pos`/`cats_neg` = how many categories agree.
- **Regime-conditional thresholds** decide the label: in a **bull** market the buy side is
  looser and the sell side tighter; in a **bear** market the reverse; neutral is the baseline.
  (e.g. neutral STRONG-BUY needs `tot≥4.0` and ≥3 positive categories.)
- **Confidence** (15–95) scales with signal strength, category agreement, and data quality,
  then takes **penalties** for: high realized vol, low liquidity, high ATR, daily-vs-weekly
  conflict, and gap-up extension. A separate `sizing_confidence` floors STRONG signals to 55 so
  a penalty-shaved STRONG-BUY still gets a real position.

Earnings within 5 days → **forced HOLD** (binary event risk overrides everything).

---

## 4. Entry gates (must all pass to buy)

- Signal is BUY or STRONG-BUY (watchlist) / confidence ≥ 60 (outside scan).
- Not stale, price > 0, sector is known (ETFs like SPY/QQQ have no sector → never bought).
- **Regime filter:** bear → only RSI<35 or a dip; neutral → require ADX≥20 (a real trend) for
  new names; explosive volatility + RSI<35 → blocked.
- **Time-of-day:** new buys only inside 09:45–15:30 ET (skips the volatile open/close). *Only a
  human "run now" bypasses this now — the after-close auto-resume no longer does (A3).*
- **Cooldown:** a just-sold name is benched (2h if the exit was a loss, 30 min otherwise). A
  fresh STRONG-BUY ≥80% can jump a *non-loss* cooldown; loss cooldowns are never bypassed.
- **Caps:** ≤10 concurrent positions; ≤35% equity per position; ≤55% per sector; ≤45% per
  correlation group (e.g. AI-semis); 2% cash reserve; drawdown >12% from peak pauses all buys.

---

## 5. Position sizing — a 5-factor stack

```
spend = spendable_cash × weight × pos_size × win_rate × env × ticker
```

1. **weight** — this pick's `sizing_confidence` share across the cycle's picks.
2. **pos_size** — 0.7 for a new name, 1.0 for a pyramid add.
3. **win_rate** — **Kelly** multiplier (§7), clamped 0.10–1.5.
4. **env** — regime size (bull 1.0 / neutral 0.8 / bear 0.5) × realized-vol gate (×0.5 high-risk,
   ×0 panic) × cold-streak downsizer.
5. **ticker** — dip/overbought tilt (strong dip +50%, dip +25%, dying-momentum overbought −50%),
   with explosive-vol −30% folded in.

The result must clear **`MIN_POSITION_USD = $400`** (A2) — below that, the $1.98 round-trip
commission eats >0.5% of the trade. Pyramiding adds only into a winner (≥+3%) after a 60-min
cooldown, and must be ≥$400 too. Skips are logged with the full multiplier breakdown.

---

## 6. Exit ladder (SELL pass, evaluated top-down)

Per holding, using the shared, unit-tested math in **`trading/exits.py`**:

1. **Hard stop** — ATR-scaled (`−2×ATR`, clamped −1.5%..−15%), or regime-static if no ATR.
   Tightens to ½ after 2h of going nowhere underwater.
2. **Cost-aware ratchet (A1)** — *replaced the old breakeven*. Arms once peak P&L ≥ +3%
   (later for tiny positions), then locks `max(round-trip-cost + 0.3%, ½ × peak)`. The lock
   ratchets up as the peak grows and is **always net-positive after commissions**, so winners
   keep running on the trail above the lock instead of being strangled at +0.1%. Tagged
   `ratchet` (a profit exit, short cooldown) — not `loss`.
3. **Trailing stop** — width = 2× intraday ATR, or **daily ATR when intraday is dead (A5)**,
   or regime-static; only active once up >2%; tightens over time on long winners.
4. **Signal flip** — turned SELL while losing → exit; flipped STRONG-SELL while green → take it.
5. **Trend failure** — closed >2% below the 30-day MA while underwater → exit.
6. **Aging** — held past the regime's age limit, flat, and ≥$200 → free the capital.

Min-hold 20 min gates 3–6 (not the hard stop). Realized P&L recorded **net of commission (A4)**.

---

## 7. Learning loops

- **Bayesian category weights** — every closed trade updates a Beta(α,β) posterior per
  `(regime, category)`; weight = `0.5 + P(edge>0.5)`, clamped 0.5–1.5. This is what makes a
  category vote count for more or less over time. (A6 fixed a first-trade double-count.)
- **Kelly** — half-Kelly from the last 50 **same-regime** net outcomes, clamped 0.10–1.5.
- **Cold-streak breaker** — 3+ losses in the last 5 closes → extra downsizing.
- **Confidence calibration** — `/api/attribution` buckets closed trades by entry-confidence and
  reports win-rate per bucket (ideally monotonic). All four now consume **net** P&L (A4), so
  they no longer overstate edge by the commission.

---

## 8. What changed this round and why it adds EV (not noise)

Every change removes value-destroying behavior or de-biases the learning loop — **no new
signals, no threshold curve-fitting:**

| # | Fix | Why it adds EV |
|---|---|---|
| A1 | Cost-aware ratchet replaces the +0.1% breakeven | Old rule sold winners at a **net loss** (<$2k) and capped every runner at a 1.4pp give-back. Ratchet lets winners run, never locks a loss. |
| A2 | $400 minimum position (new + outside) | $50 positions paid 4% round-trip commission — structurally unprofitable churn. |
| A3 | `user_forced` split from `force` | Auto-resume was force-buying a marginal name in the volatile first 15 min of every session. |
| A4 | Net-of-commission outcome P&L | Kelly/win-rate/calibration were learning from gross P&L → overstated edge → mis-sized. |
| A5 | Daily-ATR trail fallback | The dynamic trail was dead on PA (intraday ATR always 0); now volatility-scaled instead of flat. |
| A6 | α/β seed-order fix | First trade per category was double-counted in the Bayesian weight. |

---

## 9. How to measure it — `GET /api/backtest/portfolio`

Walk-forward portfolio backtest (`trading/backtest.py`): runs the **same** entry signal, exit
ladder (via `trading/exits.py`), sizing, caps, and $0.99 commission over multi-year daily bars;
trains the category weights on the first half, freezes them, evaluates on the held-out half, and
reports **`fixed` (post-fix) vs `legacy` (pre-fix) vs `fixed_default_weights`**, plus alpha vs
buy-hold SPY, max drawdown, Sharpe, profit factor, and an exit-reason breakdown.

```
/api/backtest/portfolio?years=4&train_frac=0.5&universe=scan      # scan universe (default)
/api/backtest/portfolio?years=4&universe=watchlist               # your watchlist
```
First run ~1 min (cached 1h). yfinance-first data, Stooq fallback.

**The honest result (pure-technical, 2024–26 bull window):** both rule-sets **trail SPY
buy-hold by ~30 percentage points.** A long-only technical-timing system underperforms simply
holding the index in a strong bull — the efficient-market null is real and is the benchmark to
beat. Within that, **`fixed` beats `legacy` on every axis**: ~75% fewer trades, lower drawdown,
less-negative return, and ratchet exits average net-positive where the old breakeven averaged
net-negative. The fixes make the machine leaner and less self-defeating; they do not manufacture
alpha that the technical signals don't contain.

**Caveat:** pure-technical only (news/analyst/insider/intraday aren't reconstructable
historically); daily granularity approximates the intraday min-hold/aging thresholds;
partial-take/degrade-trim are omitted (identical across both modes).

---

## 10. Bottom line

The bot is a disciplined, multi-signal, regime-aware, self-tuning **paper** trader with honest
risk controls. This round fixed the ways it was leaking money to itself (commission churn,
winner-strangling breakeven, biased learning) and — most importantly — shipped the measurement
tool that proves whether any future change actually helps. The current evidence says the
technical layer alone does not beat buy-and-hold; the right next experiments are measured against
this harness, not shipped blind.
