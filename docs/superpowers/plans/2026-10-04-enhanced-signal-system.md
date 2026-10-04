# Enhanced Signal System v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform `coin_scanner.py` into a multi-timeframe, regime-aware crypto scanning and signaling engine with enhanced Telegram notifications, position sizing, risk management, and outcome tracking.

**Architecture:** Extend existing provider and math infrastructure in `coin_scanner.py`. Add multi-timeframe candle fetching (HTF for trend bias, LTF for entry timing), regime detection (trend direction + volatility state), confluence-based ranking, and rich Telegram signal formatting with multiple TP targets, entry zones, and expiry parameters.

**Tech Stack:** Python 3, NumPy, Pandas, Requests, ThreadPoolExecutor.

**Spec:** In-chat approved design for Enhanced Signal System v2.

## Global Constraints

- Preserve all existing command-line arguments (`--interval`, `--bars`, `--top`, `--min-vol`, `--horizon`, `--fee`, `--risk`, `--show`, `--telegram`, `--send`, `--log`, `--cooldown`, `--max-signals`, `--sources`, `--demo`, `--csv`).
- Output format must strictly maintain Telegram Markdown compatibility.
- Free tier hides Top 1-5 watchlist entries; Pro tier shows all Top 1-10 entries.
- Verifiable with standalone Python unit/integration tests using standard library `unittest` or direct script execution.

---

### Task 1: Multi-Timeframe Data Fetcher & Regime Detector

**Files:**
- Modify: `C:\Users\User\Documents\files\coin_scanner.py`
- Create: `C:\Users\User\Documents\files\test_scanner_v2.py`

**Interfaces:**
- Consumes: `Provider.klines(base, interval, bars)`
- Produces: `fetch_multi_tf(universe, htf, ltf, bars)` -> dict of `{symbol: {"htf": d_htf, "ltf": d_ltf}}`, `detect_regime(logp_htf, r_htf, logp_ltf, r_ltf)` -> `dict(trend_state, vol_state, regime_score)`

- [ ] **Step 1: Write unit tests for regime detection & multi-tf handling in test_scanner_v2.py**

```python
import unittest
import numpy as np
from coin_scanner import detect_regime

class TestRegimeDetector(unittest.TestCase):
    def test_detect_regime_bullish(self):
        logp = np.linspace(4.0, 4.5, 200)
        r = np.diff(logp)
        res = detect_regime(logp, r, logp, r)
        self.assertIn("trend", res)
        self.assertIn("vol", res)
        self.assertIn("score", res)
        self.assertEqual(res["trend"], "BULLISH")

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify failure**

Run: `python C:\Users\User\Documents\files\test_scanner_v2.py`
Expected: FAIL with `ImportError: cannot import name 'detect_regime'`

- [ ] **Step 3: Implement `detect_regime` and multi-TF support in `coin_scanner.py`**

Add `detect_regime` function combining Kalman slope, EWMA volatility percentile, and Hurst exponent:

```python
def detect_regime(logp_htf, r_htf, logp_ltf, r_ltf):
    slope_htf, pvar_htf = kalman_trend(logp_htf, r_htf)
    t_htf = slope_htf / np.sqrt(pvar_htf + 1e-12)
    h_htf = hurst(logp_htf)
    vol_htf = ewma_vol(r_htf)
    
    if t_htf > 0.5:
        trend = "BULLISH"
    elif t_htf < -0.5:
        trend = "BEARISH"
    else:
        trend = "SIDEWAYS"
        
    vol_state = "HIGH_VOL" if vol_htf > 0.02 else "NORMAL_VOL"
    regime_score = float(np.clip((t_htf / 3.0) * (h_htf / 0.5), -1.0, 1.0))
    
    return {
        "trend": trend,
        "vol": vol_state,
        "score": regime_score,
        "t_htf": float(t_htf),
        "hurst_htf": float(h_htf)
    }
```

- [ ] **Step 4: Run test to verify pass**

Run: `python C:\Users\User\Documents\files\test_scanner_v2.py`
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add coin_scanner.py test_scanner_v2.py
git commit -m "feat: add regime detection and multi-timeframe analytics"
```

---

### Task 2: Multi-TP & Enhanced Signal Generator

**Files:**
- Modify: `C:\Users\User\Documents\files\coin_scanner.py`
- Modify: `C:\Users\User\Documents\files\test_scanner_v2.py`

**Interfaces:**
- Consumes: `entry_plan(mu, sigma, horizon, fee, rng)`
- Produces: `format_signal_v2(r, htf, ltf, horizon, risk, sid)` -> string formatted for Telegram

- [ ] **Step 1: Write test for enhanced signal formatter**

```python
def test_format_signal_v2(self):
    from coin_scanner import format_signal_v2
    import pandas as pd
    row = pd.Series({
        "symbol": "BTC",
        "close": 67000.0,
        "entry": 66800.0,
        "entry_deep": 66200.0,
        "sl": 65500.0,
        "tp1": 68500.0,
        "tp2": 70000.0,
        "tp3": 72000.0,
        "rr": 2.6,
        "alloc": 0.02,
        "fill_bars": 6,
        "p_tp": 0.42,
        "p_sl": 0.28,
        "ev_trade": 0.018,
        "trend_regime": "BULLISH",
        "vol_regime": "NORMAL_VOL"
    })
    msg = format_signal_v2(row, "4h", "1h", 24, 0.01, "20261004-BTC")
    self.assertIn("TP1:", msg)
    self.assertIn("HTF 4h / LTF 1h", msg)
```

- [ ] **Step 2: Run test to verify failure**

Run: `python C:\Users\User\Documents\files\test_scanner_v2.py`
Expected: FAIL with `ImportError: cannot import name 'format_signal_v2'`

- [ ] **Step 3: Implement multi-TP calculation and `format_signal_v2` in `coin_scanner.py`**

Extend `entry_plan` to return `tp1`, `tp2`, `tp3` based on volatility multiples and compute `format_signal_v2`:

```python
def format_signal_v2(r, htf, ltf, horizon, risk, sid):
    f = lambda x: f"{x:.6g}"
    sl_pct = (r.sl / r.entry - 1) * 100
    tp1_pct = (r.tp1 / r.entry - 1) * 100
    tp2_pct = (r.tp2 / r.entry - 1) * 100
    tp3_pct = (r.tp3 / r.entry - 1) * 100
    return (
        f"📊 SIGNAL {sid}\n"
        f"{r.symbol}/USDT · SPOT · LONG · HTF {htf} / LTF {ltf}\n\n"
        f"🎯 Entry Zone: {f(r.entry)} (limit)\n"
        f"   Deep Entry: {f(r.entry_deep)} (25% fill prob)\n"
        f"🛑 Stop Loss: {f(r.sl)} ({sl_pct:+.1f}%)\n"
        f"✅ TP1: {f(r.tp1)} ({tp1_pct:+.1f}%) | TP2: {f(r.tp2)} ({tp2_pct:+.1f}%) | TP3: {f(r.tp3)} ({tp3_pct:+.1f}%)\n"
        f"⚖️ Risk:Reward = 1 : {r.rr:.2f}\n\n"
        f"📈 HTF Bias: {r.trend_regime} | Vol: {r.vol_regime}\n"
        f"💼 Position sizing: max risk {risk*100:.1f}% capital (alloc ~{r.alloc*100:.1f}%)\n"
        f"⏳ Valid: {horizon} bars ({ltf}) | Cancel limit if unfilled in {int(r.fill_bars)} bars.\n\n"
        f"📊 Model Estimate: P(TP) {r.p_tp*100:.0f}% · P(SL) {r.p_sl*100:.0f}% · EV {r.ev_trade*100:+.2f}%\n\n"
        f"⚠️ Model estimate, not financial advice. Practice risk management."
    )
```

- [ ] **Step 4: Run test to verify pass**

Run: `python C:\Users\User\Documents\files\test_scanner_v2.py`
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add coin_scanner.py test_scanner_v2.py
git commit -m "feat: implement multi-TP entry planning and enhanced Telegram signal formatting"
```

---

### Task 3: Portfolio View & Integration Testing

**Files:**
- Modify: `C:\Users\User\Documents\files\coin_scanner.py`
- Modify: `C:\Users\User\Documents\files\test_scanner_v2.py`

**Interfaces:**
- Consumes: Scanned DataFrame with all calculated signals
- Produces: Integrated execution path with CLI flags and complete end-to-end signal dispatching

- [ ] **Step 1: Write integration test for main pipeline**

```python
def test_full_pipeline_demo(self):
    import os, subprocess
    cmd = ["python", "coin_scanner.py", "--demo", "--send", "--telegram"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    self.assertEqual(res.returncode, 0)
    self.assertIn("TOP 15", res.stdout)
```

- [ ] **Step 2: Run test to verify failure/pass status**

Run: `python C:\Users\User\Documents\files\test_scanner_v2.py`

- [ ] **Step 3: Update `analyze_all` and `main` in `coin_scanner.py` to seamlessly execute the new pipeline**

Ensure `analyze_all` attaches `tp1`, `tp2`, `tp3`, `trend_regime`, `vol_regime` and uses `format_signal_v2` for Telegram dispatches.

- [ ] **Step 4: Run full test suite**

Run: `python C:\Users\User\Documents\files\test_scanner_v2.py`
Expected: ALL PASS

- [ ] **Step 5: Final Commit**

```bash
git add coin_scanner.py test_scanner_v2.py
git commit -m "feat: finalize Enhanced Signal System v2 integration"
```
