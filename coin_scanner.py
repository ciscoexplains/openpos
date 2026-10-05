#!/usr/bin/env python3
"""
Coin Scanner multi-exchange - ranking koin berdasarkan edge statistik (bukan jaminan profit).

SUMBER DATA (dicek otomatis, yang diblokir ISP dilewati):
  Binance (+host cadangan & data-api.binance.vision), MEXC, Bybit, OKX, KuCoin, Gate.io, Indodax

METODE MATEMATIKA:
  1. Log-return + EWMA volatility (RiskMetrics, lambda=0.94)
  2. Kalman filter (local linear trend) -> estimasi drift + ketidakpastiannya
  3. Shrinkage Empirical-Bayes (James-Stein) -> drift ditarik ke rata-rata pasar
  4. Hurst exponent -> rezim trending vs mean-reverting
  5. Monte Carlo Student-t (fat tails) -> P(untung), EV, CVaR 5% setelah fee
  6. Sortino, max drawdown, half-Kelly
  7. Skor komposit dari cross-sectional z-score (winsorized)

PEMAKAIAN:
  pip install numpy pandas requests
  python coin_scanner.py                         # semua sumber, 1h, top 40
  python coin_scanner.py --interval 4h --top 60 --horizon 12
  python coin_scanner.py --sources bybit,okx     # pilih sumber tertentu
  python coin_scanner.py --demo                  # data sintetis (tanpa internet)
"""
import argparse
import sys
import time
import traceback


def _pause():
    """Tahan jendela supaya tidak langsung menutup saat di-double-click."""
    try:
        input("\nTekan Enter untuk keluar...")
    except EOFError:
        pass


try:
    import numpy as np
    import pandas as pd
    import requests
except ImportError as e:
    print(f"Paket belum terinstall: {e}\nJalankan:  pip install numpy pandas requests")
    _pause()
    sys.exit(1)

from concurrent.futures import ThreadPoolExecutor, as_completed

UA = {"User-Agent": "Mozilla/5.0 (coin-scanner)"}
INTERVAL_SEC = {"15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}
STABLES = {"USDT", "USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "USDE", "EUR", "EURI",
           "AEUR", "UST", "XUSD", "USD1", "PYUSD", "RLUSD", "USDD", "GUSD"}
WRAPPED = {"WBTC", "WBETH", "STETH", "WSTETH", "WEETH", "CBBTC", "RETH", "BETH"}
MIN_BARS = 120


# =========================================================================== DATA LAYER
class Provider:
    name = ""
    bases = []
    caps = 500
    imap = {}

    def __init__(self):
        self.ok_base = None

    def get(self, path, params=None):
        """GET dengan fallback antar-host. Verifikasi SSL tetap aktif."""
        bases = ([self.ok_base] if self.ok_base else []) + [b for b in self.bases if b != self.ok_base]
        errs = []
        for b in bases:
            try:
                r = requests.get(b + path, params=params, headers=UA, timeout=12)
                r.raise_for_status()
                j = r.json()
                self.ok_base = b
                return j
            except Exception as e:
                errs.append(f"{b.split('//')[1]}: {type(e).__name__}")
        raise RuntimeError("; ".join(errs))

    def tickers(self):
        """-> {BASE: volume_24h_dalam_USDT}"""
        raise NotImplementedError

    def klines(self, base, interval, bars):
        """-> (close[], quote_volume[]) urut dari lama ke baru"""
        raise NotImplementedError


class BinanceLike(Provider):
    """Binance & MEXC (format sama)."""

    def __init__(self, name, bases, imap, caps):
        super().__init__()
        self.name, self.bases, self.imap, self.caps = name, bases, imap, caps

    def tickers(self):
        d = self.get("/api/v3/ticker/24hr")
        return {x["symbol"][:-4]: float(x["quoteVolume"]) for x in d if x["symbol"].endswith("USDT")}

    def klines(self, base, interval, bars):
        k = self.get("/api/v3/klines", {"symbol": base + "USDT", "interval": self.imap[interval],
                                        "limit": min(bars, self.caps)})
        return (np.array([float(x[4]) for x in k]), np.array([float(x[7]) for x in k]))


class Bybit(Provider):
    name = "Bybit"
    bases = ["https://api.bybit.com", "https://api.bytick.com"]
    imap = {"15m": "15", "30m": "30", "1h": "60", "4h": "240", "1d": "D"}
    caps = 1000

    def tickers(self):
        j = self.get("/v5/market/tickers", {"category": "spot"})
        return {x["symbol"][:-4]: float(x.get("turnover24h") or 0)
                for x in j["result"]["list"] if x["symbol"].endswith("USDT")}

    def klines(self, base, interval, bars):
        j = self.get("/v5/market/kline", {"category": "spot", "symbol": base + "USDT",
                                          "interval": self.imap[interval], "limit": min(bars, self.caps)})
        rows = j["result"]["list"][::-1]  # API: terbaru dulu
        return (np.array([float(x[4]) for x in rows]), np.array([float(x[6]) for x in rows]))


class OKX(Provider):
    name = "OKX"
    bases = ["https://www.okx.com", "https://aws.okx.com"]
    imap = {"15m": "15m", "30m": "30m", "1h": "1H", "4h": "4H", "1d": "1D"}
    caps = 300

    def tickers(self):
        j = self.get("/api/v5/market/tickers", {"instType": "SPOT"})
        return {x["instId"][:-5]: float(x.get("volCcy24h") or 0)
                for x in j["data"] if x["instId"].endswith("-USDT")}

    def klines(self, base, interval, bars):
        j = self.get("/api/v5/market/candles", {"instId": base + "-USDT", "bar": self.imap[interval],
                                                "limit": min(bars, self.caps)})
        rows = j["data"][::-1]
        return (np.array([float(x[4]) for x in rows]),
                np.array([float(x[7]) if len(x) > 7 else float(x[6]) for x in rows]))


class KuCoin(Provider):
    name = "KuCoin"
    bases = ["https://api.kucoin.com"]
    imap = {"15m": "15min", "30m": "30min", "1h": "1hour", "4h": "4hour", "1d": "1day"}
    caps = 1500

    def tickers(self):
        j = self.get("/api/v1/market/allTickers")
        return {x["symbol"][:-5]: float(x.get("volValue") or 0)
                for x in j["data"]["ticker"] if x["symbol"].endswith("-USDT")}

    def klines(self, base, interval, bars):
        now = int(time.time())
        n = min(bars, self.caps)
        j = self.get("/api/v1/market/candles", {"symbol": base + "-USDT", "type": self.imap[interval],
                                                "startAt": now - n * INTERVAL_SEC[interval], "endAt": now})
        rows = j["data"][::-1]
        # urutan KuCoin: [time, open, close, high, low, volume, turnover]
        return (np.array([float(x[2]) for x in rows]), np.array([float(x[6]) for x in rows]))


class Gate(Provider):
    name = "Gate.io"
    bases = ["https://api.gateio.ws"]
    imap = {"15m": "15m", "30m": "30m", "1h": "1h", "4h": "4h", "1d": "1d"}
    caps = 1000

    def tickers(self):
        d = self.get("/api/v4/spot/tickers")
        return {x["currency_pair"][:-5]: float(x.get("quote_volume") or 0)
                for x in d if x["currency_pair"].endswith("_USDT")}

    def klines(self, base, interval, bars):
        k = self.get("/api/v4/spot/candlesticks", {"currency_pair": base + "_USDT",
                                                   "interval": self.imap[interval],
                                                   "limit": min(bars, self.caps)})
        # urutan Gate: [ts, quote_vol, close, high, low, open, base_vol, closed]
        return (np.array([float(x[2]) for x in k]), np.array([float(x[1]) for x in k]))


class Indodax(Provider):
    """Harga dalam IDR (ada sedikit noise kurs USDT/IDR). Dipakai hanya jika lainnya gagal."""
    name = "Indodax"
    bases = ["https://indodax.com"]
    imap = {"15m": "15", "30m": "30", "1h": "60", "4h": "240", "1d": "1D"}
    caps = 1000

    def tickers(self):
        t = self.get("/api/ticker_all")["tickers"]
        fx = float(t.get("usdt_idr", {}).get("last", 16000) or 16000)
        out = {}
        for k, v in t.items():
            if k.endswith("_idr") and k != "usdt_idr":
                out[k[:-4].upper()] = float(v.get("vol_idr", 0) or 0) / fx
        return out

    def klines(self, base, interval, bars):
        now = int(time.time())
        n = min(bars, self.caps)
        rows = self.get("/tradingview/history_v2", {"symbol": base + "IDR", "tf": self.imap[interval],
                                                    "from": now - n * INTERVAL_SEC[interval], "to": now})
        close = np.array([float(x["Close"]) for x in rows])
        vol = np.array([float(x["Volume"]) for x in rows])
        return close, close * vol


def all_providers():
    return {
        "binance": BinanceLike("Binance", [
            "https://data-api.binance.vision", "https://api.binance.com", "https://api1.binance.com",
            "https://api2.binance.com", "https://api3.binance.com", "https://api4.binance.com"],
            {"15m": "15m", "30m": "30m", "1h": "1h", "4h": "4h", "1d": "1d"}, 1000),
        "mexc": BinanceLike("MEXC", ["https://api.mexc.com"],
                            {"15m": "15m", "30m": "30m", "1h": "60m", "4h": "4h", "1d": "1d"}, 500),
        "bybit": Bybit(), "okx": OKX(), "kucoin": KuCoin(), "gate": Gate(), "indodax": Indodax(),
    }


def probe(providers):
    """Cek semua sumber paralel. Return (aktif, {nama: {BASE: vol}})."""
    def one(p):
        try:
            return p, p.tickers(), None
        except Exception as e:
            return p, None, str(e)[:110]

    with ThreadPoolExecutor(len(providers)) as ex:
        res = list(ex.map(one, providers))
    active, tick = [], {}
    print("Cek koneksi sumber data:")
    for p, t, err in res:
        if t:
            active.append(p)
            tick[p.name] = t
            print(f"  [OK]    {p.name:<8} {len(t):>5} pair  ({p.ok_base})")
        else:
            print(f"  [BLOK]  {p.name:<8} {err or 'data kosong'}")
    return active, tick


def is_junk(base, allbases):
    import re
    if base in STABLES or base in WRAPPED or re.search(r"\d[LS]$", base):
        return True
    for suf in ("UP", "DOWN", "BULL", "BEAR"):
        if base.endswith(suf) and base[: -len(suf)] in allbases:
            return True
    return False


def build_universe(active, tick, top_n, min_vol, interval, bars):
    byname = {p.name: p for p in active}
    vols = {}
    for pname, t in tick.items():
        for base, v in t.items():
            vols.setdefault(base, {})[pname] = v
    allbases = set(vols)
    cands = [(b, sum(v.values())) for b, v in vols.items()
             if max(v.values()) >= min_vol and not is_junk(b, allbases)]
    cands.sort(key=lambda x: -x[1])
    if top_n > 0:
        cands = cands[:top_n]
    print(f"Kandidat: {len(cands)} koin. Mengunduh candle {interval}...")

    def fetch(base, total):
        order = sorted(vols[base], key=lambda n: -vols[base][n])
        for pname in order:
            try:
                close, qv = byname[pname].klines(base, interval, bars)
                if len(close) >= MIN_BARS and np.all(close > 0):
                    return base, dict(close=close, qvol=qv, src=pname, vol24=total)
            except Exception:
                continue
        return base, None

    universe, done = {}, 0
    with ThreadPoolExecutor(6) as ex:
        futs = [ex.submit(fetch, b, t) for b, t in cands]
        for f in as_completed(futs):
            base, d = f.result()
            done += 1
            print(f"\r  {done}/{len(cands)}", end="", flush=True)
            if d:
                universe[base] = d
    print()
    return universe


def demo_universe(n=40, T=500, seed=7):
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n):
        mu = rng.normal(0, 4e-4)
        sig = rng.uniform(0.004, 0.03)
        r = mu + sig * rng.standard_t(4, T) * np.sqrt(2 / 4)
        qv = rng.uniform(1e6, 5e8, T)
        out[f"DEMO{i:02d}"] = dict(close=100 * np.exp(np.cumsum(r)), qvol=qv, src="demo",
                                   vol24=qv[-24:].sum())
    return out


# =========================================================================== MATH
def ewma_vol(r, lam=0.94):
    s2 = np.var(r[:30]) if len(r) > 30 else np.var(r)
    for x in r:
        s2 = lam * s2 + (1 - lam) * x * x
    return np.sqrt(s2)


def kalman_trend(logp, r):
    """Local linear trend. State=[level, slope]. Return (slope, slope_variance)."""
    s2 = np.var(r) + 1e-12
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.diag([0.5 * s2, 1e-3 * s2])
    R = 0.5 * s2
    x = np.array([logp[0], 0.0])
    P = np.diag([s2, s2])
    for y in logp[1:]:
        x = F @ x
        P = F @ P @ F.T + Q
        S = P[0, 0] + R
        K = P[:, 0] / S
        x = x + K * (y - x[0])
        P = P - np.outer(K, P[0, :])
    return x[1], max(P[1, 1], 1e-18)


def hurst(logp, max_lag=24):
    lags = np.arange(2, max_lag)
    tau = np.array([np.std(logp[l:] - logp[:-l]) for l in lags])
    ok = tau > 0
    if ok.sum() < 5:
        return 0.5
    return float(np.polyfit(np.log(lags[ok]), np.log(tau[ok]), 1)[0])


def max_drawdown(close):
    peak = np.maximum.accumulate(close)
    return float(np.max(1 - close / peak))


def sortino(r):
    dd = np.sqrt(np.mean(np.minimum(r, 0) ** 2)) + 1e-12
    return float(np.mean(r) / dd)


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


def monte_carlo(mu, sigma, horizon, fee, rng, n_sim=8000, df=4):
    z = rng.standard_t(df, size=(n_sim, horizon)) * np.sqrt((df - 2) / df)
    cum = (mu + sigma * z).sum(axis=1)
    net = np.expm1(cum) - fee
    q = np.quantile(net, 0.05)
    return float((net > 0).mean()), float(net.mean()), float(net[net <= q].mean())


def entry_plan(mu, sigma, horizon, fee, rng, n_sim=6000, df=4):
    """
    Rencana trade dari simulasi Monte Carlo Student-t (harga awal dinormalisasi = 0 di skala log).
      - Entry limit : kedalaman pullback yang punya peluang terisi ~50% (dan ~25% utk 'dalam')
                      dalam F bar ke depan (kuantil dari distribusi 'drawdown awal' tiap path).
      - SL / TP     : grid kelipatan sigma*sqrt(horizon); syarat P(TP)>=15%, lalu dipilih yang memaksimalkan
                      mean/std return per-trade (Sharpe per trade) setelah fee, dihitung
                      lewat first-passage (barrier) di setiap path simulasi.
    """
    z = rng.standard_t(df, size=(n_sim, horizon)) * np.sqrt((df - 2) / df)
    cum = np.cumsum(mu + sigma * z, axis=1)
    F = max(2, horizon // 4)
    depth = -np.minimum(cum[:, :F].min(axis=1), 0.0)       # kedalaman dip dalam F bar pertama
    d50, d25 = float(np.quantile(depth, 0.50)), float(np.quantile(depth, 0.75))

    scale = sigma * np.sqrt(horizon)
    big = horizon + 1
    best = None
    for sl_m in (0.5, 0.75, 1.0, 1.5):
        for tp_m in (0.4, 0.6, 0.8, 1.0, 1.5):
            sl, tp = sl_m * scale, tp_m * scale
            h_tp, h_sl = cum >= tp, cum <= -sl
            t_tp = np.where(h_tp.any(1), h_tp.argmax(1), big)
            t_sl = np.where(h_sl.any(1), h_sl.argmax(1), big)
            ret = np.where(t_tp < t_sl, np.expm1(tp),
                           np.where(t_sl < t_tp, np.expm1(-sl), np.expm1(cum[:, -1]))) - fee
            p_tp_, p_sl_ = float((t_tp < t_sl).mean()), float((t_sl < t_tp).mean())
            sharpe = ret.mean() / (ret.std() + 1e-12)
            # TP yang hampir tak pernah tercapai tidak berguna: wajib P(TP) >= 15%
            key = (p_tp_ >= 0.15, sharpe)
            if best is None or key > best[0]:
                best = (key, sl, tp, float(ret.mean()), p_tp_, p_sl_)
    _, sl, tp, ev, p_tp, p_sl = best
    rr = float(np.expm1(tp) / (-np.expm1(-sl)))
    tp1 = tp * 0.6
    tp2 = tp * 1.0
    tp3 = tp * 1.5
    return dict(d_ideal=d50, d_deep=d25, sl=float(sl), tp=float(tp), tp1=float(tp1), tp2=float(tp2), tp3=float(tp3),
                ev_trade=ev, p_tp=p_tp, p_sl=p_sl, rr=rr, fill_bars=F)


def format_signal_v2(r, htf, ltf, horizon, risk, sid):
    f = lambda x: f"{x:.6g}"
    side = getattr(r, "side", "LONG")
    market = "FUTURES" if side == "SHORT" else "SPOT"
    sl_pct = (r.sl / r.entry - 1) * 100
    tp1_pct = (r.tp1 / r.entry - 1) * 100 if hasattr(r, "tp1") and r.tp1 else (r.tp / r.entry - 1) * 100 * 0.6
    tp2_pct = (r.tp2 / r.entry - 1) * 100 if hasattr(r, "tp2") and r.tp2 else (r.tp / r.entry - 1) * 100
    tp3_pct = (r.tp3 / r.entry - 1) * 100 if hasattr(r, "tp3") and r.tp3 else (r.tp / r.entry - 1) * 100 * 1.5
    
    tp1_val = r.tp1 if hasattr(r, "tp1") and r.tp1 else r.entry * (1 + tp1_pct/100)
    tp2_val = r.tp2 if hasattr(r, "tp2") and r.tp2 else r.tp
    tp3_val = r.tp3 if hasattr(r, "tp3") and r.tp3 else r.entry * (1 + tp3_pct/100)
    
    trend_regime = getattr(r, "trend_regime", "BULLISH").replace("_", " ")
    vol_regime = getattr(r, "vol_regime", "NORMAL_VOL").replace("_", " ")
    side_emoji = "🟢" if side == "LONG" else "🔴"
    
    return (
        f"{side_emoji} *#{r.symbol}USDT — {side}*\n"
        f"`{market}` · HTF {htf} / LTF {ltf}\n"
        f"───────────────────────────\n"
        f"🎯 *ENTRY ZONE*\n"
        f"• Limit: `{f(r.entry)}` \n"
        f"• Deep: `{f(r.entry_deep)}` (25% fill prob)\n\n"
        f"🎯 *TARGETS*\n"
        f"• TP1: `{f(tp1_val)}` ({tp1_pct:+.1f}%)\n"
        f"• TP2: `{f(tp2_val)}` ({tp2_pct:+.1f}%)\n"
        f"• TP3: `{f(tp3_val)}` ({tp3_pct:+.1f}%)\n"
        f"• SL: `{f(r.sl)}` ({sl_pct:+.1f}%)\n\n"
        f"📊 *METRICS*\n"
        f"• R:R Ratio: `1 : {r.rr:.2f}`\n"
        f"• HTF Bias: `{trend_regime}` | Vol: `{vol_regime}`\n"
        f"• Win Prob: `{r.p_tp*100:.0f}%` | EV: `{r.ev_trade*100:+.2f}%`\n"
        f"• Max Risk: `{risk*100:.1f}%` (Alloc ~`{r.alloc*100:.1f}%`)\n"
        f"• Valid: `{horizon} bars` ({ltf}) | Cancel in `{int(r.fill_bars)} bars`\n"
        f"───────────────────────────\n"
        f"⚠️ _Statistical estimate, not financial advice._"
    )


def format_signal(r, interval, horizon, risk, sid):
    """Pesan sinyal gaya Telegram."""
    return format_signal_v2(r, "4h", interval, horizon, risk, sid)


def format_top10_daily(df, interval, tier="pro"):
    """Format top 10 coins untuk daily trading watchlist."""
    f = lambda x: f"{x:.6g}"
    lines = [
        "📋 *DAILY WATCHLIST*",
        f"Timeframe: `{interval}` · Analyzed: `{len(df)} coins`",
        "───────────────────────────"
    ]
    top10 = df.head(10)
    for i, (_, r) in enumerate(top10.iterrows(), 1):
        if tier == "free" and i <= 5:
            lines.append(f"`{i:2d}.` 🔒 *Premium Member Only*")
        else:
            status_emoji = "🟢" if r.status == "ENTRY" else ("🟡" if r.status == "PANTAU" else "🔴")
            side_str = getattr(r, "side", "LONG")
            lines.append(
                f"`{i:2d}.` {status_emoji} *{r.symbol}* (`{side_str}`) · Score: `{r.score:+.2f}` · Win: `{r.p_win*100:.1f}%`"
            )
    lines.extend([
        "───────────────────────────",
        "🟢 ENTRY  🟡 PANTAU  🔴 SKIP",
        "⚠️ _Not financial advice. DYOR._"
    ])
    if tier == "free":
        lines.append("💎 Upgrade to Premium: @openposconnect")
    return "\n".join(lines)


def load_env():
    """Baca file .env di folder script (KEY=VALUE per baris). Env var sistem tetap diprioritaskan."""
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def send_telegram(text, chat_id=None):
    import os
    load_env()
    tok = os.environ.get("TELEGRAM_BOT_TOKEN")
    if chat_id is None:
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not chat_id or str(chat_id).startswith("ISI_"):
        print("  (kirim dilewati: isi TELEGRAM_BOT_TOKEN dan TELEGRAM_CHAT_ID di file .env)")
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                          data={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}, timeout=15)
        return r.ok
    except Exception as e:
        print(f"  (gagal kirim: {type(e).__name__})")
        return False


def zscore(s):
    s = s.clip(s.quantile(0.05), s.quantile(0.95))  # winsorize
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else s * 0


def analyze_all(universe, horizon, fee, risk=0.01, seed=42, side="both"):
    rng = np.random.default_rng(seed)
    rows = []
    for sym, d in universe.items():
        close = d["close"]
        if len(close) < MIN_BARS or np.any(close <= 0):
            continue
        logp = np.log(close)
        r = np.diff(logp)
        slope, pvar = kalman_trend(logp, r)
        rows.append(dict(symbol=sym, src=d["src"], close=close[-1], mu_raw=slope, mu_var=pvar,
                         sigma=ewma_vol(r), hurst=hurst(logp), sortino=sortino(r),
                         mdd=max_drawdown(close), liq=np.log10(d["vol24"] + 1)))
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Shrinkage Empirical-Bayes: tarik drift ke prior lintas-koin
    w = 1 / df.mu_var
    m0 = float((df.mu_raw * w).sum() / w.sum())
    tau2 = max(df.mu_raw.var() - df.mu_var.mean(), 1e-14)
    B = df.mu_var / (df.mu_var + tau2)
    df["mu"] = B * m0 + (1 - B) * df.mu_raw
    df["trend_t"] = df.mu_raw / np.sqrt(df.mu_var)

    if side == "long":
        df["side"] = "LONG"
    elif side == "short":
        df["side"] = "SHORT"
    else:
        df["side"] = np.where(df.trend_t >= 0, "LONG", "SHORT")

    is_short = df["side"] == "SHORT"
    df["mu_trade"] = np.where(is_short, -df.mu, df.mu)
    df["trend_dir"] = np.where(is_short, -df.trend_t, df.trend_t)

    # Sinyal sadar-rezim: H>0.5 hargai tren, H<0.5 hukum tren
    df["regime"] = np.sign(df.trend_dir) * (df.hurst - 0.5) * np.abs(df.trend_dir).clip(upper=4)

    res = [monte_carlo(m, s, horizon, fee, rng) for m, s in zip(df.mu_trade, df.sigma)]
    df["p_win"], df["ev"], df["cvar5"] = zip(*res)

    df["kelly"] = (0.5 * df.mu_trade / (df.sigma ** 2)).clip(0, 0.25)  # half-Kelly, dibatasi maks 25%

    # Rencana entry / SL / TP per koin
    plans = [entry_plan(m, sg, horizon, fee, rng) for m, sg in zip(df.mu_trade, df.sigma)]
    pl = pd.DataFrame(plans)
    df["entry"] = np.where(is_short, df.close * np.exp(pl.d_ideal), df.close * np.exp(-pl.d_ideal))
    df["entry_deep"] = np.where(is_short, df.close * np.exp(pl.d_deep), df.close * np.exp(-pl.d_deep))
    df["sl"] = np.where(is_short, df.entry * np.exp(pl.sl), df.entry * np.exp(-pl.sl))
    df["tp"] = np.where(is_short, df.entry * np.exp(-pl.tp), df.entry * np.exp(pl.tp))
    df["tp1"] = np.where(is_short, df.entry * np.exp(-pl.tp1), df.entry * np.exp(pl.tp1))
    df["tp2"] = np.where(is_short, df.entry * np.exp(-pl.tp2), df.entry * np.exp(pl.tp2))
    df["tp3"] = np.where(is_short, df.entry * np.exp(-pl.tp3), df.entry * np.exp(pl.tp3))
    df["trend_regime"] = np.where(df.trend_t > 0.5, "BULLISH", np.where(df.trend_t < -0.5, "BEARISH", "SIDEWAYS"))
    df["vol_regime"] = np.where(df.sigma > 0.02, "HIGH_VOL", "NORMAL_VOL")
    
    for c in ("rr", "p_tp", "p_sl", "ev_trade", "fill_bars"):
        df[c] = pl[c]
    sl_pct = 1 - np.exp(-pl.sl)
    df["alloc"] = np.minimum(df.kelly, risk / sl_pct)          # % modal; risiko ~`risk` jika kena SL
    df["status"] = np.where((df.ev_trade > 0) & (df.trend_dir > 0) & (df.p_win >= 0.5), "ENTRY",
                            np.where(df.ev_trade > 0, "PANTAU", "SKIP"))

    ratio = df.ev / (df.cvar5.abs() + 1e-9)
    df["score"] = (0.30 * zscore(ratio) + 0.25 * zscore(df.p_win) + 0.20 * zscore(df.trend_dir)
                   + 0.15 * zscore(df.regime) + 0.10 * zscore(df.sortino)
                   - 0.10 * zscore(df.mdd) + 0.05 * zscore(df.liq))
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.index += 1
    return df


# =========================================================================== MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h", choices=list(INTERVAL_SEC))
    ap.add_argument("--bars", type=int, default=500)
    ap.add_argument("--top", type=int, default=0, help="jumlah koin urut volume (0 = semua/maks koin)")
    ap.add_argument("--min-vol", type=float, default=1e5, help="min volume 24j USDT (default: 100rb USDT)")
    ap.add_argument("--horizon", type=int, default=24, help="horizon simulasi (jumlah bar)")
    ap.add_argument("--fee", type=float, default=0.002, help="biaya round-trip (0.002 = 0.2%%)")
    ap.add_argument("--risk", type=float, default=0.01,
                    help="risiko per trade jika kena SL, fraksi modal (0.01 = 1%%)")
    ap.add_argument("--side", default="both", choices=["both", "long", "short"],
                    help="pilih tipe sinyal: long, short, atau both")
    ap.add_argument("--show", type=int, default=10, help="jumlah koin di tabel saran entry")
    ap.add_argument("--telegram", action="store_true", help="cetak pesan sinyal siap-kirim")
    ap.add_argument("--send", action="store_true", help="kirim sinyal ke Telegram (butuh env token)")
    ap.add_argument("--log", default="", help="catat sinyal ke CSV untuk track record, mis. signals_log.csv")
    ap.add_argument("--cooldown", type=float, default=12.0,
                    help="jam tunggu sebelum koin yang sama boleh dikirim lagi (butuh --log)")
    ap.add_argument("--max-signals", type=int, default=3, help="maks sinyal per scan (selektif)")
    ap.add_argument("--sources", default="all",
                    help="all atau daftar: binance,mexc,bybit,okx,kucoin,gate,indodax")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--csv", default="scan_result.csv")
    a = ap.parse_args()

    if a.demo:
        n_demo = a.top if a.top > 0 else 40
        universe = demo_universe(n_demo, a.bars)
    else:
        provs = all_providers()
        if a.sources != "all":
            names = [s.strip().lower() for s in a.sources.split(",")]
            bad = [n for n in names if n not in provs]
            if bad:
                sys.exit(f"Sumber tidak dikenal: {bad}. Pilihan: {list(provs)}")
            provs = {n: provs[n] for n in names}
        active, tick = probe(list(provs.values()))
        if not active:
            sys.exit("\nSemua sumber tidak bisa dijangkau.\n"
                     "Solusi: ganti DNS ke 1.1.1.1 / 8.8.8.8, nyalakan VPN, atau pindah jaringan.\n"
                     "Atau jalankan dengan --demo.")
        universe = build_universe(active, tick, a.top, a.min_vol, a.interval, a.bars)

    df = analyze_all(universe, a.horizon, a.fee, a.risk, side=a.side)
    if df.empty:
        sys.exit("Tidak ada data yang cukup.")

    show = df[["symbol", "side", "src", "close", "score", "p_win", "ev", "cvar5", "trend_t", "hurst",
               "sortino", "mdd", "kelly"]].copy()
    for c in ("p_win", "ev", "cvar5", "mdd"):
        show[c] = (show[c] * 100).round(2)
    pd.set_option("display.width", 220)
    print(f"\nTOP 15 (horizon {a.horizon} bar {a.interval}, fee {a.fee*100:.2f}%, side {a.side}) - {len(df)} koin dianalisis")
    print(show.head(15).round(3).to_string())
    def fmt(x):
        return f"{x:.6g}"

    def pct(x, ref):
        return f"{fmt(x)} ({(x / ref - 1) * 100:+.1f}%)"

    rows = []
    for sym, r in df.head(a.show).iterrows():
        rows.append({"symbol": r.symbol, "side": r.side, "status": r.status, "harga": fmt(r.close),
                     "entry_limit": pct(r.entry, r.close), "entry_dalam": pct(r.entry_deep, r.close),
                     "SL": pct(r.sl, r.entry), "TP": pct(r.tp, r.entry), "RR": round(r.rr, 2),
                     "P(TP)%": round(r.p_tp * 100, 1), "P(SL)%": round(r.p_sl * 100, 1),
                     "EV%": round(r.ev_trade * 100, 2), "alokasi%": round(r.alloc * 100, 1)})
    fb = int(df.fill_bars.iloc[0])
    print(f"\nSARAN ENTRY (top {a.show}; risiko {a.risk*100:.1f}% modal per trade jika kena SL)")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"- entry_limit: peluang terisi ~50% dalam {fb} bar; entry_dalam: ~25%. Batalkan limit jika belum terisi dalam {fb} bar {a.interval}.")
    print(f"- Keluar paksa di harga pasar jika posisi belum kena SL/TP setelah {a.horizon} bar {a.interval}.")
    print("- ENTRY = EV positif & tren naik & P(untung)>=50%; PANTAU = EV positif tapi sinyal tren lemah; SKIP = EV negatif setelah fee.")
    n_ok = int((df.status == "ENTRY").sum())
    if n_ok == 0:
        print("- Saat ini TIDAK ADA koin berstatus ENTRY. Menunggu (tidak trading) adalah keputusan yang valid.")
    if a.telegram or a.send or a.log:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        sigs = df[df.status == "ENTRY"]
        import os
        if a.log and os.path.exists(a.log):
            try:
                old = pd.read_csv(a.log)
                t = pd.to_datetime(old.time_utc, utc=True)
                recent = set(old.loc[(now - t).dt.total_seconds() < a.cooldown * 3600, "symbol"])
                skipped = [x for x in sigs.symbol if x in recent]
                if skipped:
                    print(f"\nDilewati (masih cooldown {a.cooldown:g} jam): {', '.join(skipped)}")
                sigs = sigs[~sigs.symbol.isin(recent)]
            except Exception as e:
                print(f"(log lama tidak terbaca, cooldown dilewati: {type(e).__name__})")
        sigs = sigs.head(a.max_signals)
        if sigs.empty:
            print("\nTidak ada sinyal yang memenuhi syarat ENTRY. Jangan kirim sinyal paksa.")
        log_rows = []
        
        # Load chat IDs
        load_env()
        regular_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        premium_chat_id = os.environ.get("PREMIUM_TELEGRAM_CHAT_ID")
        
        # Determine which chat IDs to send to based on --send flag
        send_targets = []
        if a.send:
            if regular_chat_id and not str(regular_chat_id).startswith("ISI_"):
                send_targets.append(("free", regular_chat_id))
            if premium_chat_id and not str(premium_chat_id).startswith("ISI_"):
                send_targets.append(("pro", premium_chat_id))
            # If only one is configured, use that
            if not send_targets and regular_chat_id and not str(regular_chat_id).startswith("ISI_"):
                send_targets.append(("free", regular_chat_id))
        
        top10_sent = {tid: False for tid, _ in send_targets}
        
        for _, r in sigs.iterrows():
            sid = f"{now:%Y%m%d-%H%M}-{r.symbol}"
            msg = format_signal_v2(r, "4h", a.interval, a.horizon, a.risk, sid)
            try:
                print("\n" + "-" * 60 + "\n" + msg)
            except Exception:
                print("\n" + "-" * 60 + "\n" + msg.encode("ascii", "replace").decode())
            if a.send:
                for tier, chat_id in send_targets:
                    if not top10_sent[tier]:
                        top10_msg = format_top10_daily(df, a.interval, tier=tier)
                        if send_telegram(top10_msg, chat_id=chat_id):
                            print(f"  -> watchlist terkirim ke {chat_id} ({tier})")
                        top10_sent[tier] = True
                    if send_telegram(msg, chat_id=chat_id):
                        print(f"  -> sinyal terkirim ke {chat_id} ({tier})")
                    else:
                        print(f"  -> TIDAK terkirim ke {chat_id} ({tier})")
            log_rows.append(dict(id=sid, time_utc=now.isoformat(timespec="seconds"), symbol=r.symbol,
                                 src=r.src, interval=a.interval, horizon=a.horizon, fill_bars=int(r.fill_bars),
                                 entry=r.entry, sl=r.sl, tp=r.tp, p_tp=r.p_tp, p_sl=r.p_sl, ev=r.ev_trade))
        if a.log and log_rows:
            pd.DataFrame(log_rows).to_csv(a.log, mode="a", header=not os.path.exists(a.log), index=False)
            print(f"\n{len(log_rows)} sinyal dicatat ke {a.log}")
    df.to_csv(a.csv)
    print(f"\nSemua hasil disimpan ke {a.csv}")
    print("Catatan: ini ranking edge statistik, bukan prediksi pasti. Gunakan stop-loss & ukuran posisi kecil.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code)
    except Exception:
        traceback.print_exc()
    _pause()
