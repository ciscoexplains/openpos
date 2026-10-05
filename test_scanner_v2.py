import unittest
import numpy as np
import pandas as pd

class TestRegimeDetector(unittest.TestCase):
    def test_detect_regime_bullish(self):
        from coin_scanner import detect_regime
        logp = np.linspace(4.0, 4.5, 200)
        r = np.diff(logp)
        res = detect_regime(logp, r, logp, r)
        self.assertIn("trend", res)
        self.assertIn("vol", res)
        self.assertIn("score", res)
        self.assertEqual(res["trend"], "BULLISH")

class TestSignalV2(unittest.TestCase):
    def test_format_signal_v2(self):
        from coin_scanner import format_signal_v2
        row = pd.Series({
            "symbol": "BTC",
            "side": "LONG",
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
        self.assertIn("HTF", msg)
        self.assertIn("LTF", msg)
        self.assertIn("LONG", msg)

    def test_format_signal_v2_short(self):
        from coin_scanner import format_signal_v2
        row = pd.Series({
            "symbol": "ETH",
            "side": "SHORT",
            "close": 3000.0,
            "entry": 3050.0,
            "entry_deep": 3080.0,
            "sl": 3150.0,
            "tp1": 2950.0,
            "tp2": 2900.0,
            "tp3": 2800.0,
            "rr": 2.5,
            "alloc": 0.02,
            "fill_bars": 6,
            "p_tp": 0.45,
            "p_sl": 0.25,
            "ev_trade": 0.02,
            "trend_regime": "BEARISH",
            "vol_regime": "NORMAL_VOL"
        })
        msg = format_signal_v2(row, "4h", "1h", 24, 0.01, "20261004-ETH")
        self.assertIn("FUTURES", msg)
        self.assertIn("SHORT", msg)
        self.assertIn("ETHUSDT", msg)

class TestIntegration(unittest.TestCase):
    def test_full_pipeline_demo(self):
        import subprocess
        cmd = ["python", "coin_scanner.py", "--demo", "--send", "--telegram"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("TOP 15", res.stdout)

if __name__ == "__main__":
    unittest.main()
