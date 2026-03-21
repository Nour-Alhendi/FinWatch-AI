import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from return_lags import run_return_lags
from momentum import run_momentum
from vol_change import run_vol_change
from trend_strength import run_trend_strength
from volatility_ratio import run_volatility_ratio

def run_advanced_features():
    print("--- Advanced Features ---")
    run_return_lags()
    run_momentum()
    run_vol_change()
    run_trend_strength()
    run_volatility_ratio()

if __name__ == "__main__":
    run_advanced_features()
