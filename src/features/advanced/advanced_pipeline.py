import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from return_lags import run_return_lags
from momentum import run_momentum
from vol_change import run_vol_change
from trend_strength import run_trend_strength
from volatility_ratio import run_volatility_ratio
from volume_trend import run_volume_trend

def run_advanced_features():
    print("--- Advanced Features ---")

    print("1) Return Lags")
    run_return_lags()

    print("2) Momentum")
    run_momentum()
    
    print("3) Vol Change")
    run_vol_change()
    
    print("4) Trend Strength")
    run_trend_strength()
    
    print("5) Volatility Ratio")
    run_volatility_ratio()
    
    print("6) Volume Trend") 
    run_volume_trend()


if __name__ == "__main__":
    run_advanced_features()
