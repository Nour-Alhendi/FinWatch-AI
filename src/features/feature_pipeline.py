import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.returns import run_returns
from features.volatility import run_volatility
from features.rolling_stats import run_rolling_stats
from features.beta import run_beta
from features.correlation import run_correlation

def run():
    print("--- Layer 3: Feature Engineering ---")
    run_returns()
    run_volatility()
    run_rolling_stats()
    run_beta()
    run_correlation()

if __name__ == "__main__":
    run()
