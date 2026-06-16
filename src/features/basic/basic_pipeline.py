from features.basic.returns import run_returns
from features.basic.volatility import run_volatility
from features.basic.rolling_stats import run_rolling_stats
from features.basic.beta import run_beta
from features.basic.correlation import run_correlation
from features.basic.rsi import run_rsi
from features.basic.drawdown import run_drawdown
from features.basic.expected_shortfall import run as run_expected_shortfall

def run():
    print("--- Layer 3: Feature Engineering ---")
    run_returns()
    run_volatility()
    run_rolling_stats()
    run_beta()
    run_correlation()
    run_rsi()
    run_drawdown()
    run_expected_shortfall()  # needs 'returns' — must stay after run_returns()


if __name__ == "__main__":
    run()
