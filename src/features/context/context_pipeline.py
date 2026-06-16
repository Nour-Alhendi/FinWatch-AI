from features.context.daily_context import run_daily_context
from features.context.trend_context import run_trend_context
from features.context.state_context import run_state_context

def run_context_pipeline():
    run_daily_context()
    run_trend_context()
    run_state_context()


if __name__ == "__main__":
    run_context_pipeline()
