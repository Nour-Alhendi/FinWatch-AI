from features.basic.basic_pipeline import run as run_basic
from features.context.context_pipeline import run_context_pipeline as run_context
from features.advanced.advanced_pipeline import run_advanced_features as run_advanced

def run_feature_pipeline():
    run_basic()
    run_context()
    run_advanced()