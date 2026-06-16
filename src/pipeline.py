import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

sys.path.insert(0, str(Path(__file__).parent))

from ingestion.download_historical          import run as run_download
from ingestion.sentiment_collector          import collect as collect_sentiment, save as save_sentiment
from ingestion.earnings_collector           import collect as collect_earnings,  save as save_earnings
from ingestion.insider_collector            import collect as collect_insider,   save as save_insider
from ingestion.options_collector            import collect as collect_options,   save as save_options
from ingestion.valuation_collector          import collect as collect_valuation, save as save_valuation
from quality.quality_pipeline               import run_quality_pipeline
from features.feature_pipeline              import run_feature_pipeline
from detection.detection_pipeline           import run as run_detection
from prediction.prediction_pipeline         import run_prediction_pipeline
from decision.decision_pipeline             import run as run_decision
from explainability.explainability_pipeline import run as run_explainability
from reporting.anomaly_log                  import log as log_anomalies
from reporting.daily_report                 import run as run_daily_report
from cache.cache_manager                    import CacheManager

ROOT = Path(__file__).resolve().parent.parent
SRC  = Path(__file__).resolve().parent


def _parse_args():
    p = argparse.ArgumentParser(description="FinWatch AI pipeline")
    p.add_argument(
        "--force", action="store_true",
        help="Bypass ALL layer caches and recompute everything from scratch",
    )
    p.add_argument(
        "--force-layer", type=int, metavar="N", dest="force_layer",
        help=(
            "Recompute from layer N onward and ignore its cache "
            "(3=Features, 4=Detection, 5=Prediction, 7=Explainability). "
            "Layer 6 (Decision) is never cached."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args  = _parse_args()
    cache = CacheManager(force=args.force, force_from_layer=args.force_layer)

    if args.force:
        print("⚠  --force: all caches bypassed — full recompute")
    elif args.force_layer:
        print(f"⚠  --force-layer {args.force_layer}: recomputing layer {args.force_layer} and downstream")

    # ── Step 0: Download latest price data (not cached — always runs) ────────
    print("\n" + "=" * 55)
    print("STEP 0 — Download latest data")
    print("=" * 55)
    run_download()

    # ── Step 1–2: Quality checks (not cached — fast, always runs) ───────────
    run_quality_pipeline()

    # ── Layer 3: Feature Engineering ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print("LAYER 3 — Feature Engineering")
    print("=" * 55)

    hash3 = cache.run_layer(
        layer_name   = "layer3_features",
        layer_num    = 3,
        input_dirs   = [ROOT / "data/raw/raw_clean"],
        source_dirs  = [SRC / "features"],
        upstream_hash= "",
        fn           = run_feature_pipeline,
        label        = "Layer 3 (Features)",
    )

    # ── Layer 4: Anomaly Detection ───────────────────────────────────────────
    print("\n" + "=" * 55)
    print("LAYER 4 — Anomaly Detection")
    print("=" * 55)

    hash4 = cache.run_layer(
        layer_name   = "layer4_detection",
        layer_num    = 4,
        input_dirs   = [ROOT / "data/features"],
        source_dirs  = [SRC / "detection"],
        upstream_hash= hash3,
        fn           = run_detection,
        label        = "Layer 4 (Detection + Severity)",
    )

    # ── Step 4: Collect fundamental signals (always runs — external APIs) ────
    print("\n" + "=" * 55)
    print("STEP 4 — Collect fundamental signals")
    print("=" * 55)

    earnings_df = collect_earnings()
    if not earnings_df.empty:
        print(f"Earnings saved → {save_earnings(earnings_df)}")

    insider_df = collect_insider()
    if not insider_df.empty:
        print(f"Insider saved → {save_insider(insider_df)}")

    options_df = collect_options()
    if not options_df.empty:
        print(f"Options saved → {save_options(options_df)}")

    valuation_df = collect_valuation()
    if not valuation_df.empty:
        print(f"Valuation saved → {save_valuation(valuation_df)}")

    # ── Step 5: News sentiment (always runs — external APIs) ─────────────────
    print("\n" + "=" * 55)
    print("STEP 5 — Collect news sentiment")
    print("=" * 55)
    sentiment_df = collect_sentiment()
    if not sentiment_df.empty:
        print(f"Sentiment saved → {save_sentiment(sentiment_df)}")

    # ── Layer 5: Prediction ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("LAYER 5 — Prediction")
    print("=" * 55)

    hash5 = cache.run_layer(
        layer_name   = "layer5_prediction",
        layer_num    = 5,
        input_dirs   = [ROOT / "data/detection"],
        source_dirs  = [SRC / "prediction"],
        upstream_hash= hash4,
        fn           = run_prediction_pipeline,
        label        = "Layer 5 (Prediction)",
    )

    # ── Layer 6: Decision — NEVER cached (trading signals must be fresh) ─────
    print("\n" + "=" * 55)
    print("LAYER 6 — Decision Engine  [never cached]")
    print("=" * 55)
    decisions_df = run_decision()

    # ── Layer 7: Explainability ──────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("LAYER 7 — Explainability")
    print("=" * 55)

    cache.run_layer(
        layer_name   = "layer7_explainability",
        layer_num    = 7,
        input_dirs   = [ROOT / "data/detection", ROOT / "data/prediction"],
        source_dirs  = [SRC / "explainability"],
        upstream_hash= hash5,
        fn           = run_explainability,
        label        = "Layer 7 (Explainability)",
    )

    # ── Reporting (always runs) ───────────────────────────────────────────────
    log_anomalies(decisions_df)
    run_daily_report()
