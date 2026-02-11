"""Unified evaluation pipeline for GAVEL.

This script runs calibration and evaluation in a single pass, processing
logits in-memory without requiring separate extraction steps.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from gavel.config import load_config
from gavel.evaluation.calibration import calibrate
from gavel.evaluation.metrics import evaluate
from gavel.models import load_trained_classifier
from gavel.preprocessing import extract_dialogues_in_memory
from gavel.training import load_model_and_tokenizer
from gavel.utils.logging import add_verbose_arg, setup_logger

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run unified calibration and evaluation pipeline")
add_verbose_arg(parser)
parser.add_argument(
    "--config",
    "-c",
    default="config.json",
    help="Path to configuration file (default: config.json)",
)
parser.add_argument(
    "--force-recalibrate",
    action="store_true",
    help="Force recalibration even if thresholds.json exists",
)
parser.add_argument(
    "--skip-calibration",
    action="store_true",
    help="Skip calibration and use existing thresholds (fails if missing)",
)
parser.add_argument(
    "--unified-ruleset",
    default=None,
    help="Path to unified ruleset JSON (default: models/{model_name}/rules.json)",
)
parser.add_argument("--calibration-only", action="store_true", help="Run only calibration and exit")
args = parser.parse_args()

# Setup logger
logger = setup_logger(__name__, verbose=args.verbose)

# Load configuration
logger.info("Loading configuration...")
config = load_config(args.config)

# Load labels
labels = config.labels

# Load unified ruleset

if args.unified_ruleset is None:
    if config.ruleset_path:
        args.unified_ruleset = config.ruleset_path
    else:
        args.unified_ruleset = f"models/{config.model_name}/rules.json"

if not Path(args.unified_ruleset).exists():
    logger.error(f"Unified ruleset not found: {args.unified_ruleset}")
    exit(1)

with open(args.unified_ruleset, "r") as f:
    unified_ruleset = json.load(f)

# Filter to enabled rulesets only
unified_ruleset = {k: v for k, v in unified_ruleset.items() if v.get("enabled", True)}

# Determine paths
thresholds_path = config.paths.thresholds_path
rnn_path = config.paths.rnn_model_path

# ------------------------------------------------------------------ #
# Load models (once)
# ------------------------------------------------------------------ #
logger.info("Loading LLM and RNN classifier...")
model, tokenizer = load_model_and_tokenizer(config.model.name_or_path)
rnn_model = load_trained_classifier(
    rnn_path,
    config.num_labels,
    len(config.model.selected_layers),
    rnn_config=config.rnn,
)

# ------------------------------------------------------------------ #
# Calibration Phase
# ------------------------------------------------------------------ #
needs_calibration = args.force_recalibrate or (
    not args.skip_calibration and not Path(thresholds_path).exists()
)

if args.skip_calibration and not Path(thresholds_path).exists():
    logger.error(f"--skip-calibration specified but thresholds not found: {thresholds_path}")
    exit(1)

if needs_calibration:
    logger.info("=" * 60)
    logger.info("CALIBRATION PHASE")
    logger.info("=" * 60)

    logger.info("Extracting calibration dialogues...")
    calib_dialogues = extract_dialogues_in_memory(
        dataset_root=config.paths.calibration_dataset,
        splits=["usecase_level", "CE_level"],
        model=model,
        rnn_model=rnn_model,
        tokenizer=tokenizer,
        selected_layers=config.model.selected_layers,
        window_size=config.eval.window_size,
        window_stride=config.eval.window_stride,
        batch_size=32,
        max_samples_per_usecase=None,  # Use all calibration data
        logger=logger,
        show_progress=True,
    )

    logger.info("Running calibration...")
    thresholds = calibrate(
        output_dir=config.paths.calibration_dir,
        labels=labels,
        unified_ruleset=unified_ruleset,
        dialogue_data=calib_dialogues,
        show_progress=True,
        generate_plots=True,
        logger=logger,
    )
else:
    logger.info(f"Using existing thresholds from {thresholds_path}")

if args.calibration_only:
    logger.info("Calibration complete. Exiting (--calibration-only specified).")
    exit(0)


# ------------------------------------------------------------------ #
# Evaluation Phase
# ------------------------------------------------------------------ #
logger.info("=" * 60)
logger.info("EVALUATION PHASE")
logger.info("=" * 60)

logger.info("Extracting evaluation dialogues...")
eval_dialogues = extract_dialogues_in_memory(
    dataset_root=config.paths.eval_dataset,
    splits=["positive", "negative", "neutral"],
    model=model,
    rnn_model=rnn_model,
    tokenizer=tokenizer,
    selected_layers=config.model.selected_layers,
    window_size=config.eval.window_size,
    window_stride=config.eval.window_stride,
    batch_size=32,
    max_samples_per_usecase=config.eval.max_samples_per_usecase,  # Limit for evaluation
    logger=logger,
    show_progress=True,
)

logger.info("Running evaluation...")
results = evaluate(
    output_dir=config.paths.results_dir,
    labels=labels,
    thresholds_path=thresholds_path,
    unified_ruleset_path=args.unified_ruleset,
    dialogue_data=eval_dialogues,
    compute_auc=True,
    logger=logger,
)


# Print per-usecase metrics
metrics_df = results.get("metrics", pd.DataFrame())

if not metrics_df.empty:
    # Split into Malicious and Neutral
    neutral_keys = ["conversational", "instructive"]
    neutral_mask = metrics_df["Usecase"].isin(neutral_keys)
    neutral_df = metrics_df[neutral_mask]
    malicious_df = metrics_df[~neutral_mask]

    # Helper to print dataframe
    def print_metrics_table(df, title):
        if df.empty:
            return
        logger.info("\n" + "=" * 80)
        logger.info(f"{title}")
        logger.info("-" * 80)
        logger.info(f"{'Usecase':<40} {'TPR':<10} {'FPR':<10} {'Acc':<10} {'Supp(P/N)':<15}")
        logger.info("-" * 80)
        for _, row in df.iterrows():
            support = f"{int(row['Support_Pos'])}/{int(row['Support_Neg'])}"
            logger.info(
                f"{row['Usecase']:<40} "
                f"{row['TPR']:.3f}      "
                f"{row['FPR']:.3f}      "
                f"{row['Accuracy']:.3f}      "
                f"{support:<15}"
            )
        logger.info("=" * 80)

    print_metrics_table(malicious_df, "MALICIOUS USE CASES")
    print_metrics_table(neutral_df, "NEUTRAL USE CASES")

# Print AUC results if available
if "auc" in results and not results["auc"].empty:
    logger.info("\n" + "=" * 60)
    logger.info(f"{'Usecase':<40} {'ROC-AUC':<10} {'PR-AUC':<10}")
    logger.info("-" * 60)
    for _, row in results["auc"].iterrows():
        roc = f"{row['ROC_AUC']:.4f}" if pd.notna(row["ROC_AUC"]) else "N/A"
        pr = f"{row['PR_AUC']:.4f}" if pd.notna(row["PR_AUC"]) else "N/A"
        logger.info(f"{row['Usecase']:<40} {roc:<10} {pr:<10}")
    logger.info("=" * 60)

logger.info("\n" + "=" * 60)
logger.info("FINAL RESULTS (Weighted Averages)")
logger.info("=" * 60)
for metric, value in results.get("weighted_averages", {}).items():
    logger.info(f"{metric:<25}: {value:.4f}")
logger.info("=" * 60)

logger.info("Pipeline complete!")
logger.info(f"Calibration results: {config.paths.calibration_dir}")
logger.info(f"Evaluation results: {config.paths.results_dir}")
logger.info("=" * 60)
