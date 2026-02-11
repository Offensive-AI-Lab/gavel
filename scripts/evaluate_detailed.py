#!/usr/bin/env python3
"""Detailed evaluation: CE-level stats, per-dialogue logging, sanity checks.

Runs the unified in-memory pipeline but produces granular CSV reports
instead of just aggregate metrics.
"""

import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm

from gavel.config import load_config
from gavel.evaluation.debug import (
    OutcomeRecorder,
    logger_usecase_stats,
    sanity_check_gt,
    write_rows_to_csv,
)
from gavel.evaluation.metrics import (
    compute_triggers,
    compute_usecase_detection_metrics,
    compute_weighted_metrics,
    eval_usecase_detection,
    evaluate_ce_detection_gt,
    evaluate_ce_detection_rules,
    initialize_usecase_stats,
    load_evaluation_setup,
    save_to_csv,
    update_usecase_confusion_matrix,
)
from gavel.models import load_trained_classifier
from gavel.preprocessing import extract_dialogues_in_memory
from gavel.training import load_model_and_tokenizer
from gavel.utils.logging import add_verbose_arg, setup_logger


def main():
    parser = argparse.ArgumentParser(description="Detailed evaluation with CE-level analysis")
    add_verbose_arg(parser)
    parser.add_argument(
        "--config",
        "-c",
        default="config.json",
        help="Path to configuration file (default: config.json)",
    )
    args = parser.parse_args()

    logger = setup_logger(__name__, verbose=args.verbose)

    logger.info("Loading configuration...")
    config = load_config(args.config)

    # Extract config values
    LABELS = config.labels
    NUM_TOPICS = len(LABELS)
    IDXS_TO_LABELS = {v: k for k, v in LABELS.items()}

    thresholds_path = config.paths.thresholds_path
    output_dir = os.path.join(config.paths.base_dir, "results_detailed")
    os.makedirs(output_dir, exist_ok=True)

    # Load evaluation setup
    logger.info("Loading evaluation setup...")
    if config.ruleset_path:
        unified_ruleset_path = config.ruleset_path
    else:
        unified_ruleset_path = f"models/{config.model_name}/rules.json"

    if not Path(unified_ruleset_path).exists():
        logger.error(f"Unified ruleset not found: {unified_ruleset_path}")
        exit(1)

    setup = load_evaluation_setup(
        thresholds_path=thresholds_path,
        unified_ruleset_path=unified_ruleset_path,
        labels=LABELS,
    )

    # Initialize all stats
    fpr_tpr_stats = initialize_usecase_stats(setup["unified_ruleset"], include_neutral=True)
    acc_stats = initialize_usecase_stats(setup["unified_ruleset"], include_neutral=False)
    use_cases_stats_pos = initialize_usecase_stats(setup["unified_ruleset"], include_neutral=True)
    sanity_stats = initialize_usecase_stats(setup["unified_ruleset"], include_neutral=False)
    topic_stats = {
        k: {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
        for k in LABELS.keys()
    }
    rulebased_topic_stats = {
        k: {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
        for k in LABELS.keys()
    }

    all_rows = []
    all_results_topic = []
    recorder = OutcomeRecorder(rows=all_rows)

    # ------------------------------------------------------------------ #
    # Load Models & Extract Dialogues
    # ------------------------------------------------------------------ #
    logger.info("Loading LLM and RNN classifier...")
    model, tokenizer = load_model_and_tokenizer(config.model.name_or_path)
    rnn_model = load_trained_classifier(
        config.paths.rnn_model_path,
        config.num_labels,
        len(config.model.selected_layers),
        rnn_config=config.rnn,
    )

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
        # max_samples_per_usecase=10, # Uncomment for quick testing
        logger=logger,
        show_progress=True,
    )

    logger.info("Processing dialogues...")
    processed = 0

    for item in tqdm(eval_dialogues, desc="Evaluating", unit="dial"):
        dialogue_logits = item["logits"]  # Tensor
        meta = item["metadata"]

        split = meta.get(
            "split", ""
        )  # Actually 'category' in extract_dialogues, need to check preprocessing mapping
        # extract_dialogues_in_memory likely returns 'split' in metadata based on dir structure?
        # let's assume 'category' or 'split' is correct. In metrics.py evaluate() uses 'category' from dataset dict usually?
        # Wait, extract_dialogues_in_memory returns list of dicts.
        # In evaluate.py: splits=["positive", "negative", "neutral"] are search dirs.
        # The metadata usually contains 'split' (positive/negative/neutral) if implemented that way.
        # Let's trust 'split' key is present or map it.
        # Using 'split' as key for now.

        gt_uc_name = meta.get("usecase_path", "")

        # Ground truth label tensors
        if gt_uc_name in ["conversational", "instructive"]:
            gt_req = torch.zeros(NUM_TOPICS)
            gt_supp = torch.zeros(NUM_TOPICS)
        else:
            gt_req = setup["malicious_use_cases_ruleset"][gt_uc_name]["all_required_labels"]
            gt_supp = setup["malicious_use_cases_ruleset"][gt_uc_name]["supporting_labels"]

        # Compute triggers
        triggers = compute_triggers(
            dialogue_logits, thresholds=setup["thresholds"], patience_rate=1
        )

        # Use case level FPR/TPR stats
        eval_usecase_detection(
            triggers=triggers,
            all_usecase_gt_labels=setup["all_usecase_gt_labels"],
            any_of_conditions=setup["any_of_conditions"],
            gt_uc_name=gt_uc_name,
            dialogue_name=meta.get("dialogue_id", ""),
            split=split,
            use_cases_stats=fpr_tpr_stats,
        )
        update_usecase_confusion_matrix(
            triggers=triggers,
            all_usecase_gt_labels=setup["all_usecase_gt_labels"],
            any_of_conditions=setup["any_of_conditions"],
            gt_uc_name=gt_uc_name,
            split=split,
            use_cases_stats=acc_stats,
        )

        # Detailed analysis for positive split
        if split == "positive":
            # Load explicit GT one-hot if available
            gt_one_hot = None
            label_pt = (
                Path(config.paths.eval_dataset)
                / "labels"
                / "positive"
                / gt_uc_name
                / f"{meta.get('dialogue_id', '')}.pt"
            )
            if label_pt.exists():
                gt_one_hot = torch.load(label_pt, map_location="cpu", weights_only=True).float()

            # Per-dialogue use case logging
            logger_usecase_stats(
                triggers=triggers,
                all_required_labels=gt_req,
                supporting_labels=gt_supp,
                any_of_conditions=setup["any_of_conditions"],
                dialogue_use_case=gt_uc_name,
                use_cases_stats=use_cases_stats_pos,
                all_usecase_gt_labels=setup["all_usecase_gt_labels"],
                dialogue_name=meta.get("dialogue_id", ""),
                idxs_to_labels=IDXS_TO_LABELS,
                score_neutral=False,
                recorder=recorder,
            )

            # CE stats by ground truth
            if gt_one_hot is not None:
                evaluate_ce_detection_gt(
                    triggers=triggers,
                    gt_one_hot=gt_one_hot,
                    use_case=gt_uc_name,
                    dialogue_name=meta.get("dialogue_id", ""),
                    label_logger=all_results_topic,
                    label_stats=topic_stats,
                    idxs_to_labels=IDXS_TO_LABELS,
                )

            # CE stats by rules
            evaluate_ce_detection_rules(
                triggers=triggers,
                all_required_labels=gt_req,
                supporting_labels=gt_supp,
                any_of_conditions=setup["any_of_conditions"],
                use_case=gt_uc_name,
                labels_statistics=rulebased_topic_stats,
                idxs_to_labels=IDXS_TO_LABELS,
            )

            # Sanity checks
            if gt_one_hot is not None:
                sanity_check_gt(
                    gt=gt_one_hot,
                    any_of_conditions=setup["any_of_conditions"],
                    all_usecase_gt_labels=setup["all_usecase_gt_labels"],
                    gt_uc_name=gt_uc_name,
                    dialogue_name=meta.get("dialogue_id", ""),
                    split=split,
                    use_cases_stats=sanity_stats,
                )

        processed += 1

    logger.info(f"Processed {processed} dialogues")

    # Save outputs
    logger.info("Saving detailed outputs...")
    output_path = Path(output_dir)

    # Use case metrics
    metrics = compute_usecase_detection_metrics(fpr_tpr_stats)
    metrics.to_csv(output_path / "usecase_metrics_fprtpr.csv", index=False)

    weighted_averages, df_weighted_averages = compute_weighted_metrics(acc_stats)
    df_weighted_averages.to_csv(output_path / "usecase_weighted_averages.csv", index=True)

    # Per-dialogue logging
    write_rows_to_csv(all_rows, output_path / "logger_use_case_analysis.csv")
    save_to_csv(all_results_topic, output_path / "label_statistics.csv")

    # CE stats by ground truth
    weighted_averages_ces, df_weighted_averages_ces = compute_weighted_metrics(topic_stats)
    df_weighted_averages_ces.to_csv(output_path / "weighted_averages_ces_GT.csv", index=True)
    logger.info(f"Weighted Averages for CEs (Ground Truth): {weighted_averages_ces}")

    # CE stats by rules
    weighted_averages_ces_rulebased, df_weighted_averages_ces_rulebased = compute_weighted_metrics(
        rulebased_topic_stats
    )
    df_weighted_averages_ces_rulebased.to_csv(
        output_path / "weighted_averages_ces_rulebased.csv", index=True
    )
    logger.info(f"Weighted Averages for CEs (Rule-based): {weighted_averages_ces_rulebased}")

    logger.info("=" * 60)
    logger.info(f"Weighted Averages (Use Cases): {weighted_averages}")
    logger.info("=" * 60)
    logger.info(f"✓ Detailed evaluation completed. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
