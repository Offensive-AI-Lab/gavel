"""Evaluation utilities for GAVEL."""

from gavel.evaluation.calibration import calibrate, calibrate_from_logits
from gavel.evaluation.metrics import (
    compute_triggers,
    compute_usecase_auc,
    evaluate,
    evaluate_from_logits,
    evaluate_usecase_detection_from_logits,
    initialize_usecase_stats,
    load_evaluation_setup,
    plot_usecase_metrics_table,
)
from gavel.models import load_trained_classifier
from gavel.utils.io import iter_dialogue_files

__all__ = [
    "load_trained_classifier",
    "compute_triggers",
    "iter_dialogue_files",
    "plot_usecase_metrics_table",
    "calibrate",
    "calibrate_from_logits",
    "load_evaluation_setup",
    "evaluate_usecase_detection_from_logits",
    "compute_usecase_auc",
    "evaluate",
    "evaluate_from_logits",
    "initialize_usecase_stats",
]
