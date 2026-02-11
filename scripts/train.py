# train.py
"""Train the GAVEL RNN classifier."""

import argparse
import os

import torch

from gavel.config import load_config
from gavel.models import TopicRNN, train_rnn_model
from gavel.training import (
    create_dataloaders_for_sequences,
    create_dataloaders_from_directory,
    extract_per_sequence_reps,
    load_model_and_tokenizer,
    split_dataset_into_train_val,
)
from gavel.training.utils import _head_geometry  # Internal function
from gavel.utils import cleanup_embeddings
from gavel.utils.logging import add_verbose_arg, setup_logger

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train RNN model")
add_verbose_arg(parser)
parser.add_argument(
    "--config",
    "-c",
    default="config.json",
    help="Path to configuration file (default: config.json)",
)
args = parser.parse_args()

# Setup logger
logger = setup_logger(__name__, verbose=args.verbose)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------
# 1) Load configuration
# ---------------------------
logger.info(f"Loading configuration from {args.config}...")
config = load_config(args.config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extract key values from config
base_directory = config.paths.base_dir
train_data_directory = config.paths.train_dataset
labels = config.labels
num_classes = config.num_labels
selected_layers = config.model.selected_layers

logger.debug(f"Base directory: {base_directory}")
logger.debug(f"Training dataset: {train_data_directory}")
logger.debug(f"Number of labels: {num_classes}")
logger.debug(f"Selected layers: {list(selected_layers)}")

# ---------------------------
# 2) Load base LM + tokenizer (for extraction)
# ---------------------------
logger.info(f"STEP 1/4: Loading base model and tokenizer: {config.model.name_or_path}")
model, tokenizer = load_model_and_tokenizer(config.model.name_or_path)

# ---------------------------
# 3) Train/Val split for the original text dataset
# ---------------------------
logger.info("STEP 2/4: Splitting dataset into train/val...")
split_dataset_into_train_val(
    dataset_root_path=train_data_directory,
    train_ratio=0.8,
    random_seed=42,
)

# Build text dataloaders
text_dataloaders = create_dataloaders_from_directory(
    base_directory=train_data_directory,
    tokenizer=tokenizer,
    batch_size=config.training.batch_size_text,
    max_length=config.training.max_length,
)
train_text_loaders = text_dataloaders["train_dataloaders"]
val_text_loaders = text_dataloaders["val_dataloaders"]

# ---------------------------
# 4) EXTRACT per-sequence reps (assistant span only)
# ---------------------------
seq_out_train = os.path.join(config.paths.embeddings_dir, "train")
seq_out_val = os.path.join(config.paths.embeddings_dir, "val")

try:
    logger.info("STEP 3/4: Extracting per-sequence representations...")
    extract_per_sequence_reps(
        dataloaders=train_text_loaders,
        model=model,
        tokenizer=tokenizer,
        selected_layers=selected_layers,
        save_root=seq_out_train,
        dtype=torch.float16,
    )
    logger.debug("Extracting per-sequence representations for validation set...")
    extract_per_sequence_reps(
        dataloaders=val_text_loaders,
        model=model,
        tokenizer=tokenizer,
        selected_layers=selected_layers,
        save_root=seq_out_val,
        dtype=torch.float16,
    )

    dataloaders_new, class_counts, used_min = create_dataloaders_for_sequences(
        base_directory=base_directory,
        labels=labels,
        batch_size=config.training.batch_size,
        sequence_length=config.rnn.sequence_length,
        seed=42,
        num_workers=4,
    )

    logger.debug(f"Dataset Sizes After Stratification: {class_counts}")
    logger.debug(f"Per-split min used: {used_min}")

    # ---------------------------
    # 5) Build and train RNN model
    # ---------------------------
    # Compute input_dim for TopicRNN from the model head geometry
    _, n_v_heads, head_dim, _ = _head_geometry(model)
    readout_dim = n_v_heads * head_dim

    rnn_model = TopicRNN(
        num_layers=len(selected_layers),
        input_dim=readout_dim,
        hidden_dim=config.rnn.hidden_dim,
        num_rnn_layers=config.rnn.num_rnn_layers,
        num_topics=num_classes,
        rnn_type=config.rnn.rnn_type,
        proj_dim=config.rnn.proj_dim,
    ).to(device)

    logger.info("STEP 4/4: Starting RNN training...")
    trained_rnn = train_rnn_model(
        model=rnn_model,
        labels_dict=labels,
        train_loader=dataloaders_new["train"],
        val_loader=dataloaders_new["val"],
        epochs=config.training.epochs,
        train_class_counts=class_counts["train"],
        val_class_counts=class_counts["val"],
        checkpoint_dir=os.path.join(config.paths.model_dir, "checkpoints"),
        learning_rate=config.training.learning_rate,
        patience=config.training.patience,
        early_stopping=config.training.early_stopping,
        use_wandb=config.training.use_wandb,
    )

    torch.save(trained_rnn.state_dict(), config.paths.rnn_model_path)
    logger.info(f"Training complete! Model saved at: {config.paths.rnn_model_path}")


finally:
    # Cleanup embeddings directory if configured
    if config.training.cleanup_embeddings:
        cleanup_embeddings(config.paths.embeddings_dir, logger)
