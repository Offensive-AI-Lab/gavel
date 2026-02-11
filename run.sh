#!/bin/bash
# GAVEL End-to-End Pipeline Runner
# This script runs the simplified pipeline: train -> evaluate
set -e  # Exit on any error

# Ensure we're running from the repository root
cd "$(dirname "$0")"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

# Step 1: Training
echo "Starting Training..."
python scripts/train.py
echo ""

# Step 2: Evaluation
echo "Starting Evaluation..."
python scripts/evaluate.py

echo ""
echo "Pipeline execution complete!"