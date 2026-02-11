# GAVEL Installation and Usage

## Quick Setup

You have two options to make the `gavel` package importable:

### Option 1: Install in Editable Mode (Recommended)

This installs the package in your virtual environment so it's always available:

```bash
# Activate your virtual environment
source .venv/bin/activate

# Install gavel package in editable mode
pip install -e .
```

After this, you can run the pipeline normally:
```bash
./run.sh
```

### Option 2: Use PYTHONPATH (No Installation)

The updated `run.sh` now sets `PYTHONPATH` automatically, so you can just run:

```bash
./run.sh
```

Or for individual steps:
```bash
export PYTHONPATH=$(pwd)
python scripts/train.py
python scripts/extract_logits.py
python scripts/calibrate.py
python scripts/evaluate.py
```

## Running the Pipeline

**Complete pipeline:**
```bash
./run.sh
```

**Individual steps:**
```bash
python scripts/train.py          # Step 1: Training
python scripts/extract_logits.py # Step 2: Extract logits
python scripts/calibrate.py      # Step 3: Calibration
python scripts/evaluate.py       # Step 4: Evaluation
```

## Notebooks

To use the package in Jupyter notebooks:

1. If you installed via `pip install -e .`:
   ```python
   from gavel.models import TopicRNN
   from gavel.training.utils import LABELS
   ```

2. If using PYTHONPATH, add to your notebook:
   ```python
   import sys
   sys.path.insert(0, '/path/to/attention_based_classification')
   from gavel.models import TopicRNN
   ```
