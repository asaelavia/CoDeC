# CODEC: Constraints Guided Diverse Counterfactuals

CODEC generates diverse counterfactual explanations for binary classification models while respecting domain constraints. It helps users understand what changes would lead to different classification outcomes.

## Installation & Usage

Install required packages:
```bash
pip install -r requirements.txt
```

Run the GUI application:
```bash
python gui.py
```

## Interface Overview

### Input Parameters Tab
![Input Parameters Tab](Input_tab3.png)

- **Left**: Load dataset (CSV) and constraints file, set number of counterfactuals
- **Center**: Dataset preview with color-coded classifications (green=positive, red=negative)
- **Right**: Configure instance attributes and select immutable features

### Constraints Tab
![Constraints Tab](constraints_tab.png)

Displays active denial constraints that ensure counterfactual feasibility.

### Results Tab
![Results Tab](results_tab_small.PNG)

Shows generated counterfactuals with:
- Changed attributes highlighted with arrows (→)
- Diversity score (DPP) measuring solution variety
- Distance scores showing proximity to original instance

## Quick Start

1. Load a binary classification dataset (CSV with 'label' column)
2. Load denial constraints file (optional)
3. Configure the instance to explain
4. Mark any immutable attributes
5. Click "Generate Counterfactuals"

## Requirements

- Binary classification dataset with 'label' column (0/1)
- Python environment with dependencies from requirements.txt
- Pre-trained model or training data