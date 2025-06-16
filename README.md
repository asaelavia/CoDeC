# CODEC: Constraints Guided Diverse Counterfactuals

CODEC generates diverse counterfactual explanations for binary classification models while respecting domain constraints. It helps users understand what changes would lead to different classification outcomes.

## Installation & Usage

To create and activate the environment from the `environment.yml` file:
```bash
conda env create -f environment.yml
conda activate my_env_name
```

Run the GUI application:
```bash
python gui.py
```

## Quick Demo

Try the included example:
1. Load `nyhouse.csv` dataset
2. Load `ny_DCs.txt` for constraints
3. The system will automatically use the pre-trained model provided for this dataset

## Interface Overview

### Input Parameters Tab
![Input Parameters Tab](Input.PNG)

- **Left**: Load dataset (CSV) and show dataset preview. Load constraints file, and open constraints manager. Set number of counterfactuals
- **Right**: Configure instance attributes and select immutable features

### Constraints Tab
![Constraints Tab](constraints.PNG)

Displays active denial constraints that ensure counterfactual feasibility.

### Results Tab
![Results Tab](Results.PNG)

Shows generated counterfactuals with:
- Changed attributes highlighted with arrows (→) and changed font color.
- Diversity score (DPP) measuring solution variety
- Distance scores showing proximity to original instance

Top right choose between DiCE and CoDeC for comparison.
## Quick Start

1. Load a binary classification dataset (CSV with 'label' column)
2. Load denial constraints file
3. Configure the instance to explain
4. Mark any immutable attributes
5. Click "Generate Counterfactuals"

## Model Handling

- **Pre-trained models**: If a matching model exists for your dataset, it loads automatically
- **Automatic training**: For new datasets without pre-trained models, the system will train and save a model automatically

## Denial Constraints Format

Constraints should be specified in a text file using the following format:

```
¬{ t0.type == t1.type ∧ t0.beds > t1.beds ∧ t0.bath > t1.bath ∧ t0.propertysqft < t1.propertysqft }
¬{ t0.type == "Condo_for_sale" ∧ t0.bath >= 7 }
```

Each constraint defines conditions that counterfactuals must not violate.

## Requirements

- Binary classification dataset with 'label' column (0/1)
- Denial constraints text file
- Python environment with dependencies from environment.yml