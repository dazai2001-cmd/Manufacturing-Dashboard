# Manufacturing Predictive Maintenance Dashboard

Streamlit dashboard for predictive and prescriptive maintenance using the UCI AI4I 2020 Predictive Maintenance dataset.

The project replays held-out test rows as live machine readings, predicts whether each incoming record is likely to fail, compares the prediction with the actual held-out label, explains the strongest failure drivers, and recommends maintenance actions with estimated cost impact.

## What The App Does

1. Loads the AI4I dataset from `data/ai4i2020.csv`.
2. Splits the dataset into 70% training rows and 30% held-out test rows.
3. Trains and compares multiple tabular classifiers.
4. Selects the best model using validation F2 score, which gives recall more weight than precision.
5. Saves the trained model to `models/failure_model.joblib`.
6. Replays only the held-out test rows in the dashboard as live KPI data.
7. Predicts failure probability before showing the actual test outcome.
8. Explains likely failure drivers and recommends maintenance actions.
9. Estimates expected downtime, preventive cost, failure cost, and cost avoided.

## Project Structure

```text
.
|-- app.py
|-- assets/
|   `-- alert.mp3
|-- data/
|   `-- ai4i2020.csv
|-- models/
|   `-- failure_model.joblib
|-- reports/
|   |-- model_comparison.csv
|   `-- model_metrics.json
|-- scripts/
|   `-- train_model.py
|-- src/
|   `-- manufacturing_dashboard/
|       |-- analytics.py
|       |-- dashboard.py
|       |-- data.py
|       |-- model.py
|       |-- training.py
|       `-- __init__.py
`-- requirements.txt
```

## Data Split

The split is fixed and reproducible:

```text
70% train
30% held-out test
stratified by Machine failure
random_state = 42
```

The dashboard live replay uses only the 30% test split. The saved model is trained only on the 70% training split.

## Predictive Model

The training script compares:

```text
Logistic Regression
Random Forest
Histogram Gradient Boosting
```

The current saved model is:

```text
Random Forest
```

It was selected because it produced the best validation F2 score. F2 is used because missed failures are usually more costly than early inspections.

Target:

```text
machine_failure
```

Features:

```text
air_temperature_k
process_temperature_k
rotational_speed_rpm
torque_nm
tool_wear_min
```

Current held-out test metrics:

```text
accuracy  = 96.8%
precision = 51.9%
recall    = 79.4%
F2        = 71.8%
ROC-AUC   = 97.3%
threshold = 51.5%
```

## Prescriptive Logic

The prediction model answers:

```text
Is this machine likely to fail?
```

The prescriptive layer answers:

```text
Why is it risky?
What should maintenance do?
How urgent is it?
What cost could preventive action avoid?
```

The prescriptive logic checks:

- tool wear approaching failure range
- heat dissipation stress
- abnormal power load
- overstrain from torque and tool wear
- bearing temperature and vibration trends
- lubrication and hydraulic stress
- efficiency, defect rate, and energy per unit

The dashboard also uses local feature contribution logic to show which model inputs raised or lowered the predicted failure probability against the training baseline.

## Live Replay Controls

The dashboard includes controls for:

- pause/play replay
- replay speed
- step one held-out row at a time
- reset replay history

This makes it easier to verify that the model predicts before the actual held-out failure label is shown.

## Train The Model

Install dependencies:

```powershell
pip install -r requirements.txt
```

Train, compare, tune, and save the model:

```powershell
python .\scripts\train_model.py
```

This updates:

```text
models/failure_model.joblib
reports/model_metrics.json
reports/model_comparison.csv
```

## Run The Dashboard

```powershell
streamlit run app.py
```

Open:

```text
http://127.0.0.1:8501
```

## Why Random Forest Instead Of LSTM?

AI4I is tabular process data, not true per-machine timestamped telemetry. Random Forest, Logistic Regression, and Gradient Boosting are better fits for this dataset.

An LSTM would be a stronger candidate if the project had continuous sensor histories for each physical machine, where the order and shape of readings over time carried predictive signal.
