# Manufacturing Predictive Maintenance Dashboard

A Streamlit dashboard for predictive and prescriptive maintenance using the UCI AI4I 2020 Predictive Maintenance dataset.

The app is designed to demonstrate how manufacturing teams can:

- replay held-out machine records as live sensor/KPI readings
- predict whether an incoming machine state is likely to fail
- compare the prediction with the actual held-out failure label
- explain which machine conditions increased failure probability
- recommend maintenance actions based on risk, evidence, urgency, and cost impact

## Current Status

The current version uses the UCI AI4I dataset, a saved Random Forest model, a 70/30 train/test split, manual test-row replay, and a separate prescriptive analytics layer.

The dashboard is intentionally set up so the model predicts first, and the actual test outcome is shown afterward for validation.

## Features

- Streamlit web dashboard
- UCI AI4I 2020 dataset integration
- 70% train / 30% held-out test split
- Live-style replay using only held-out test rows
- Product type labels from the dataset: `H`, `M`, `L`
- Offline model training script
- Saved model artifact with `joblib`
- Model comparison report
- Tuned failure threshold
- Random Forest failure probability prediction
- Prediction-vs-actual evaluation cards
- Local feature contribution explanations
- Prescriptive maintenance reasons and recommended actions
- Estimated downtime, failure cost, preventive cost, and cost avoided
- Streamlit config for localhost preview stability

## Project Structure

```text
.
|-- app.py
|-- .streamlit/
|   `-- config.toml
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
|       |-- __init__.py
|       |-- analytics.py
|       |-- dashboard.py
|       |-- data.py
|       |-- model.py
|       `-- training.py
|-- requirements.txt
`-- README.md
```

## Main Files

### `app.py`

Streamlit entrypoint.

It adds `src/` to `sys.path`, then executes `dashboard.py` with `runpy.run_path()`.

This is important because Streamlit reruns `app.py` in the same Python process. A normal import such as:

```python
import manufacturing_dashboard.dashboard
```

can be cached by Python and may not rerun the dashboard UI code on later browser refreshes. The current entrypoint avoids that by executing the dashboard file each Streamlit run.

### `.streamlit/config.toml`

Local Streamlit preview configuration:

```toml
[server]
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
```

This helps the local preview connect reliably to the Streamlit backend.

### `src/manufacturing_dashboard/data.py`

Loads and prepares the AI4I dataset.

Responsibilities:

- read `data/ai4i2020.csv`
- rename dataset columns into Python-friendly names
- create derived values such as Celsius temperatures and power load
- create `machine_type` from the AI4I product type
- create human-readable failure type labels
- split the dataset into train and test rows
- replay held-out test rows as dashboard records

Important split settings:

```text
train fraction: 70%
test fraction: 30%
random_state: 42
stratified by: machine_failure
```

### `src/manufacturing_dashboard/training.py`

Contains the offline training and evaluation pipeline.

The training pipeline:

1. Loads the 70% training split.
2. Creates an internal validation split from the training data.
3. Trains candidate models.
4. Tunes the classification threshold on validation data.
5. Selects the best model by validation F2 score.
6. Retrains the selected model on the full 70% training split.
7. Evaluates on the 30% held-out test split.
8. Saves the model artifact and reports.

Candidate models:

```text
Logistic Regression
Random Forest
Histogram Gradient Boosting
```

### `scripts/train_model.py`

Command-line script for training, comparing, tuning, and saving the model.

Run:

```powershell
python .\scripts\train_model.py
```

Outputs:

```text
models/failure_model.joblib
reports/model_metrics.json
reports/model_comparison.csv
```

### `src/manufacturing_dashboard/model.py`

Loads the saved model artifact and performs prediction.

Responsibilities:

- load `models/failure_model.joblib`
- predict failure probability
- apply the tuned threshold
- estimate downtime
- generate model-facing explanation text
- compute local feature contribution values against training baselines
- provide model diagnostics to the dashboard

### `src/manufacturing_dashboard/analytics.py`

Prescriptive analytics layer.

This is separate from the machine learning model.

The model answers:

```text
Is this incoming row likely to fail?
```

The prescriptive layer answers:

```text
Why might maintenance be needed?
What action should be taken?
How urgent is it?
What cost could preventive action avoid?
```

It checks conditions such as:

- tool wear approaching failure range
- heat dissipation stress
- abnormal power load
- overstrain from torque and accumulated tool wear
- derived oil, hydraulic, bearing, and vibration stress
- efficiency and defect-rate drift
- energy usage per unit

### `src/manufacturing_dashboard/dashboard.py`

Main Streamlit UI.

It displays:

- replay controls
- overall manufacturing KPIs
- selected machine KPIs
- predictive failure probability
- tuned threshold
- model diagnostics
- feature contribution explanation table
- predicted-vs-actual test outcome
- prescriptive maintenance risk
- cost impact estimates
- risk chart and recommendation table
- gauges, trends, alerts, and recent records

## Dataset

The app expects:

```text
data/ai4i2020.csv
```

The dataset is the UCI AI4I 2020 Predictive Maintenance dataset.

Important input features:

```text
air_temperature_k
process_temperature_k
rotational_speed_rpm
torque_nm
tool_wear_min
```

Target:

```text
machine_failure
```

Failure type columns:

```text
TWF - Tool wear failure
HDF - Heat dissipation failure
PWF - Power failure
OSF - Overstrain failure
RNF - Random failure
```

## Train/Test Logic

The project uses a fixed 70/30 split:

```text
70% training data
30% held-out test data
```

The dashboard replay uses only the held-out test split.

This means the dashboard can show whether the prediction was correct without leaking the actual outcome into the model.

Flow:

```text
AI4I dataset
-> fixed 70/30 split
-> model trains on 70%
-> dashboard replays 30% test rows
-> model predicts failure probability
-> dashboard then shows actual test outcome
-> dashboard labels the result as correct failure, missed failure, false positive, or correct normal
```

## Saved Model

Current saved model:

```text
Random Forest
```

The Random Forest was selected because it achieved the best validation F2 score.

F2 gives recall more weight than precision. That is useful for predictive maintenance because missing a real failure is usually more expensive than checking a machine early.

Current tuned threshold:

```text
51.5%
```

## Current Metrics

Held-out test metrics from `reports/model_metrics.json`:

```text
accuracy  = 96.8%
precision = 51.9%
recall    = 79.4%
F1        = 62.8%
F2        = 71.8%
ROC-AUC   = 97.3%
```

Confusion matrix on the held-out test split:

```text
true negatives  = 2823
false positives = 75
false negatives = 21
true positives  = 81
```

Validation comparison:

```text
Random Forest                F2 77.1%, Recall 83.0%, Precision 60.0%
Histogram Gradient Boosting  F2 73.9%, Recall 72.3%, Precision 81.0%
Logistic Regression          F2 53.9%, Recall 61.7%, Precision 35.8%
```

## Running Locally

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start the dashboard from the repository root:

```powershell
streamlit run app.py
```

Open:

```text
http://127.0.0.1:8501
```

If port `8501` is already busy, run on another port:

```powershell
streamlit run app.py --server.port 8502
```

## Dashboard Usage

### Step One Row

Advances the held-out test replay by one row for each machine type.

Use this when you want to verify:

- the model predicts first
- the actual held-out label appears afterward
- the prediction result changes as new test data arrives

### Reset Replay

Clears replay history and restarts the held-out replay cursor.

### Machine Selector

Selects one AI4I product type:

```text
H
M
L
```

The selected machine controls the gauges, prediction cards, prescriptive recommendation, and recent records table.

## Why The Replay Is Manual

Earlier versions used a blocking `time.sleep()` plus `st.rerun()` loop.

That could leave Streamlit stuck on a blank shell during reruns, especially inside preview panes.

The current dashboard uses manual replay controls instead. This keeps the page visible and makes model validation easier because each row advance is deliberate.

## Predictive vs Prescriptive Analytics

Predictive analytics:

```text
Will this incoming machine state fail?
```

Prescriptive analytics:

```text
Why is maintenance needed, when should it happen, and what action should be taken?
```

The machine learning model and the prescriptive layer are intentionally separate. The model predicts failure probability from AI4I input features. The prescriptive layer turns current readings, trends, thresholds, and costs into a maintenance recommendation.

## Why Random Forest Instead Of LSTM?

AI4I is tabular process data, not continuous per-machine time-series telemetry.

A Random Forest, Gradient Boosting model, or Logistic Regression model is a better fit for this dataset because each row is an independent machine-state record.

An LSTM would make more sense if the project had real timestamped sensor histories for each physical machine, where the sequence of readings over time carried predictive signal.

## Troubleshooting

### Blank Streamlit Page

If the page shows only the Streamlit shell or `Deploy` button:

1. Make sure the current `app.py` uses `runpy.run_path(...)`.
2. Restart the Streamlit process.
3. Refresh the browser.
4. Confirm `http://127.0.0.1:8501` returns HTTP `200`.

Check the port:

```powershell
netstat -ano | Select-String ':8501'
```

Check the owner:

```powershell
Get-Process -Id <PID>
```

### Port Already In Use

Run:

```powershell
netstat -ano | Select-String ':8501'
```

If another process owns the port, either stop that process or run Streamlit on a different port:

```powershell
streamlit run app.py --server.port 8502
```

### Model Artifact Missing

If `models/failure_model.joblib` is missing, regenerate it:

```powershell
python .\scripts\train_model.py
```

### Dataset Missing

The app expects:

```text
data/ai4i2020.csv
```

If the dataset is missing, the app falls back to synthetic machine data, but the AI4I predictive model path will not be available.

## Limitations

- AI4I is not a true machine time-series dataset.
- The replay simulates live data by stepping through held-out test rows.
- The prescriptive reason layer is rule-based and interpretable, not trained directly from maintenance work orders.
- Feature contribution explanations are local probability-delta explanations, not full SHAP values.
- Cost values are illustrative assumptions for dashboard demonstration.

## Suggested Next Improvements

- Add real sensor time-series data.
- Add maintenance work-order history.
- Add operator notes and downtime reason codes.
- Add SHAP if exact model explainability is required.
- Add a database for persistent machine history.
- Add automated tests around train/test split integrity.
- Add a CI workflow that runs model training smoke tests and Streamlit import checks.
