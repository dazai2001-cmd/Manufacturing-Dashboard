# Manufacturing Predictive Maintenance Dashboard

Streamlit dashboard for testing predictive and prescriptive maintenance logic with the UCI AI4I 2020 Predictive Maintenance dataset.

The app replays held-out AI4I test rows as live records, predicts failure probability with a supervised model, compares the prediction with the actual test label, and explains likely maintenance reasons with an interpretable rules layer.

## Project Structure

```text
.
├── app.py                              # Streamlit entrypoint
├── assets/
│   └── alert.mp3                       # Optional alert sound
├── data/
│   └── ai4i2020.csv                    # UCI AI4I dataset
├── src/
│   └── manufacturing_dashboard/
│       ├── analytics.py                # Prescriptive maintenance scoring and recommendations
│       ├── dashboard.py                # Streamlit UI
│       ├── data.py                     # Dataset loading, train/test split, live replay
│       └── model.py                    # Random Forest model training and prediction
└── requirements.txt
```

## Data And Split

The dataset is split once with a fixed random seed:

- 70% training rows
- 30% held-out test rows
- stratified by `Machine failure`
- zero overlap between train and test rows

The dashboard live feed uses only the held-out test split. The model trains only on the training split.

## Model

The current predictive model is a balanced `RandomForestClassifier`.

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

The model predicts failure probability before the actual held-out label is shown. The dashboard then displays whether the prediction was a correct failure, correct normal, false positive, or missed failure.

## Prescriptive Logic

The model predicts risk. The reason and recommendation come from a separate interpretable rules layer in `analytics.py`.

It checks conditions such as:

- high tool wear
- high process-to-air temperature gap
- abnormal power load
- overstrain from torque and tool wear
- derived bearing, vibration, lubrication, and hydraulic stress indicators

This keeps the prediction model and the explanation logic separate.

## Run Locally

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start the dashboard:

```powershell
streamlit run app.py
```

Then open:

```text
http://127.0.0.1:8501
```

## Notes

AI4I is tabular process data, not true per-machine time-series telemetry. A Random Forest is therefore a better fit than an LSTM for this dataset. An LSTM would make more sense with continuous timestamped sensor histories per physical machine.
