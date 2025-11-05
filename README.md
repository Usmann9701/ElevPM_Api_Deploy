\# Elevator Predictive Maintenance API 



A FastAPI-based predictive maintenance app that:

\- Loads an ML model trained on elevator vibration data.

\- Provides endpoints for prediction, anomaly detection, and 30-step forecasting.



\## Endpoints



| Route | Description |

|--------|--------------|

| `/` | API status check |

| `/predict` | Runs the model, returns predictions, MAE, R², and anomalies |

| `/forecast` | Generates 30-step forecast with confidence intervals |



\## Local Run

```bash

uvicorn main:app --reload



