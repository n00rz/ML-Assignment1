# Insurance ML App

A small full-stack machine learning app built on the Kaggle
[Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance).

| Part | What it does |
|------|--------------|
| **Regression** (Linear Regression) | Predicts the insurance **charges** (a number) from age, sex, BMI, children, smoker, region |
| **Classification** (Random Forest) | Predicts whether a person is a **smoker** (yes/no) from age, sex, BMI, children, region, charges |
| **Backend** (FastAPI) | `POST /predict/regression` and `POST /predict/classification` |
| **Frontend** (HTML) | One page with a form for each model |

## Project structure
```
data/insurance.csv        <- download from Kaggle (see below)
train.py                  <- trains and saves both models
main.py                   <- FastAPI backend (also serves the page)
static/index.html         <- frontend
requirements.txt
```

## How to run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Download the dataset** from the Kaggle link above and save it as `data/insurance.csv`.
3. **Train the models** (prints R² and accuracy, creates `regression_model.joblib` and `classification_model.joblib`):
   ```bash
   python train.py
   ```
4. **Start the backend** (keep this terminal open):
   ```bash
   uvicorn main:app --reload
   ```
5. **Open the app** from a second terminal (the backend must keep running in the first):
   ```bash
   start static/index.html        # Windows
   open static/index.html         # Mac
   ```

API docs are available at http://127.0.0.1:8000/docs.

Run all commands from the project root folder. Do not run `main.py` directly; the server is started by `uvicorn`.

## Results
Test set = 20% of the data.

- Linear Regression R²: **FILL_IN**
- Random Forest accuracy: **FILL_IN**

## Example API calls
```bash
curl -X POST http://127.0.0.1:8000/predict/regression \
  -H "Content-Type: application/json" \
  -d '{"age":35,"sex":"male","bmi":27.5,"children":1,"smoker":"no","region":"northeast"}'

curl -X POST http://127.0.0.1:8000/predict/classification \
  -H "Content-Type: application/json" \
  -d '{"age":35,"sex":"male","bmi":27.5,"children":1,"region":"northeast","charges":30000}'
```
