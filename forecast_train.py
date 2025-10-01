# src/ml/forecast_train.py
import pandas as pd
import random
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_delay_model(history_csv="data/historical_with_delays.csv", model_out=f"{MODEL_DIR}/delay.xgb"):
    df = pd.read_csv(history_csv)
    # Minimal feature engineering (user should expand)
    df['dep_hour'] = pd.to_datetime(df['dep_time']).dt.hour
    df['dow'] = pd.to_datetime(df['dep_time']).dt.weekday
    # create features (one-hot/encoding omitted for brevity)
    X = df[['dep_hour','dow','flight_time_minutes']].fillna(0)
    y = (df['delay_minutes'] > 30).astype(int)
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {"objective":"binary:logistic","eval_metric":"auc","verbosity":1}
    bst = xgb.train(params, dtrain, num_boost_round=200, evals=[(dval,"val")], early_stopping_rounds=10)
    bst.save_model(model_out)
    print("Saved delay model to", model_out)
    print("Val AUC:", roc_auc_score(y_val, bst.predict(dval)))
    return model_out

if __name__ == "__main__":
    # If you don't have historical_with_delays.csv, create a quick synthetic augmentation from sample data
    sample_sched = pd.read_csv("data/sample_schedule.csv")
    # synthetic history: add random delays
    sample_sched['delay_minutes'] = sample_sched['flight_time_minutes'].apply(lambda x: 30 if random.random()<0.05 else 0)
    sample_sched.to_csv("data/historical_with_delays.csv", index=False)
    train_delay_model("data/historical_with_delays.csv")
