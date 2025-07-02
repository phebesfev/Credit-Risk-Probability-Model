# src/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE

def load_and_split_data(path):
    df = pd.read_csv(path)
    drop_cols = [
        "TransactionId", "BatchId", "AccountId", "SubscriptionId", "CustomerId",
        "TransactionStartTime", "CurrencyCode", "CountryCode", "FraudResult"
    ]
    X = df.drop(columns=drop_cols).fillna(0)
    y = df["FraudResult"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test

def preprocess_data(X_train, X_test, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    return X_train_res, X_test_scaled, y_train_res

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    return rf

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    # Optional: print report for local debugging
    print(f"\n=== Model Evaluation ===")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k}: {v:.4f}")
    print("Confusion Matrix:\n", metrics["confusion_matrix"])

    return metrics


if __name__ == "__main__":
    # Quick run example
    X_train, X_test, y_train, y_test = load_and_split_data('../data/processed/pdata.csv')
    X_train_res, X_test_scaled, y_train_res = preprocess_data(X_train, X_test, y_train)
    model = train_random_forest(X_train_res, y_train_res)
    evaluate_model(model, X_test_scaled, y_test)
