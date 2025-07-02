# tests/test_train.py

import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import load_and_split_data, train_random_forest, evaluate_model

def test_load_and_split_data_shapes(tmp_path):
    df = pd.DataFrame({
        "TransactionId": [1, 2, 3, 4, 5, 6],
        "CustomerId": [10, 11, 12, 13, 14, 15],
        "Amount": [100, 200, 150, 120, 130, 140],
        "Value": [10, 20, 15, 12, 13, 14],
        "CurrencyCode": ["KES"] * 6,
        "TransactionStartTime": pd.to_datetime(["2024-01-01"] * 6),
        "FraudResult": [0, 0, 1, 0, 1, 1],  # enough minority class for stratify
        "BatchId": [101, 102, 103, 104, 105, 106],
        "AccountId": [1, 2, 3, 4, 5, 6],
        "SubscriptionId": [111, 112, 113, 114, 115, 116]
    })

    csv_file = tmp_path / "fake_processed.csv"
    df.to_csv(csv_file, index=False)

    X_train, X_test, y_train, y_test = load_and_split_data(str(csv_file))

    assert X_train.shape[0] + X_test.shape[0] == 6
    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})

def test_train_and_evaluate():
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_classes=2,
        weights=[0.9, 0.1],
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    model = train_random_forest(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert "roc_auc" in metrics
