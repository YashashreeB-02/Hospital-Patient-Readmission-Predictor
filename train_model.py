import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from preprocess import load_and_preprocess

def train():
    print("=" * 60)
    print("🏥 Training Hospital Patient Readmission Predictor Model")
    print("=" * 60)

    dataset_path = "diabetic_data.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset '{dataset_path}' not found in current directory.")

    print(f"Loading and preprocessing dataset from '{dataset_path}'...")
    X, y, feature_columns, encoders = load_and_preprocess(dataset_path)
    print(f"Dataset successfully prepared: {len(X):,} patient encounters with {X.shape[1]} features.")

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train size: {len(X_train):,} samples | Test size: {len(X_test):,} samples")

    # Regularized Gradient Boosted Decision Trees
    print("\nTraining regularized Gradient Boosting model...")
    model = HistGradientBoostingClassifier(
        max_iter=180,
        max_depth=6,
        learning_rate=0.08,
        min_samples_leaf=20,
        l2_regularization=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluation
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1]

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    auc_score = roc_auc_score(y_test, test_prob)

    print("\n" + "=" * 60)
    print(f"✅ Training Complete!")
    print(f"Train Accuracy: {train_acc:.4f} ({train_acc * 100:.2f}%)")
    print(f"Test Accuracy:  {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"ROC-AUC Score:  {auc_score:.4f}")
    print("=" * 60)

    print("\nClassification Report:")
    print(classification_report(y_test, test_pred, target_names=["No Readmission (0)", "Readmitted (1)"], digits=4))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, test_pred)
    print(f"TN: {cm[0,0]:<6} | FP: {cm[0,1]:<6}")
    print(f"FN: {cm[1,0]:<6} | TP: {cm[1,1]:<6}")

    # Save model bundle
    bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "encoders": encoders,
        "metrics": {
            "train_accuracy": round(float(train_acc), 4),
            "test_accuracy": round(float(test_acc), 4),
            "roc_auc": round(float(auc_score), 4)
        }
    }

    output_model_path = "readmission_model.pkl"
    joblib.dump(bundle, output_model_path)
    print(f"\nModel artifact and feature encoders successfully saved to '{output_model_path}'.")

if __name__ == "__main__":
    train()
