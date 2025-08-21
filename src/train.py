import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import wandb

# -------------------------
# Load and Split Data
# -------------------------
def load_and_split_data(file_path, test_size=0.2, random_state=42):
    """Load dataset and split into train/test."""
    df = pd.read_csv(file_path)

    # Features, labels, subjects
    X = df.drop(columns=['Subject_Names', 'Class'])
    y = df['Class']
    subject_names = df['Subject_Names']

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_index, test_index = next(sss.split(X, y))

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    subject_names_test = subject_names.iloc[test_index]

    return X_train, X_test, y_train, y_test, subject_names_test


# -------------------------
# Training and Evaluation
# -------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test, subject_names_test):
    """Train RF model, log with W&B, evaluate, and save model."""
    wandb.init(project="BC-JMIR-Singletranscripts")

    config = wandb.config
    sweep_name = wandb.run.sweep_id or "manual"
    run_name = wandb.run.name
    run_dir = os.path.join("saved_models", sweep_name)
    os.makedirs(run_dir, exist_ok=True)

    rf_classifier = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_features=config.max_features,
        random_state=config.random_seed,
        max_depth=config.max_depth,
        criterion=config.criterion
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_scores = cross_val_score(rf_classifier, X_train, y_train, cv=cv, scoring="accuracy")
    precision_scores = cross_val_score(rf_classifier, X_train, y_train, cv=cv, scoring="precision_macro")
    recall_scores = cross_val_score(rf_classifier, X_train, y_train, cv=cv, scoring="recall_macro")

    wandb.log({
        "cv/accuracy_mean": np.mean(accuracy_scores),
        "cv/accuracy_std": np.std(accuracy_scores),
        "cv/precision_mean": np.mean(precision_scores),
        "cv/precision_std": np.std(precision_scores),
        "cv/recall_mean": np.mean(recall_scores),
        "cv/recall_std": np.std(recall_scores)
    })

    # Train final model
    rf_classifier.fit(X_train, y_train)

    # Log feature importances
    feature_importances = rf_classifier.feature_importances_
    features_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)

    wandb.log({"feature_importances": wandb.Table(dataframe=features_df)})

    important_features = features_df["Feature"].iloc[:20].tolist()
    wandb.log({"important_features": important_features})

    # Test set evaluation
    y_pred_test = rf_classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test, average="macro", zero_division=0)
    test_recall = recall_score(y_test, y_pred_test, average="macro")

    wandb.log({
        "test/accuracy": test_accuracy,
        "test/precision": test_precision,
        "test/recall": test_recall
    })

    print(f"Test Accuracy: {test_accuracy:.3f} | Precision: {test_precision:.3f} | Recall: {test_recall:.3f}")

    # Save trained model
    model_path = os.path.join(run_dir, f"rf_model_{run_name}.pkl")
    joblib.dump(rf_classifier, model_path)

    wandb.finish()


# -------------------------
# Main Script
# -------------------------
if __name__ == "__main__":
    data_file_path = "data/madi_single_transcriptome_tcga_TMM_label.csv"  # <-- update with your path
    X_train, X_test, y_train, y_test, subject_names_test = load_and_split_data(data_file_path)

    # Sweep config (can also be in sweep_bc.yaml)
    sweep_config = {
        "method": "random",
        "metric": {"goal": "maximize", "name": "cv/accuracy_mean"},
        "parameters": {
            "n_estimators": {"min": 5, "max": 150},
            "max_features": {"values": ["sqrt", "log2", None]},
            "random_seed": {"values": [42]},
            "max_depth": {"min": 5, "max": 200},
            "criterion": {"values": ["gini", "entropy"]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="BC-JMIR-Singletranscripts")
    wandb.agent(sweep_id, function=lambda: train_and_evaluate(X_train, X_test, y_train, y_test, subject_names_test), count=20)
