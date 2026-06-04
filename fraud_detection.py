"""
Credit Card Fraud Detection Using Machine Learning
Group 2 - CSC Project
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "creditcard.csv")
PLOT_DIR  = os.path.join(BASE_DIR, "plots")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(PLOT_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── 1. Load Data ─────────────────────────────────────────────────────────────
def load_data():
    print("\n" + "="*60)
    print("  STEP 1: LOADING DATASET")
    print("="*60)
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape      : {df.shape}")
    print(f"Features           : {list(df.columns)}")
    print(f"Missing values     : {df.isnull().sum().sum()}")
    print(f"\nClass distribution :")
    counts = df["Class"].value_counts()
    print(f"  Legitimate (0)   : {counts[0]:,}  ({counts[0]/len(df)*100:.2f}%)")
    print(f"  Fraudulent (1)   : {counts[1]:,}  ({counts[1]/len(df)*100:.4f}%)")
    return df

# ─── 2. EDA ───────────────────────────────────────────────────────────────────
def exploratory_data_analysis(df):
    print("\n" + "="*60)
    print("  STEP 2: EXPLORATORY DATA ANALYSIS")
    print("="*60)

    # 2a. Class distribution bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts = df["Class"].value_counts()
    axes[0].bar(["Legitimate", "Fraudulent"], counts.values,
                color=["steelblue", "crimson"], edgecolor="black")
    axes[0].set_title("Class Distribution")
    axes[0].set_ylabel("Number of Transactions")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 500, f"{v:,}", ha="center", fontweight="bold")

    # 2b. Transaction amount distribution
    axes[1].hist(df[df["Class"]==0]["Amount"], bins=60, alpha=0.6,
                 label="Legitimate", color="steelblue")
    axes[1].hist(df[df["Class"]==1]["Amount"], bins=60, alpha=0.8,
                 label="Fraudulent", color="crimson")
    axes[1].set_title("Transaction Amount Distribution")
    axes[1].set_xlabel("Amount (USD)")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].set_xlim(0, 2500)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "01_class_and_amount.png"), dpi=150)
    plt.close()
    print("  Saved: 01_class_and_amount.png")

    # 2c. Correlation heatmap (top 15 features by absolute correlation with Class)
    corr = df.corr()["Class"].drop("Class").abs().sort_values(ascending=False)
    top_feats = corr.head(15).index.tolist()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[top_feats + ["Class"]].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap – Top 15 Features vs Class")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "02_correlation_heatmap.png"), dpi=150)
    plt.close()
    print("  Saved: 02_correlation_heatmap.png")

    # 2d. Time vs fraud rate
    df_copy = df.copy()
    df_copy["Hour"] = (df_copy["Time"] // 3600) % 24
    hourly = df_copy.groupby("Hour")["Class"].mean() * 100
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hourly.index, hourly.values, marker="o", color="crimson")
    ax.set_title("Fraud Rate by Hour of Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Fraud Rate (%)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "03_fraud_by_hour.png"), dpi=150)
    plt.close()
    print("  Saved: 03_fraud_by_hour.png")

    print(f"\n  Top correlated features with fraud:")
    for feat, val in corr.head(10).items():
        print(f"    {feat:<6}: {val:.4f}")

# ─── 3. Preprocessing ─────────────────────────────────────────────────────────
def preprocess(df):
    print("\n" + "="*60)
    print("  STEP 3: PREPROCESSING & HANDLING CLASS IMBALANCE")
    print("="*60)

    # Scale Amount and Time
    scaler = StandardScaler()
    df = df.copy()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    df["Time"]   = scaler.fit_transform(df[["Time"]])

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train size: {X_train.shape[0]:,}  |  Test size: {X_test.shape[0]:,}")

    # SMOTE on training set only
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE  – Legit: {(y_res==0).sum():,}  |  Fraud: {(y_res==1).sum():,}")

    # Also create undersampled set for comparison
    rus = RandomUnderSampler(random_state=42)
    X_under, y_under = rus.fit_resample(X_train, y_train)
    print(f"  After UnderSampling – Legit: {(y_under==0).sum():,}  |  Fraud: {(y_under==1).sum():,}")

    return X_train, X_test, y_train, y_test, X_res, y_res, X_under, y_under, X.columns.tolist()

# ─── 4. Train & Evaluate Models ───────────────────────────────────────────────
def train_and_evaluate(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_proba)

    print(f"\n  [{name}]")
    print(f"    Precision : {precision:.4f}")
    print(f"    Recall    : {recall:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print(f"    ROC-AUC   : {roc_auc:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=["Legitimate", "Fraudulent"],
                                digits=4))
    return model, y_pred, y_proba, {
        "Model": name, "Precision": precision,
        "Recall": recall, "F1": f1, "ROC_AUC": roc_auc
    }

def run_models(X_res, y_res, X_test, y_test, feature_names):
    print("\n" + "="*60)
    print("  STEP 4: MODEL TRAINING & EVALUATION (SMOTE data)")
    print("="*60)

    models_def = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }

    results  = []
    trained  = {}
    preds    = {}

    for name, mdl in models_def.items():
        mdl, y_pred, y_proba, metrics = train_and_evaluate(
            name, mdl, X_res, y_res, X_test, y_test
        )
        trained[name] = mdl
        preds[name]   = (y_pred, y_proba)
        results.append(metrics)

    return trained, preds, results

# ─── 5. Visualise Results ─────────────────────────────────────────────────────
def visualise_results(trained, preds, results, X_test, y_test, feature_names):
    print("\n" + "="*60)
    print("  STEP 5: VISUALISATIONS")
    print("="*60)

    colors = ["#4C72B0", "#DD8452", "#55A868"]

    # 5a. Model comparison bar chart
    df_res = pd.DataFrame(results)
    metrics = ["Precision", "Recall", "F1", "ROC_AUC"]
    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (_, row) in enumerate(df_res.iterrows()):
        ax.bar(x + i * width, [row[m] for m in metrics],
               width, label=row["Model"], color=colors[i], alpha=0.85, edgecolor="black")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "04_model_comparison.png"), dpi=150)
    plt.close()
    print("  Saved: 04_model_comparison.png")

    # 5b. ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, (_, y_proba)), color in zip(preds.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", color=color, lw=2)
    ax.plot([0,1],[0,1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – All Models")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "05_roc_curves.png"), dpi=150)
    plt.close()
    print("  Saved: 05_roc_curves.png")

    # 5c. Precision-Recall curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, (_, y_proba)), color in zip(preds.items(), colors):
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        ax.plot(rec, prec, label=name, color=color, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves – All Models")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "06_precision_recall_curves.png"), dpi=150)
    plt.close()
    print("  Saved: 06_precision_recall_curves.png")

    # 5d. Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, (y_pred, _)) in zip(axes, preds.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Legit", "Fraud"],
                    yticklabels=["Legit", "Fraud"])
        ax.set_title(name)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "07_confusion_matrices.png"), dpi=150)
    plt.close()
    print("  Saved: 07_confusion_matrices.png")

    # 5e. Random Forest feature importance
    rf = trained["Random Forest"]
    importances = pd.Series(rf.feature_importances_, index=feature_names)
    top15 = importances.sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 5))
    top15.sort_values().plot.barh(ax=ax, color="steelblue", edgecolor="black")
    ax.set_title("Top 15 Feature Importances – Random Forest")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "08_feature_importance.png"), dpi=150)
    plt.close()
    print("  Saved: 08_feature_importance.png")

    return df_res

# ─── 6. Select Best Model & Save ──────────────────────────────────────────────
def select_and_save(trained, df_res):
    print("\n" + "="*60)
    print("  STEP 6: MODEL SELECTION & SAVING")
    print("="*60)
    best_row   = df_res.loc[df_res["F1"].idxmax()]
    best_name  = best_row["Model"]
    best_model = trained[best_name]

    print(f"\n  Best model (by F1-Score): {best_name}")
    print(f"    F1      : {best_row['F1']:.4f}")
    print(f"    ROC-AUC : {best_row['ROC_AUC']:.4f}")

    path = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump(best_model, path)
    print(f"  Saved to : {path}")
    return best_name, best_model

# ─── 7. Prediction System ─────────────────────────────────────────────────────
def predict_transaction(model, transaction_dict, feature_names):
    """
    Accepts a dict of feature values and returns fraud probability + label.
    """
    row = pd.DataFrame([transaction_dict], columns=feature_names)
    prob  = model.predict_proba(row)[0][1]
    label = "FRAUDULENT" if prob >= 0.5 else "LEGITIMATE"
    return label, prob

def demo_prediction(best_model, X_test, y_test, feature_names):
    print("\n" + "="*60)
    print("  STEP 7: PREDICTION SYSTEM DEMO")
    print("="*60)

    # Pick a known fraud and a known legitimate transaction
    fraud_idx  = y_test[y_test == 1].index[0]
    legit_idx  = y_test[y_test == 0].index[0]

    for idx, true_label in [(fraud_idx, "FRAUDULENT"), (legit_idx, "LEGITIMATE")]:
        txn = X_test.loc[idx].to_dict()
        pred_label, prob = predict_transaction(best_model, txn, feature_names)
        match = "CORRECT" if pred_label == true_label else "WRONG"
        print(f"\n  True label : {true_label}")
        print(f"  Prediction : {pred_label}  (probability={prob:.4f})  [{match}]")

# ─── 8. Summary ───────────────────────────────────────────────────────────────
def print_summary(df_res, best_name):
    print("\n" + "="*60)
    print("  FINAL SUMMARY")
    print("="*60)
    print(df_res.to_string(index=False))
    print(f"\n  Best Model : {best_name}")
    print(f"  All plots saved to  : {PLOT_DIR}")
    print(f"  Best model saved to : {MODEL_DIR}/best_model.pkl")
    print("="*60 + "\n")

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "#"*60)
    print("#   CREDIT CARD FRAUD DETECTION – GROUP 2 CSC PROJECT    #")
    print("#"*60)

    df = load_data()
    exploratory_data_analysis(df)
    X_train, X_test, y_train, y_test, X_res, y_res, X_under, y_under, feat_names = preprocess(df)
    trained, preds, results = run_models(X_res, y_res, X_test, y_test, feat_names)
    df_res = visualise_results(trained, preds, results, X_test, y_test, feat_names)
    best_name, best_model = select_and_save(trained, df_res)
    demo_prediction(best_model, X_test, y_test, feat_names)
    print_summary(df_res, best_name)

if __name__ == "__main__":
    main()
