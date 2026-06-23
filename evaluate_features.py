import pandas as pd
import numpy as np
from scipy import stats
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Suppress scipy warnings for perfect correlations or empty bins
warnings.filterwarnings("ignore")

from src.data import load_transactions
from src.features import compute_account_features, aggregate_labels_to_account

# --- Configuration ---
DATA_PATH = "datasets/HI-Small_Trans.csv"  # Adjust to your dataset path
OUTPUT_CSV = "thesis_feature_stats_corr.csv"

def run_feature_evaluation():
    print(f"Loading transactions from {DATA_PATH}...")
    tx = load_transactions(DATA_PATH)
    
    print("Building NetworKit graph and computing features (this may take a moment)...")
    feat = compute_account_features(tx)
    
    print("Aggregating labels...")
    labels = aggregate_labels_to_account(tx)
    
    if labels.empty or "label" not in labels.columns:
        raise ValueError("No laundering labels found in the dataset. Cannot perform evaluation.")
    
    # Merge features and labels
    df = feat.merge(labels, on="account", how="inner")
    
    # --- 1. Undersampling for a Balanced Dataset ---
    positives = df[df["label"] == 1]
    negatives = df[df["label"] == 0]
    
    n_positives = len(positives)
    print(f"Found {n_positives} laundering accounts. Sampling {n_positives} normal accounts for balance.")
    
    if n_positives == 0:
        raise ValueError("Zero positive (laundering) accounts found after aggregation.")
        
    sampled_negatives = negatives.sample(n=n_positives, random_state=42)
    balanced_df = pd.concat([positives, sampled_negatives]).reset_index(drop=True)
    
    feature_cols = [c for c in balanced_df.columns if c not in ["account", "label"]]
    results = []
    
    print("Running statistical tests on balanced dataset...")
    for col in feature_cols:
        
        feature_data = balanced_df[col].astype(float)
        labels_data = balanced_df["label"].astype(int)
        
        pos_values = feature_data[labels_data == 1]
        neg_values = feature_data[labels_data == 0]
        
        # --- A. Correlation (Point-Biserial / Pearson) ---
        corr_coef, corr_pval = stats.pearsonr(feature_data, labels_data)
        
        # --- B. Welch's T-Test ---
        t_stat, t_pval = stats.ttest_ind(pos_values, neg_values, equal_var=False)
        
        # --- C. Chi-Square Test of Independence ---
        try:
            binned_feature = pd.qcut(feature_data, q=4, duplicates='drop')
            contingency_table = pd.crosstab(binned_feature, labels_data)
            
            if contingency_table.shape[0] > 1: 
                chi2_stat, chi2_pval, dof, _ = stats.chi2_contingency(contingency_table)
            else:
                chi2_stat, chi2_pval = np.nan, np.nan
        except Exception:
            chi2_stat, chi2_pval = np.nan, np.nan

        # --- D. Univariate Logistic Regression ---
        # Reshape for sklearn and scale the feature for accurate coefficient representation
        X = feature_data.values.reshape(-1, 1)
        y = labels_data.values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        clf = LogisticRegression(class_weight='balanced', random_state=42)
        
        try:
            # Fit on the whole balanced set to get the feature's beta weight
            clf.fit(X_scaled, y)
            logreg_coef = clf.coef_[0][0]
            
            # Use 5-fold cross-validation to get the ROC-AUC score
            cv_auc = cross_val_score(clf, X_scaled, y, cv=5, scoring='roc_auc').mean()
        except Exception:
            logreg_coef = np.nan
            cv_auc = np.nan
            
        results.append({
            "Feature": col,
            "Laundering_Mean": pos_values.mean(),
            "Normal_Mean": neg_values.mean(),
            "Correlation": corr_coef,
            "Corr_p-value": corr_pval,
            "T-Stat": t_stat,
            "T-Test_p-value": t_pval,
            "Chi2-Stat": chi2_stat,
            "Chi2_p-value": chi2_pval,
            "LogReg_Coefficient": logreg_coef,
            "LogReg_ROC_AUC": cv_auc
        })
        
    results_df = pd.DataFrame(results)
    
    # Sort by Logistic Regression AUC to see the most predictive features at the top
    results_df = results_df.sort_values(by="Correlation", ascending=False)
    
    print("\n" + "="*100)
    print("FEATURE EVALUATION RESULTS (Sorted by Predictive ROC-AUC)")
    print("="*100)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV} for your thesis.")

if __name__ == "__main__":
    run_feature_evaluation()