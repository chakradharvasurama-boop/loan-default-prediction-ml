# Loan Default Prediction

## Business Framing
Lending decisions have asymmetric risk:
- False negative (approve a likely defaulter): direct credit loss.
- False positive (reject a likely payer): missed revenue and customer value.

This project builds a default-risk classifier to support approval decisions with configurable risk thresholds.

## Problem Definition
- **Task:** Binary classification (`Charged Off` = 1, `Fully Paid` = 0)
- **Goal:** Maximize ranking and defaulter detection under class imbalance, not just raw accuracy.
- **Primary metrics:** ROC-AUC and Recall for `Charged Off`.

## Data and Scope
- Source: Lending Club application dataset (~38K rows in notebook workflow).
- Class distribution: defaulters are a minority (~14.5%), creating strong imbalance.
- Leakage controls: removed columns unavailable at approval time (e.g., `installment`, `last_pymnt_amnt`) and identifier fields.

## Pipeline Summary
- Data cleaning: type fixes, missing-value handling.
- Feature engineering: target `default_flag` from `loan_status`.
- Encoding: ordinal + one-hot via `ColumnTransformer`.
- Scaling: numeric scaling for Logistic Regression only.
- Split: train/test evaluation setup.
- Imbalance strategy:
  - `class_weight='balanced'` for tree models.
  - `scale_pos_weight` for XGBoost.
  - No synthetic resampling for tree-based models in this workflow.
- Tuning: `RandomizedSearchCV` on tree-based models with `roc_auc` scoring.
- Threshold tuning: explicit recall/accuracy trade-off analysis.

## Model Comparison (from notebook)
| Model | Key Outcome |
|---|---|
| Logistic Regression | Strong baseline; ROC-AUC ~0.71; high recall at lower thresholds. |
| Tuned Decision Tree | Recall improved vs baseline tree, but weaker ranking stability (lower ROC-AUC). |
| Tuned Random Forest | ROC-AUC ~0.72; improved ranking and recall after threshold tuning. |
| Tuned XGBoost | Best overall: ROC-AUC ~0.73 and recall ~0.67 at threshold 0.5. |

**Final model selected: XGBoost**, based on best ranking quality + stable recall-threshold behavior for risk-controlled decisions.

## Evaluation Approach
- Classification metrics: Accuracy, Precision, Recall (`Charged Off`), F1-score.
- Ranking diagnostics: ROC and Precision-Recall curves.
- Business-centric validation: threshold sweep to tune default capture vs approval strictness.

## Production Considerations
- Serve calibrated default probability and apply policy threshold by risk appetite.
- Keep preprocessing + model in one versioned inference pipeline.
- Monitor:
  - Data quality and schema checks.
  - Population drift (feature distributions) and target/performance drift (AUC, recall, approval rate, bad rate).
- Retrain policy: periodic + event-triggered (drift/performance breach).
- Governance:
  - Preserve leakage-safe features only (application-time fields).
  - Track decision outcomes for feedback-loop learning.
  - Add explainability layer (e.g., SHAP) for analyst review and adverse-action support.

## Repo Artifact
- Main analysis notebook: `Chakradhar_Reddy_Yerragudi_LendingClubLoanApprovalSystem.ipynb`
- Data file used: `loans.csv`
