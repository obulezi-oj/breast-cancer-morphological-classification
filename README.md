# Comparative Evaluation of Logistic Regression & Machine Learning Algorithms for Breast Neoplasm Classification

[![R](https://img.shields.io/badge/Language-R_v4.x-blue.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Caret](https://img.shields.io/badge/Framework-caret-orange.svg)](https://topepo.github.io/caret/)

This repository contains the complete, leak-free computational pipeline and analysis script (`analysis of breast cancer.R`) accompanying the research paper on the morphological classification of breast neoplasms using the Wisconsin Diagnostic Breast Cancer (WDBC) benchmark dataset ($N = 569$; 357 benign, 212 malignant).

---

## 📌 Project Overview

In clinical oncology, differentiating benign from malignant breast neoplasms using Fine Needle Aspiration (FNA) cytology is critical. While complex non-linear machine learning models achieve high diagnostic accuracy, their "black-box" nature often limits clinical interpretability. 

This project evaluates the accuracy-interpretability trade-off between traditional statistical modeling (regularized Logistic Regression) and modern machine learning algorithms (Random Forest, Support Vector Machines, Gradient Boosting, k-NN, and SMOTE-oversampled Random Forest).

### Key Methodological & Clinical Highlights
- **Data Leakage Elimination**: Strict partitioning into a $70\%$ training set ($n = 399$) and a $30\%$ holdout test set ($n = 170$). All feature scaling ($\mathbf{Z}$-score normalization), SMOTE resampling, and hyperparameter grids are strictly embedded inside a 10-fold cross-validation pipeline.
- **Multicollinearity Resolution**: High pairwise linear collinearity ($r > 0.90$) among nuclear dimensions is resolved via penalized Lasso ($\ell_1$) Logistic Regression and feature selection.
- **Parametric Interpretability**: Quantifies exact diagnostic risk via regularized Odds Ratios (OR), proving that regularized parametric models can achieve high accuracy ($97.65\%$, $\text{ROC-AUC} = 0.9981$) while providing transparent biological rationale.
- **Statistical Significance Testing**: Features paired DeLong statistical testing, Brier probability calibration scoring, Partial Dependence Plots (PDP), and decision threshold optimization ($p_t \in [0.05, 0.95]$) to minimize false negatives in clinical decision-making.

---

## 📊 Evaluated Models & Workflow

1. **Lasso ($\ell_1$) Logistic Regression** (`glmnet`)
2. **k-Nearest Neighbors** (`knn`)
3. **Support Vector Machines with RBF Kernel** (`svmRadial`)
4. **Random Forest Classifier** (`rf`)
5. **Gradient Boosting Machines** (`gbm`)
6. **Random Forest + SMOTE Oversampling** (`themis` / `caret`)

---

## 📂 Repository Structure

```text
.
├── analysis of breast cancer.R       # Ultimate Grand Master R execution script
├── Breast_cancer_dataset.csv         # WDBC raw input dataset (or .xlsx)
├── README.md                         # Repository documentation
├── output_tables/                    # Auto-generated CSV summary & result tables
│   ├── Table1_Mean_Summary.csv
│   ├── Table2_SE_Summary.csv
│   ├── Table3_Worst_Summary.csv
│   ├── Table4_Welch_T_Test_Results.csv
│   ├── Table5_Corrected_Odds_Ratios.csv
│   ├── Table6_CV_ROC_AUC_Uncertainty_CI.csv
│   ├── Table7_Holdout_Test_Set_Performance.csv
│   ├── Table8_DeLong_Paired_ROC_Tests.csv
│   ├── Table9_Model_Calibration_Brier_Scores.csv
│   └── Table10_Clinical_Threshold_Optimization.csv
└── output_figures/                   # High-resolution (300 DPI) publication graphics
    ├── Figure1_Annotated_Density_Plot.png
    ├── Figure2_Enhanced_Correlation_Heatmap.png
    ├── Figure3_PCA_Biplot.png
    ├── Figure4_tSNE_Manifold_Embedding.png
    ├── Figure5_Overlay_ROC.png
    ├── Figure6_CV_Resamples_Boxplot.png
    ├── Figure7_Lasso_Coefficient_Path.png
    ├── Figure8_Forest_Plot_OddsRatios.png
    ├── Figure9_Probability_Calibration_Curves.png
    └── Figure10_Partial_Dependence_Plots.png
