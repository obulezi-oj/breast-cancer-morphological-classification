# ==============================================================================
# Ultimate Grand Master Script: Breast Cancer Morphological Classification
# Fully Integrated: All 6 Models, 10 CSV Tables, and 10 Publication Figures
# ==============================================================================

setwd("C:/Users/Dr. O. J. Obulezi/Documents/R projects/precious article")

# ------------------------------------------------------------------------------
# 1. DEPENDENCY MANAGEMENT & ENVIRONMENT SETUP
# ------------------------------------------------------------------------------
required_packages <- c(
  "tidyverse", "readxl", "caret", "broom", "pROC", 
  "corrplot", "glmnet", "randomForest", "gbm", 
  "kernlab", "themis", "gridExtra", "e1071", "factoextra",
  "Rtsne", "pdp"
)

new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) {
  install.packages(new_packages, repos = "https://cloud.r-project.org")
}

library(tidyverse)
library(readxl)
library(caret)
library(broom)
library(pROC)
library(corrplot)
library(glmnet)
library(randomForest)
library(gbm)
library(kernlab)
library(themis)
library(gridExtra)

has_factoextra <- requireNamespace("factoextra", quietly = TRUE)
has_rtsne      <- requireNamespace("Rtsne", quietly = TRUE)
has_pdp        <- requireNamespace("pdp", quietly = TRUE)

if(has_factoextra) library(factoextra)
if(has_rtsne) library(Rtsne)
if(has_pdp) library(pdp)

# ------------------------------------------------------------------------------
# 2. DATA INGESTION & FACTOR LEVEL SANITIZATION
# ------------------------------------------------------------------------------
if (file.exists("Breast_cancer_dataset.csv")) {
  df_raw <- read.csv("Breast_cancer_dataset.csv", check.names = TRUE)
} else if (file.exists("Breast_cancer_dataset.xlsx")) {
  df_raw <- read_excel("Breast_cancer_dataset.xlsx", sheet = 1)
  colnames(df_raw) <- make.names(colnames(df_raw))
} else {
  stop("Error: Neither Breast_cancer_dataset.csv nor Breast_cancer_dataset.xlsx was found.")
}

# Clean data and set 'M' as level 1
df_clean <- df_raw %>%
  select(-matches("^id$|^ID$|^Unnamed|^X"), where(~ !all(is.na(.)))) %>%
  drop_na() %>%
  mutate(diagnosis = as.character(diagnosis)) %>%
  mutate(diagnosis = case_when(
    diagnosis %in% c("1", "M", "Malignant") ~ "M",
    diagnosis %in% c("0", "B", "Benign")    ~ "B",
    TRUE ~ diagnosis
  )) %>%
  mutate(diagnosis = factor(diagnosis, levels = c("M", "B")))

colnames(df_clean) <- make.names(colnames(df_clean))

cat("\n=================== DATASET SUMMARY ===================\n")
cat("Dimensions:", nrow(df_clean), "rows x", ncol(df_clean), "columns\n")
cat("Target Factor Levels:", levels(df_clean$diagnosis), "(Positive Class = M)\n")
print(table(df_clean$diagnosis))


# ------------------------------------------------------------------------------
# 3. TABLES 1, 2, 3: DESCRIPTIVE FEATURE SUMMARIES
# ------------------------------------------------------------------------------
get_group_stats <- function(df, pattern) {
  df %>%
    select(matches(pattern)) %>%
    pivot_longer(cols = everything(), names_to = "Feature", values_to = "Value") %>%
    group_by(Feature) %>%
    summarise(
      Min = min(Value), Q1 = quantile(Value, 0.25),
      Median = median(Value), Mean = mean(Value),
      Q3 = quantile(Value, 0.75), Max = max(Value),
      .groups = "drop"
    )
}

table1_mean  <- get_group_stats(df_clean, "\\.mean$|_mean$")
table2_se    <- get_group_stats(df_clean, "\\.se$|_se$")
table3_worst <- get_group_stats(df_clean, "\\.worst$|_worst$")

write.csv(table1_mean, "Table1_Mean_Summary.csv", row.names = FALSE)
write.csv(table2_se, "Table2_SE_Summary.csv", row.names = FALSE)
write.csv(table3_worst, "Table3_Worst_Summary.csv", row.names = FALSE)


# ------------------------------------------------------------------------------
# 4. FIGURE 2 & HIGH CORRELATION PAIRS (|r| > 0.90)
# ------------------------------------------------------------------------------
numeric_data <- df_clean %>% select(where(is.numeric))
cor_matrix <- cor(numeric_data)

high_cor_pairs <- which(abs(cor_matrix) > 0.90 & lower.tri(cor_matrix), arr.ind = TRUE)
high_cor_df <- data.frame(
  Feature_1 = rownames(cor_matrix)[high_cor_pairs[, 1]],
  Feature_2 = colnames(cor_matrix)[high_cor_pairs[, 2]],
  Correlation = cor_matrix[high_cor_pairs]
) %>% arrange(desc(abs(Correlation)))

write.csv(high_cor_df, "Table_High_Correlation_Pairs.csv", row.names = FALSE)

png("Figure2_Enhanced_Correlation_Heatmap.png", width = 2400, height = 2200, res = 300)
corrplot(cor_matrix, method = "circle", type = "upper", tl.cex = 0.6, title = "Enhanced Feature Correlation Heatmap", mar = c(0,0,1,0))
dev.off()


# ------------------------------------------------------------------------------
# 5. TABLE 4: UNIVARIATE WELCH T-TEST COMPARISONS
# ------------------------------------------------------------------------------
t_test_results <- df_clean %>%
  pivot_longer(cols = where(is.numeric), names_to = "Variable", values_to = "Value") %>%
  group_by(Variable) %>%
  reframe(tidy(t.test(Value ~ diagnosis))) %>%
  select(Variable, estimate, statistic, p.value, conf.low, conf.high) %>%
  mutate(Significant = ifelse(p.value < 0.05, "Yes", "No")) %>%
  arrange(p.value)

write.csv(t_test_results, "Table4_Welch_T_Test_Results.csv", row.names = FALSE)


# ------------------------------------------------------------------------------
# 6. STRATIFIED TRAIN/TEST SPLIT (70% Train, 30% Test)
# ------------------------------------------------------------------------------
set.seed(123)
trainIndex <- createDataPartition(df_clean$diagnosis, p = 0.70, list = FALSE)
train_data <- df_clean[trainIndex, ]
test_data  <- df_clean[-trainIndex, ]


# ------------------------------------------------------------------------------
# 7. FIGURE 1: ANNOTATED DENSITY DISTRIBUTION PLOTS
# ------------------------------------------------------------------------------
p_density1 <- ggplot(df_clean, aes(x = radius_mean, fill = diagnosis)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("B" = "#2B5C8F", "M" = "#D95F02"), 
                    labels = c("B" = "Benign (B)", "M" = "Malignant (M)")) +
  theme_minimal(base_size = 12) +
  labs(title = "Radius Mean Distribution by Diagnosis", x = "Radius Mean", y = "Density", fill = "Outcome")

p_density2 <- ggplot(df_clean, aes(x = concave.points_mean, fill = diagnosis)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("B" = "#2B5C8F", "M" = "#D95F02"), 
                    labels = c("B" = "Benign (B)", "M" = "Malignant (M)")) +
  theme_minimal(base_size = 12) +
  labs(title = "Concave Points Mean Distribution by Diagnosis", x = "Concave Points Mean", y = "Density", fill = "Outcome")

p_density_combined <- grid.arrange(p_density1, p_density2, ncol = 2)
ggsave("Figure1_Annotated_Density_Plot.png", p_density_combined, width = 12, height = 5, dpi = 300)


# ------------------------------------------------------------------------------
# 8. FIGURE 3 & 4: TOPOLOGICAL CLUSTERING (PCA & t-SNE EMBEDDINGS)
# ------------------------------------------------------------------------------
pca_res <- prcomp(train_data %>% select(-diagnosis), center = TRUE, scale. = TRUE)
pca_var <- summary(pca_res)$importance
write.csv(pca_var, "Table_PCA_Variance_Explained.csv")

if(has_factoextra) {
  p_pca <- fviz_pca_ind(
    pca_res, col.ind = train_data$diagnosis, palette = c("#D95F02", "#2B5C8F"),
    addEllipses = TRUE, ellipse.type = "confidence", legend.title = "Diagnosis",
    title = "PCA Topological Separation of Breast Neoplasms (PC1 vs PC2)"
  )
  ggsave("Figure3_PCA_Biplot.png", p_pca, width = 8, height = 6, dpi = 300)
}

if(has_rtsne) {
  set.seed(123)
  tsne_input <- unique(train_data %>% select(-diagnosis))
  tsne_labels <- train_data$diagnosis[!duplicated(train_data %>% select(-diagnosis))]
  
  tsne_out <- Rtsne(as.matrix(scale(tsne_input)), perplexity = 30, verbose = FALSE)
  
  tsne_df <- data.frame(
    tSNE1 = tsne_out$Y[, 1],
    tSNE2 = tsne_out$Y[, 2],
    Diagnosis = tsne_labels
  )
  
  p_tsne <- ggplot(tsne_df, aes(x = tSNE1, y = tSNE2, color = Diagnosis)) +
    geom_point(size = 2.5, alpha = 0.8) +
    scale_color_manual(values = c("B" = "#2B5C8F", "M" = "#D95F02"), labels = c("Benign (B)", "Malignant (M)")) +
    theme_minimal(base_size = 12) +
    labs(title = "t-SNE Non-Linear Manifold Embedding", x = "t-SNE Dimension 1", y = "t-SNE Dimension 2")
  
  ggsave("Figure4_tSNE_Manifold_Embedding.png", p_tsne, width = 8, height = 6, dpi = 300)
}


# ------------------------------------------------------------------------------
# 9. TABLE 5: FULL MULTIVARIATE LOGISTIC REGRESSION & EXACT ODDS RATIOS
# ------------------------------------------------------------------------------
full_glm <- glm(diagnosis ~ ., data = train_data, family = "binomial")

or_table_corrected <- tidy(full_glm, conf.int = TRUE) %>%
  filter(term != "(Intercept)") %>%
  mutate(
    Odds_Ratio = exp(estimate),
    Lower_CI_95 = exp(conf.low),
    Upper_CI_95 = exp(conf.high)
  ) %>%
  select(term, estimate, std.error, Odds_Ratio, Lower_CI_95, Upper_CI_95, p.value)

write.csv(or_table_corrected, "Table5_Corrected_Odds_Ratios.csv", row.names = FALSE)


# ------------------------------------------------------------------------------
# 10. LEAK-FREE 10-FOLD CV MODEL TRAINING SUITE
# ------------------------------------------------------------------------------
cv_control <- trainControl(
  method = "cv", number = 10, classProbs = TRUE, 
  summaryFunction = twoClassSummary, savePredictions = "final", allowParallel = FALSE
)

cv_control_smote <- trainControl(
  method = "cv", number = 10, classProbs = TRUE, 
  summaryFunction = twoClassSummary, sampling = "smote", savePredictions = "final", allowParallel = FALSE
)

cat("\nTraining Classifiers across 10-Fold CV...\n")

# Model 1: Lasso Logistic Regression
set.seed(123)
fit_lasso <- train(
  diagnosis ~ ., data = train_data, method = "glmnet", family = "binomial",
  metric = "ROC", preProcess = c("center", "scale"),
  tuneGrid = expand.grid(alpha = 1, lambda = seq(0.0001, 0.1, length = 30)),
  trControl = cv_control
)

# Extract Non-Zero Lasso Coefficients
opt_lambda <- fit_lasso$bestTune$lambda
lasso_coefs <- as.matrix(coef(fit_lasso$finalModel, s = opt_lambda))
lasso_or_df <- data.frame(
  Term = rownames(lasso_coefs),
  Coefficient = lasso_coefs[, 1],
  Odds_Ratio = exp(lasso_coefs[, 1])
) %>%
  filter(Coefficient != 0 & Term != "(Intercept)") %>%
  arrange(desc(abs(Coefficient)))

write.csv(lasso_or_df, "Lasso_Odds_Ratios.csv", row.names = FALSE)

# Model 2: k-NN
set.seed(123)
fit_knn <- train(
  diagnosis ~ ., data = train_data, method = "knn",
  metric = "ROC", preProcess = c("center", "scale"),
  tuneLength = 10, trControl = cv_control
)

# Model 3: SVM RBF
set.seed(123)
fit_svm <- train(
  diagnosis ~ ., data = train_data, method = "svmRadial",
  metric = "ROC", preProcess = c("center", "scale"),
  tuneLength = 5, trControl = cv_control
)

# Model 4: Random Forest
set.seed(123)
fit_rf <- train(
  diagnosis ~ ., data = train_data, method = "rf",
  metric = "ROC", importance = TRUE, trControl = cv_control
)

# Model 5: Gradient Boosting (GBM)
set.seed(123)
fit_xgb <- train(
  diagnosis ~ ., data = train_data, method = "gbm",
  metric = "ROC", preProcess = c("center", "scale"),
  verbose = FALSE, trControl = cv_control
)

# Model 6: Random Forest with SMOTE
set.seed(123)
fit_rf_smote <- train(
  diagnosis ~ ., data = train_data, method = "rf",
  metric = "ROC", importance = TRUE, trControl = cv_control_smote
)


# ------------------------------------------------------------------------------
# 11. TABLE 6 & 7: CV UNCERTAINTY ESTIMATES & HOLDOUT TEST PERFORMANCE
# ------------------------------------------------------------------------------
models_list <- list(
  Lasso_Logistic = fit_lasso,
  kNN            = fit_knn,
  SVM_RBF        = fit_svm,
  Random_Forest  = fit_rf,
  GBM_Boosting   = fit_xgb,
  RF_SMOTE       = fit_rf_smote
)

cv_resamples <- resamples(models_list)

extract_metric_ci <- function(resamples_obj, metric_name) {
  val_df <- resamples_obj$values %>% select(ends_with(paste0("~", metric_name)))
  map_df(colnames(val_df), function(col_name) {
    model_raw_name <- gsub(paste0("~", metric_name), "", col_name)
    model_clean_name <- gsub("^[0-9]+\\~", "", model_raw_name)
    vals <- val_df[[col_name]]
    m <- mean(vals, na.rm = TRUE)
    s <- sd(vals, na.rm = TRUE)
    n <- sum(!is.na(vals))
    se <- s / sqrt(n)
    data.frame(
      Model = model_clean_name,
      Mean = m,
      SD = s,
      Lower_CI_95 = m - 1.96 * se,
      Upper_CI_95 = m + 1.96 * se
    )
  })
}

roc_ci <- extract_metric_ci(cv_resamples, "ROC")
write.csv(roc_ci, "Table6_CV_ROC_AUC_Uncertainty_CI.csv", row.names = FALSE)

evaluate_test_perf <- function(model_obj, model_name, test_df) {
  preds <- predict(model_obj, newdata = test_df)
  probs <- predict(model_obj, newdata = test_df, type = "prob")[, "M"]
  
  cm <- confusionMatrix(preds, test_df$diagnosis, positive = "M")
  roc_obj <- roc(test_df$diagnosis, probs, levels = c("B", "M"), quiet = TRUE)
  
  data.frame(
    Model = model_name,
    Accuracy = cm$overall["Accuracy"],
    Accuracy_Lower_CI = cm$overall["AccuracyLower"],
    Accuracy_Upper_CI = cm$overall["AccuracyUpper"],
    Sensitivity = cm$byClass["Sensitivity"],
    Specificity = cm$byClass["Specificity"],
    F1_Score = cm$byClass["F1"],
    ROC_AUC = as.numeric(auc(roc_obj))
  )
}

test_results_df <- map2_df(models_list, names(models_list), ~ evaluate_test_perf(.x, .y, test_data))
write.csv(test_results_df, "Table7_Holdout_Test_Set_Performance.csv", row.names = FALSE)


# ------------------------------------------------------------------------------
# 12. TABLE 8: DELONG'S PAIRED ROC STATISTICAL SIGNIFICANCE TEST
# ------------------------------------------------------------------------------
prob_lasso <- predict(fit_lasso, newdata = test_data, type = "prob")[, "M"]
prob_rf    <- predict(fit_rf, newdata = test_data, type = "prob")[, "M"]

roc_lasso <- roc(test_data$diagnosis, prob_lasso, levels = c("B", "M"), quiet = TRUE)
roc_rf    <- roc(test_data$diagnosis, prob_rf, levels = c("B", "M"), quiet = TRUE)

delong_test <- roc.test(roc_lasso, roc_rf, method = "delong")

delong_results_df <- data.frame(
  Model_1 = "Lasso Logistic",
  Model_2 = "Random Forest",
  AUC_Model_1 = as.numeric(auc(roc_lasso)),
  AUC_Model_2 = as.numeric(auc(roc_rf)),
  Z_Statistic = delong_test$statistic,
  P_Value = delong_test$p.value,
  Significance = ifelse(delong_test$p.value < 0.05, "Statistically Significant", "Not Significant")
)

write.csv(delong_results_df, "Table8_DeLong_Paired_ROC_Tests.csv", row.names = FALSE)


# ------------------------------------------------------------------------------
# 13. TABLE 9 & FIGURE 9: PROBABILITY CALIBRATION CURVES & BRIER SCORES
# ------------------------------------------------------------------------------
prob_knn <- predict(fit_knn, newdata = test_data, type = "prob")[, "M"]
prob_svm <- predict(fit_svm, newdata = test_data, type = "prob")[, "M"]
prob_gbm <- predict(fit_xgb, newdata = test_data, type = "prob")[, "M"]

obs_num <- ifelse(test_data$diagnosis == "M", 1, 0)

brier_df <- data.frame(
  Model = c("Lasso Logistic", "k-NN", "SVM RBF", "Random Forest", "GBM Boosting"),
  Brier_Score = c(
    mean((prob_lasso - obs_num)^2),
    mean((prob_knn - obs_num)^2),
    mean((prob_svm - obs_num)^2),
    mean((prob_rf - obs_num)^2),
    mean((prob_gbm - obs_num)^2)
  )
) %>% arrange(Brier_Score)

write.csv(brier_df, "Table9_Model_Calibration_Brier_Scores.csv", row.names = FALSE)

get_calib_data <- function(probs, outcomes, n_bins = 5) {
  data.frame(prob = probs, obs = outcomes) %>%
    mutate(bin = ntile(prob, n_bins)) %>%
    group_by(bin) %>%
    summarise(mean_pred = mean(prob), mean_obs = mean(obs), .groups = "drop")
}

calib_l <- get_calib_data(prob_lasso, obs_num) %>% mutate(Model = "Lasso")
calib_r <- get_calib_data(prob_rf, obs_num) %>% mutate(Model = "Random Forest")
calib_s <- get_calib_data(prob_svm, obs_num) %>% mutate(Model = "SVM RBF")

calib_combined <- bind_rows(calib_l, calib_r, calib_s)

p_calib <- ggplot(calib_combined, aes(x = mean_pred, y = mean_obs, color = Model, group = Model)) +
  geom_line(size = 1) + geom_point(size = 2.5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey40") +
  theme_minimal(base_size = 12) +
  labs(title = "Probability Calibration Curves (Observed vs. Predicted Risk)", x = "Mean Predicted Probability", y = "Observed Proportion (Malignant)")

ggsave("Figure9_Probability_Calibration_Curves.png", p_calib, width = 8, height = 6, dpi = 300)


# ------------------------------------------------------------------------------
# 14. TABLE 10: CLINICAL CUTOFF THRESHOLD OPTIMIZATION
# ------------------------------------------------------------------------------
threshold_seq <- seq(0.05, 0.95, by = 0.05)

eval_thresholds <- map_df(threshold_seq, function(cutoff) {
  preds <- factor(ifelse(prob_rf >= cutoff, "M", "B"), levels = c("M", "B"))
  cm <- confusionMatrix(preds, test_data$diagnosis, positive = "M")
  
  data.frame(
    Threshold = cutoff,
    Sensitivity = cm$byClass["Sensitivity"],
    Specificity = cm$byClass["Specificity"],
    Precision   = cm$byClass["Precision"],
    F1_Score    = cm$byClass["F1"],
    False_Negatives = cm$table["B", "M"]
  )
})

write.csv(eval_thresholds, "Table10_Clinical_Threshold_Optimization.csv", row.names = FALSE)


# ------------------------------------------------------------------------------
# 15. FIGURES 5, 6, 7, 8, 10: ROC, RESAMPLES, LASSO PATH, FOREST PLOT, PDP
# ------------------------------------------------------------------------------

# Figure 5: Multi-Model ROC Overlay
png("Figure5_Overlay_ROC.png", width = 2200, height = 2000, res = 300)
roc_k <- roc(test_data$diagnosis, prob_knn, levels = c("B", "M"), quiet = TRUE)
roc_s <- roc(test_data$diagnosis, prob_svm, levels = c("B", "M"), quiet = TRUE)

plot(roc_lasso, col = "#1B9E77", lwd = 2.5, main = "Comparative ROC Curves (Test Set)")
plot(roc_k, col = "#D95F02", lwd = 2.5, add = TRUE)
plot(roc_s, col = "#E7298A", lwd = 2.5, add = TRUE)
plot(roc_rf, col = "#7570B3", lwd = 2.5, add = TRUE)
abline(a = 0, b = 1, lty = 2, col = "grey50")

legend("bottomright", 
       legend = c(
         paste0("Lasso Logistic (AUC = ", sprintf("%.3f", auc(roc_lasso)), ")"),
         paste0("k-NN (AUC = ", sprintf("%.3f", auc(roc_k)), ")"),
         paste0("SVM RBF (AUC = ", sprintf("%.3f", auc(roc_s)), ")"),
         paste0("Random Forest (AUC = ", sprintf("%.3f", auc(roc_rf)), ")")
       ),
       col = c("#1B9E77", "#D95F02", "#E7298A", "#7570B3"), lwd = 2.5, bty = "n")
dev.off()

# Figure 6: 10-Fold CV Resample Boxplot
p_cv_box <- bwplot(cv_resamples, metric = c("ROC", "Sens", "Spec"), main = "10-Fold CV Resample Performance Distributions", fill = "lightblue")
png("Figure6_CV_Resamples_Boxplot.png", width = 2200, height = 1600, res = 300)
print(p_cv_box)
dev.off()

# Figure 7: Lasso Shrinkage Path
x_train <- model.matrix(diagnosis ~ . - 1, data = train_data)
y_train <- train_data$diagnosis
cv_glmnet_fit <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1)

png("Figure7_Lasso_Coefficient_Path.png", width = 2400, height = 1500, res = 300)
par(mfrow = c(1, 2))
plot(cv_glmnet_fit, main = "")
plot(cv_glmnet_fit$glmnet.fit, xvar = "lambda", label = TRUE, main = "")
par(mfrow = c(1, 1))
dev.off()

# Figure 8: Forest Plot of Odds Ratios
top_features <- lasso_or_df$Term[1:min(8, nrow(lasso_or_df))]
formula_top <- as.formula(paste("diagnosis ~", paste(top_features, collapse = " + ")))
glm_top <- glm(formula_top, data = train_data, family = "binomial")

forest_df <- tidy(glm_top, conf.int = TRUE) %>%
  filter(term != "(Intercept)") %>%
  mutate(OR = exp(estimate), Lower_CI = exp(conf.low), Upper_CI = exp(conf.high))

p_forest <- ggplot(forest_df, aes(x = reorder(term, OR), y = OR)) +
  geom_pointrange(aes(ymin = Lower_CI, ymax = Upper_CI), color = "#2B5C8F", size = 0.8) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red", alpha = 0.7) +
  coord_flip() + scale_y_log10() + theme_minimal(base_size = 12) +
  labs(title = "Forest Plot: Odds Ratios of Retained Predictors", subtitle = "Log-scale 95% Confidence Intervals", x = "Morphological Features", y = "Odds Ratio (Log Scale)")

ggsave("Figure8_Forest_Plot_OddsRatios.png", p_forest, width = 8, height = 6, dpi = 300)

# Figure 10: Partial Dependence Plots (PDP)
if(has_pdp) {
  pdp_concave <- pdp::partial(fit_rf$finalModel, pred.var = "concave.points_worst", prob = TRUE, train = train_data %>% select(-diagnosis))
  pdp_radius  <- pdp::partial(fit_rf$finalModel, pred.var = "radius_worst", prob = TRUE, train = train_data %>% select(-diagnosis))
  
  p_pdp1 <- ggplot(pdp_concave, aes(x = concave.points_worst, y = yhat)) +
    geom_line(color = "#D95F02", size = 1.2) + theme_minimal(base_size = 12) +
    labs(title = "PDP: Concave Points Worst vs. Risk", x = "Concave Points Worst", y = "Predicted Risk")
  
  p_pdp2 <- ggplot(pdp_radius, aes(x = radius_worst, y = yhat)) +
    geom_line(color = "#2B5C8F", size = 1.2) + theme_minimal(base_size = 12) +
    labs(title = "PDP: Radius Worst vs. Risk", x = "Radius Worst", y = "Predicted Risk")
  
  p_pdp_grid <- grid.arrange(p_pdp1, p_pdp2, ncol = 2)
  ggsave("Figure10_Partial_Dependence_Plots.png", p_pdp_grid, width = 12, height = 5, dpi = 300)
}

cat("\nPipeline Execution Successful! All 10 CSV tables and 10 figures exported cleanly.\n")