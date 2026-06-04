# =============================================================================
# Credit Card Fraud Detection Using Machine Learning
# Group 2 - CSC312: Programming for Data Science using R and Python
# =============================================================================

cat(strrep("#", 62), "\n")
cat("#   CREDIT CARD FRAUD DETECTION - GROUP 2 CSC312 PROJECT   #\n")
cat(strrep("#", 62), "\n\n")

# ── Install & load packages ───────────────────────────────────────────────────
user_lib <- file.path(Sys.getenv("USERPROFILE"), "R", "win-library", "4.6")
dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(user_lib, .libPaths()))

needed <- c("ggplot2", "randomForest", "pROC", "ROSE", "dplyr")
missing <- needed[!sapply(needed, requireNamespace, quietly = TRUE)]
if (length(missing) > 0) {
  cat("Installing packages:", paste(missing, collapse = ", "), "\n")
  install.packages(missing, lib = user_lib,
                   repos = "https://cloud.r-project.org",
                   dependencies = FALSE, quiet = TRUE, type = "binary")
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(randomForest)
  library(rpart)
  library(pROC)
  library(ROSE)
  library(dplyr)
  library(gridExtra)
})

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK_DIR  <- "C:/Users/Hillary Chukwuma/OneDrive/Desktop/UPDATE/csc_project"
DATA_PATH <- file.path(WORK_DIR, "creditcard.csv")
PLOT_DIR  <- file.path(WORK_DIR, "plots_R")
MODEL_DIR <- file.path(WORK_DIR, "models_R")
dir.create(PLOT_DIR,  showWarnings = FALSE)
dir.create(MODEL_DIR, showWarnings = FALSE)

set.seed(42)

# =============================================================================
# STEP 1: LOAD DATASET
# =============================================================================
cat(strrep("=", 60), "\n")
cat("  STEP 1: LOADING DATASET\n")
cat(strrep("=", 60), "\n")

df <- read.csv(DATA_PATH)
cat(sprintf("  Dataset shape      : %d rows x %d columns\n", nrow(df), ncol(df)))
cat(sprintf("  Missing values     : %d\n", sum(is.na(df))))

counts <- table(df$Class)
cat("\n  Class distribution:\n")
cat(sprintf("    Legitimate (0) : %s  (%.2f%%)\n",
            format(counts["0"], big.mark = ","), counts["0"] / nrow(df) * 100))
cat(sprintf("    Fraudulent (1) : %s  (%.4f%%)\n",
            format(counts["1"], big.mark = ","), counts["1"] / nrow(df) * 100))

# =============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# =============================================================================
cat("\n", strrep("=", 60), "\n")
cat("  STEP 2: EXPLORATORY DATA ANALYSIS\n")
cat(strrep("=", 60), "\n")

df$Class_label <- ifelse(df$Class == 1, "Fraudulent", "Legitimate")

# 2a. Class distribution
p1 <- ggplot(df, aes(x = Class_label, fill = Class_label)) +
  geom_bar(color = "black") +
  scale_fill_manual(values = c("Legitimate" = "#4682B4", "Fraudulent" = "#DC143C")) +
  labs(title = "Class Distribution", x = "Transaction Type", y = "Count") +
  theme_minimal() + theme(legend.position = "none")

png(file.path(PLOT_DIR, "01_class_distribution.png"), width = 700, height = 500)
print(p1); dev.off()
cat("  Saved: 01_class_distribution.png\n")

# 2b. Amount distribution by class
p2 <- ggplot(df, aes(x = Amount, fill = Class_label)) +
  geom_histogram(bins = 60, alpha = 0.7, position = "identity") +
  scale_fill_manual(values = c("Legitimate" = "#4682B4", "Fraudulent" = "#DC143C")) +
  coord_cartesian(xlim = c(0, 2500)) +
  labs(title = "Transaction Amount Distribution",
       x = "Amount (USD)", y = "Count", fill = "Type") +
  theme_minimal()

png(file.path(PLOT_DIR, "02_amount_distribution.png"), width = 900, height = 500)
print(p2); dev.off()
cat("  Saved: 02_amount_distribution.png\n")

# 2c. Fraud rate by hour
df$Hour <- floor(df$Time / 3600) %% 24
hourly  <- df %>% group_by(Hour) %>%
  summarise(fraud_rate = mean(Class) * 100, .groups = "drop")

p3 <- ggplot(hourly, aes(x = Hour, y = fraud_rate)) +
  geom_line(color = "#DC143C", linewidth = 1) +
  geom_point(color = "#DC143C", size = 2) +
  labs(title = "Fraud Rate by Hour of Day", x = "Hour", y = "Fraud Rate (%)") +
  theme_minimal()

png(file.path(PLOT_DIR, "03_fraud_by_hour.png"), width = 900, height = 450)
print(p3); dev.off()
cat("  Saved: 03_fraud_by_hour.png\n")

# 2d. Top feature correlations with Class
num_df   <- df %>% select(-Class_label, -Hour)
cors     <- cor(num_df)[, "Class"]
cors     <- sort(abs(cors), decreasing = TRUE)
top10    <- names(cors)[2:11]

cat("\n  Top correlated features with fraud:\n")
for (f in top10) cat(sprintf("    %-6s : %.4f\n", f, cors[f]))

cor_df <- data.frame(Feature = top10, Correlation = cors[top10])
p4 <- ggplot(cor_df, aes(x = reorder(Feature, Correlation), y = Correlation)) +
  geom_bar(stat = "identity", fill = "steelblue", color = "black") +
  coord_flip() +
  labs(title = "Top 10 Features Correlated with Fraud",
       x = "Feature", y = "|Correlation|") +
  theme_minimal()

png(file.path(PLOT_DIR, "04_feature_correlation.png"), width = 800, height = 500)
print(p4); dev.off()
cat("  Saved: 04_feature_correlation.png\n")

# =============================================================================
# STEP 3: PREPROCESSING & CLASS IMBALANCE
# =============================================================================
cat("\n", strrep("=", 60), "\n")
cat("  STEP 3: PREPROCESSING & HANDLING CLASS IMBALANCE\n")
cat(strrep("=", 60), "\n")

df_clean        <- df %>% select(-Class_label, -Hour)
df_clean$Amount <- as.numeric(scale(df_clean$Amount))
df_clean$Time   <- as.numeric(scale(df_clean$Time))
df_clean$Class  <- as.factor(df_clean$Class)

# Stratified 80/20 split
fraud_idx  <- which(df_clean$Class == 1)
legit_idx  <- which(df_clean$Class == 0)
train_fraud <- sample(fraud_idx, floor(0.8 * length(fraud_idx)))
train_legit <- sample(legit_idx, floor(0.8 * length(legit_idx)))
train_idx   <- c(train_fraud, train_legit)

train_df <- df_clean[train_idx, ]
test_df  <- df_clean[-train_idx, ]
cat(sprintf("  Train size : %d  |  Test size : %d\n", nrow(train_df), nrow(test_df)))

# ROSE oversampling on training set
rose_out <- ROSE(Class ~ ., data = train_df, seed = 42)$data
rose_out$Class <- as.factor(rose_out$Class)
cat(sprintf("  After ROSE – Legit: %d  |  Fraud: %d\n",
            sum(rose_out$Class == 0), sum(rose_out$Class == 1)))

X_test <- test_df %>% select(-Class)
y_test <- as.integer(as.character(test_df$Class))

# =============================================================================
# STEP 4: MODEL TRAINING & EVALUATION
# =============================================================================
cat("\n", strrep("=", 60), "\n")
cat("  STEP 4: MODEL TRAINING & EVALUATION\n")
cat(strrep("=", 60), "\n")

# Helper: compute metrics
get_metrics <- function(y_true, y_pred, y_prob) {
  tp <- sum(y_pred == 1 & y_true == 1)
  fp <- sum(y_pred == 1 & y_true == 0)
  fn <- sum(y_pred == 0 & y_true == 1)
  precision <- if ((tp + fp) == 0) 0 else tp / (tp + fp)
  recall    <- if ((tp + fn) == 0) 0 else tp / (tp + fn)
  f1        <- if ((precision + recall) == 0) 0 else
                  2 * precision * recall / (precision + recall)
  roc_obj   <- roc(y_true, y_prob, quiet = TRUE)
  auc_val   <- as.numeric(auc(roc_obj))
  list(precision = precision, recall = recall,
       f1 = f1, auc = auc_val, roc = roc_obj)
}

print_metrics <- function(name, m, y_true, y_pred) {
  cat(sprintf("\n  [%s]\n",   name))
  cat(sprintf("    Precision : %.4f\n", m$precision))
  cat(sprintf("    Recall    : %.4f\n", m$recall))
  cat(sprintf("    F1-Score  : %.4f\n", m$f1))
  cat(sprintf("    ROC-AUC   : %.4f\n", m$auc))
  cat("    Confusion Matrix (rows=Actual, cols=Predicted):\n")
  print(table(Actual = y_true, Predicted = y_pred))
}

# ── Logistic Regression ───────────────────────────────────────────────────────
cat("\n  Training Logistic Regression...\n")
lr_model  <- glm(Class ~ ., data = rose_out, family = binomial)
lr_prob   <- predict(lr_model, X_test, type = "response")
lr_pred   <- as.integer(lr_prob >= 0.5)
lr_m      <- get_metrics(y_test, lr_pred, lr_prob)
print_metrics("Logistic Regression", lr_m, y_test, lr_pred)

# ── Decision Tree ─────────────────────────────────────────────────────────────
cat("\n  Training Decision Tree...\n")
dt_model  <- rpart(Class ~ ., data = rose_out, method = "class",
                   control = rpart.control(maxdepth = 8))
dt_prob   <- predict(dt_model, X_test, type = "prob")[, 2]
dt_pred   <- as.integer(dt_prob >= 0.5)
dt_m      <- get_metrics(y_test, dt_pred, dt_prob)
print_metrics("Decision Tree", dt_m, y_test, dt_pred)

# ── Random Forest ─────────────────────────────────────────────────────────────
cat("\n  Training Random Forest...\n")
rf_model  <- randomForest(Class ~ ., data = rose_out, ntree = 100,
                          importance = TRUE)
rf_prob   <- predict(rf_model, X_test, type = "prob")[, 2]
rf_pred   <- as.integer(rf_prob >= 0.5)
rf_m      <- get_metrics(y_test, rf_pred, rf_prob)
print_metrics("Random Forest", rf_m, y_test, rf_pred)

# =============================================================================
# STEP 5: VISUALISATIONS
# =============================================================================
cat("\n", strrep("=", 60), "\n")
cat("  STEP 5: VISUALISATIONS\n")
cat(strrep("=", 60), "\n")

model_names <- c("Logistic Regression", "Decision Tree", "Random Forest")
metrics_all <- list(lr_m, dt_m, rf_m)
colors_vec  <- c("#4C72B0", "#DD8452", "#55A868")

# 5a. Model comparison bar chart
comp_df <- data.frame(
  Model  = rep(model_names, each = 4),
  Metric = rep(c("Precision","Recall","F1","ROC-AUC"), 3),
  Score  = c(
    lr_m$precision, lr_m$recall, lr_m$f1, lr_m$auc,
    dt_m$precision, dt_m$recall, dt_m$f1, dt_m$auc,
    rf_m$precision, rf_m$recall, rf_m$f1, rf_m$auc
  )
)
comp_df$Model  <- factor(comp_df$Model,  levels = model_names)
comp_df$Metric <- factor(comp_df$Metric, levels = c("Precision","Recall","F1","ROC-AUC"))

p5 <- ggplot(comp_df, aes(x = Metric, y = Score, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge", color = "black", alpha = 0.85) +
  scale_fill_manual(values = setNames(colors_vec, model_names)) +
  ylim(0, 1.1) +
  labs(title = "Model Performance Comparison", y = "Score") +
  theme_minimal()

png(file.path(PLOT_DIR, "05_model_comparison.png"), width = 1000, height = 500)
print(p5); dev.off()
cat("  Saved: 05_model_comparison.png\n")

# 5b. ROC Curves
png(file.path(PLOT_DIR, "06_roc_curves.png"), width = 800, height = 600)
plot(lr_m$roc, col = colors_vec[1], lwd = 2, main = "ROC Curves – All Models")
plot(dt_m$roc, col = colors_vec[2], lwd = 2, add = TRUE)
plot(rf_m$roc, col = colors_vec[3], lwd = 2, add = TRUE)
legend("bottomright",
       legend = sprintf("%s (AUC=%.4f)", model_names,
                        sapply(metrics_all, `[[`, "auc")),
       col = colors_vec, lwd = 2, bty = "n")
dev.off()
cat("  Saved: 06_roc_curves.png\n")

# 5c. Confusion matrices
make_cm_plot <- function(y_true, y_pred, title) {
  cm    <- as.data.frame(table(Actual = factor(y_true, c(0,1)),
                               Predicted = factor(y_pred, c(0,1))))
  cm$Actual    <- ifelse(cm$Actual == "1",    "Fraud", "Legit")
  cm$Predicted <- ifelse(cm$Predicted == "1", "Fraud", "Legit")
  ggplot(cm, aes(x = Predicted, y = Actual, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), size = 6, fontface = "bold") +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(title = title) +
    theme_minimal() + theme(legend.position = "none")
}

cm1 <- make_cm_plot(y_test, lr_pred, "Logistic Regression")
cm2 <- make_cm_plot(y_test, dt_pred, "Decision Tree")
cm3 <- make_cm_plot(y_test, rf_pred, "Random Forest")

png(file.path(PLOT_DIR, "07_confusion_matrices.png"), width = 1200, height = 400)
gridExtra::grid.arrange(cm1, cm2, cm3, ncol = 3)
dev.off()
cat("  Saved: 07_confusion_matrices.png\n")

# 5d. Random Forest feature importance
imp_df <- data.frame(
  Feature    = rownames(importance(rf_model)),
  Importance = importance(rf_model)[, "MeanDecreaseGini"]
)
imp_df <- imp_df[order(imp_df$Importance, decreasing = TRUE), ][1:15, ]

p6 <- ggplot(imp_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue", color = "black") +
  coord_flip() +
  labs(title = "Top 15 Feature Importances – Random Forest",
       x = "Feature", y = "Mean Decrease Gini") +
  theme_minimal()

png(file.path(PLOT_DIR, "08_feature_importance.png"), width = 900, height = 500)
print(p6); dev.off()
cat("  Saved: 08_feature_importance.png\n")

# =============================================================================
# STEP 6: SELECT BEST MODEL & SAVE
# =============================================================================
cat("\n", strrep("=", 60), "\n")
cat("  STEP 6: MODEL SELECTION & SAVING\n")
cat(strrep("=", 60), "\n")

f1_scores <- c(lr_m$f1, dt_m$f1, rf_m$f1)
best_idx  <- which.max(f1_scores)
best_name <- model_names[best_idx]
best_model <- list(lr_model, dt_model, rf_model)[[best_idx]]
best_m    <- metrics_all[[best_idx]]

cat(sprintf("\n  Best model (by F1-Score) : %s\n", best_name))
cat(sprintf("    F1      : %.4f\n", best_m$f1))
cat(sprintf("    ROC-AUC : %.4f\n", best_m$auc))

model_path <- file.path(MODEL_DIR, "best_model.rds")
saveRDS(best_model, model_path)
cat(sprintf("  Saved to : %s\n", model_path))

# =============================================================================
# STEP 7: PREDICTION SYSTEM DEMO
# =============================================================================
cat("\n", strrep("=", 60), "\n")
cat("  STEP 7: PREDICTION SYSTEM DEMO\n")
cat(strrep("=", 60), "\n")

predict_transaction <- function(model, txn_row, model_type = "rf") {
  if (model_type == "lr") {
    prob <- predict(model, txn_row, type = "response")[[1]]
  } else if (model_type == "dt") {
    prob <- predict(model, txn_row, type = "prob")[1, 2]
  } else {
    prob <- predict(model, txn_row, type = "prob")[1, 2]
  }
  label <- ifelse(prob >= 0.5, "FRAUDULENT", "LEGITIMATE")
  list(label = label, probability = prob)
}

fraud_rows <- which(y_test == 1)
legit_rows <- which(y_test == 0)

for (info in list(
  list(idx = fraud_rows[1], true = "FRAUDULENT", type = "rf"),
  list(idx = legit_rows[1], true = "LEGITIMATE",  type = "rf")
)) {
  txn    <- X_test[info$idx, , drop = FALSE]
  result <- predict_transaction(best_model, txn, info$type)
  match  <- ifelse(result$label == info$true, "CORRECT", "WRONG")
  cat(sprintf("\n  True label : %s\n",         info$true))
  cat(sprintf("  Prediction : %s  (prob=%.4f)  [%s]\n",
              result$label, result$probability, match))
}

# =============================================================================
# FINAL SUMMARY
# =============================================================================
cat("\n", strrep("=", 60), "\n")
cat("  FINAL SUMMARY\n")
cat(strrep("=", 60), "\n")

summary_df <- data.frame(
  Model     = model_names,
  Precision = round(sapply(metrics_all, `[[`, "precision"), 4),
  Recall    = round(sapply(metrics_all, `[[`, "recall"),    4),
  F1        = round(sapply(metrics_all, `[[`, "f1"),        4),
  ROC_AUC   = round(sapply(metrics_all, `[[`, "auc"),       4)
)
print(summary_df, row.names = FALSE)
cat(sprintf("\n  Best Model  : %s\n", best_name))
cat(sprintf("  Plots saved : %s\n",  PLOT_DIR))
cat(sprintf("  Model saved : %s\n",  model_path))
cat(strrep("=", 60), "\n\n")
