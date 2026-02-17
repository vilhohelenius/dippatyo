###############################################################
# scaling_experiment.R
# Learning curves (MICROBIOME): train on subset, evaluate on fixed test set
# Includes TabPFN scaling results (from Python) in plots if available
###############################################################

source("/Users/vilhohelenius/Documents/Kouluhommat/Dippatyö/r_koodi/config.R")

library(dplyr)
library(readr)
library(purrr)
library(ggplot2)
library(tibble)
library(glmnet)
library(xgboost)
library(catboost)

dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(PLOTS_DIR, showWarnings = FALSE, recursive = TRUE)

###############################################################
# Load train/test (CSV) — MICROBIOME ONLY
###############################################################

train_df <- read_csv(file.path(EXPORT_DIR, "train_microbiome.csv"), show_col_types = FALSE)
test_df  <- read_csv(file.path(EXPORT_DIR, "test_microbiome.csv"),  show_col_types = FALSE)

y_train_full <- train_df$bmi
X_train_full <- train_df %>% select(-bmi)

y_test <- test_df$bmi
X_test <- test_df %>% select(-bmi)

###############################################################
# Dataset sizes to test
###############################################################

SIZES <- c(50, 100, 500, 1000, 2000, 5000, nrow(X_train_full))

###############################################################
# Metrics
###############################################################

evaluate_predictions <- function(y_true, y_pred) {
  mse  <- mean((y_true - y_pred)^2)
  rmse <- sqrt(mse)
  mae  <- mean(abs(y_true - y_pred))
  r2   <- 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
  tibble(R2 = r2, RMSE = rmse, MAE = mae)
}

###############################################################
# Model wrappers (train on subset, predict test)
###############################################################

run_lasso <- function(X_sub, y_sub, X_test) {
  fit <- cv.glmnet(as.matrix(X_sub), y_sub, alpha = 1, nfolds = 5)
  pred <- predict(fit, as.matrix(X_test), s = "lambda.min")
  as.numeric(pred)
}

run_xgb <- function(X_sub, y_sub, X_test) {
  dtrain <- xgb.DMatrix(data = as.matrix(X_sub), label = y_sub)
  
  # CV for best iteration
  cv <- xgb.cv(
    params = list(
      objective = "reg:squarederror",
      eval_metric = "rmse",
      max_depth = 6,
      eta = 0.05,
      subsample = 0.8,
      colsample_bytree = 0.8
    ),
    data = dtrain,
    nrounds = 2000,
    nfold = 5,
    early_stopping_rounds = 30,
    verbose = 0
  )
  
  best_nrounds <- cv$early_stop$best_iteration
  
  model <- xgb.train(
    params = list(
      objective = "reg:squarederror",
      eval_metric = "rmse",
      max_depth = 6,
      eta = 0.05,
      subsample = 0.8,
      colsample_bytree = 0.8
    ),
    data = dtrain,
    nrounds = best_nrounds,
    verbose = 0
  )
  
  predict(model, as.matrix(X_test))
}

run_cat <- function(X_sub, y_sub, X_test) {
  # Light-weight early stopping: split subset into train/val
  set.seed(SEED)
  n <- nrow(X_sub)
  idx <- sample(seq_len(n), size = floor(0.8 * n))
  
  X_tr <- X_sub[idx, , drop = FALSE]
  y_tr <- y_sub[idx]
  X_val <- X_sub[-idx, , drop = FALSE]
  y_val <- y_sub[-idx]
  
  train_pool <- catboost.load_pool(data = X_tr, label = y_tr)
  val_pool   <- catboost.load_pool(data = X_val, label = y_val)
  
  model <- catboost.train(
    learn_pool = train_pool,
    test_pool  = val_pool,
    params = list(
      loss_function = "RMSE",
      depth = 6,
      learning_rate = 0.05,
      iterations = 5000,
      od_type = "Iter",
      od_wait = 50,
      verbose = 100
    )
  )
  
  test_pool <- catboost.load_pool(data = X_test)
  as.numeric(catboost.predict(model, test_pool))
}

###############################################################
# Run experiment (R models)
###############################################################

results <- list()

for (n in SIZES) {
  
  n <- min(n, nrow(X_train_full))
  cat("Running size:", n, "\n")
  
  X_sub <- X_train_full[1:n, , drop = FALSE]
  y_sub <- y_train_full[1:n]
  
  # Lasso
  pred_lasso <- run_lasso(X_sub, y_sub, X_test)
  m_lasso <- evaluate_predictions(y_test, pred_lasso) %>%
    mutate(model = "Lasso", n = n, dataset = "microbiome")
  
  # XGBoost
  pred_xgb <- run_xgb(X_sub, y_sub, X_test)
  m_xgb <- evaluate_predictions(y_test, pred_xgb) %>%
    mutate(model = "XGBoost", n = n, dataset = "microbiome")
  
  # CatBoost
  pred_cat <- run_cat(X_sub, y_sub, X_test)
  m_cat <- evaluate_predictions(y_test, pred_cat) %>%
    mutate(model = "CatBoost", n = n, dataset = "microbiome")
  
  results[[length(results) + 1]] <- bind_rows(m_lasso, m_xgb, m_cat)
}

results_df <- bind_rows(results)

###############################################################
# Save R results
###############################################################

write_csv(results_df, file.path(RESULTS_DIR, "scaling_results_r_test_microbiome.csv"))

###############################################################
# Load TabPFN scaling results (Python) and combine for plots
###############################################################

tabpfn_candidates <- c(
  file.path(RESULTS_DIR, "tabpfn", "scaling_results_microbiome.csv"),
  file.path(RESULTS_DIR, "tabpfn", "scaling_results.csv")
)

tabpfn_path <- tabpfn_candidates[file.exists(tabpfn_candidates)][1]

if (!is.na(tabpfn_path) && file.exists(tabpfn_path)) {
  tabpfn_df <- read_csv(tabpfn_path, show_col_types = FALSE)
  
  required_cols <- c("n", "R2", "RMSE", "MAE")
  missing_cols <- setdiff(required_cols, colnames(tabpfn_df))
  if (length(missing_cols) > 0) {
    stop("TabPFN scaling file is missing columns: ", paste(missing_cols, collapse = ", "))
  }
  
  tabpfn_df <- tabpfn_df %>%
    transmute(
      n = as.numeric(n),
      model = "TabPFN",
      R2 = as.numeric(R2),
      RMSE = as.numeric(RMSE),
      MAE = as.numeric(MAE),
      dataset = "microbiome"
    )
  
  combined_df <- bind_rows(results_df, tabpfn_df)
  message("Included TabPFN scaling results from: ", tabpfn_path)
  
} else {
  combined_df <- results_df
  warning("TabPFN scaling results not found. Plots include only R models.")
}

write_csv(combined_df, file.path(RESULTS_DIR, "scaling_results_combined_test_microbiome.csv"))

###############################################################
# Plots
###############################################################

p_r2 <- ggplot(combined_df, aes(x = n, y = R2, color = model)) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 2) +
  scale_x_log10() +
  labs(
    title = "Learning curves (microbiome): performance vs training set size (test set)",
    x = "Training set size (log scale)",
    y = expression(R^2),
    color = "Model"
  ) +
  theme_minimal(base_size = 13)

ggsave(
  file.path(PLOTS_DIR, "scaling_experiment_R2_test_microbiome.png"),
  p_r2,
  width = 7,
  height = 5,
  dpi = 300
)

p_rmse <- ggplot(combined_df, aes(x = n, y = RMSE, color = model)) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 2) +
  scale_x_log10() +
  labs(
    title = "Learning curves (microbiome): RMSE vs training set size (test set)",
    x = "Training set size (log scale)",
    y = "RMSE (lower is better)",
    color = "Model"
  ) +
  theme_minimal(base_size = 13)

ggsave(
  file.path(PLOTS_DIR, "scaling_experiment_RMSE_test_microbiome.png"),
  p_rmse,
  width = 7,
  height = 5,
  dpi = 300
)

print(p_r2)
print(p_rmse)

cat("\nScaling experiment (microbiome test-set) completed.\n")
cat("Saved:\n")
cat(" -", file.path(RESULTS_DIR, "scaling_results_r_test_microbiome.csv"), "\n")
cat(" -", file.path(RESULTS_DIR, "scaling_results_combined_test_microbiome.csv"), "\n")
cat(" -", file.path(PLOTS_DIR, "scaling_experiment_R2_test_microbiome.png"), "\n")
cat(" -", file.path(PLOTS_DIR, "scaling_experiment_RMSE_test_microbiome.png"), "\n")