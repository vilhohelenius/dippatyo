###############################################################
# microbiome_vs_metadata.R
# Compare predictive performance:
# 1) metadata only (age_years + sex)
# 2) microbiome only
# 3) full (microbiome + metadata)
###############################################################

source("/Users/vilhohelenius/Documents/Kouluhommat/Dippatyö/r_koodi/config.R")

library(dplyr)
library(readr)
library(tibble)
library(glmnet)
library(xgboost)
library(catboost)

dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

###############################################################
# Load datasets (CSV-first)
###############################################################

train_full <- read_csv(file.path(EXPORT_DIR, "train_full.csv"), show_col_types = FALSE)
test_full  <- read_csv(file.path(EXPORT_DIR, "test_full.csv"),  show_col_types = FALSE)

train_meta <- read_csv(file.path(EXPORT_DIR, "train_metadata.csv"), show_col_types = FALSE)
test_meta  <- read_csv(file.path(EXPORT_DIR, "test_metadata.csv"),  show_col_types = FALSE)

train_micro <- read_csv(file.path(EXPORT_DIR, "train_microbiome.csv"), show_col_types = FALSE)
test_micro  <- read_csv(file.path(EXPORT_DIR, "test_microbiome.csv"),  show_col_types = FALSE)

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
# Model wrappers (train on train-set, evaluate on test-set)
###############################################################

run_lasso <- function(train_df, test_df) {
  y_train <- train_df$bmi
  X_train <- train_df %>% select(-bmi)
  y_test  <- test_df$bmi
  X_test  <- test_df  %>% select(-bmi)
  
  model <- cv.glmnet(as.matrix(X_train), y_train, alpha = 1)
  pred  <- predict(model, as.matrix(X_test), s = "lambda.min")
  as.numeric(pred)
}

run_xgb <- function(train_df, test_df) {
  y_train <- train_df$bmi
  X_train <- train_df %>% select(-bmi)
  y_test  <- test_df$bmi
  X_test  <- test_df  %>% select(-bmi)
  
  dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
  dtest  <- xgb.DMatrix(data = as.matrix(X_test))
  
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
    nrounds = 300,
    verbose = 0
  )
  
  predict(model, dtest)
}

run_cat <- function(train_df, test_df) {
  y_train <- train_df$bmi
  X_train <- train_df %>% select(-bmi)
  y_test  <- test_df$bmi
  X_test  <- test_df  %>% select(-bmi)
  
  train_pool <- catboost.load_pool(data = X_train, label = y_train)
  test_pool  <- catboost.load_pool(data = X_test)
  
  model <- catboost.train(
    learn_pool = train_pool,
    test_pool  = NULL,
    params = list(
      loss_function = "RMSE",
      depth = 6,
      learning_rate = 0.05,
      iterations = 500,
      verbose = 100
    )
  )
  
  catboost.predict(model, test_pool)
}

###############################################################
# Run comparison
###############################################################

experiments <- list(
  metadata   = list(train = train_meta,  test = test_meta),
  microbiome = list(train = train_micro, test = test_micro),
  full       = list(train = train_full,  test = test_full)
)

results <- list()

for (name in names(experiments)) {
  
  cat("Running experiment:", name, "\n")
  
  tr <- experiments[[name]]$train
  te <- experiments[[name]]$test
  
  # Lasso
  pred_lasso <- run_lasso(tr, te)
  m_lasso <- evaluate_predictions(te$bmi, pred_lasso) %>%
    mutate(model = "Lasso", dataset = name)
  
  # XGBoost
  pred_xgb <- run_xgb(tr, te)
  m_xgb <- evaluate_predictions(te$bmi, pred_xgb) %>%
    mutate(model = "XGBoost", dataset = name)
  
  # CatBoost
  pred_cat <- run_cat(tr, te)
  m_cat <- evaluate_predictions(te$bmi, pred_cat) %>%
    mutate(model = "CatBoost", dataset = name)
  
  results[[name]] <- bind_rows(m_lasso, m_xgb, m_cat)
}

results_df <- bind_rows(results)

###############################################################
# Save results
###############################################################

write_csv(
  results_df,
  file.path(RESULTS_DIR, "microbiome_vs_metadata_results.csv")
)

cat("\nSaved:\n")
cat(" -", file.path(RESULTS_DIR, "microbiome_vs_metadata_results.csv"), "\n")