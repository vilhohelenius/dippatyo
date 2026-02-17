source("/Users/vilhohelenius/Documents/Kouluhommat/Dippatyö/r_koodi/config.R")

library(catboost)
library(dplyr)
library(readr)
library(tibble)
library(purrr)
library(yardstick)

dir.create("/Users/vilhohelenius/Documents/Kouluhommat/Dippatyö/r_koodi/results/catboost", recursive = TRUE, showWarnings = FALSE)

# =========================
# Load data
# =========================

train <- read_csv("export/train_microbiome.csv", show_col_types = FALSE)
test  <- read_csv("export/test_microbiome.csv",  show_col_types = FALSE)

y_train <- train$bmi
y_test  <- test$bmi

X_train <- train %>% select(-bmi)
X_test  <- test  %>% select(-bmi)

# =========================
# CV folds (same logic as others)
# =========================

set.seed(SEED)
folds <- sample(rep(1:CV_FOLDS, length.out = nrow(X_train)))

# =========================
# Params
# =========================

params <- list(
  loss_function = "RMSE",
  iterations = 5000,          # upper bound, early stopping will stop earlier
  learning_rate = 0.05,
  depth = 6,
  random_seed = SEED,
  od_type = "Iter",
  od_wait = EARLY_STOPPING_ROUNDS,
  verbose = 100
)

cv_metrics <- list()
cv_importance <- list()

cat("Running CV for CatBoost...\n")

for (k in 1:CV_FOLDS) {
  
  idx_train <- which(folds != k)
  idx_val   <- which(folds == k)
  
  X_tr <- X_train[idx_train, , drop = FALSE]
  y_tr <- y_train[idx_train]
  
  X_val <- X_train[idx_val, , drop = FALSE]
  y_val <- y_train[idx_val]
  
  train_pool <- catboost.load_pool(data = X_tr, label = y_tr)
  val_pool   <- catboost.load_pool(data = X_val, label = y_val)
  
  model <- catboost.train(
    learn_pool = train_pool,
    test_pool  = val_pool,
    params = params
  )
  
  preds <- catboost.predict(model, val_pool)
  
  rmse_val <- rmse_vec(y_val, preds)
  mae_val  <- mae_vec(y_val, preds)
  r2_val   <- rsq_vec(y_val, preds)
  
  cv_metrics[[k]] <- tibble(
    fold = k,
    RMSE = rmse_val,
    MAE  = mae_val,
    R2   = r2_val
  )
  
  # Fold-level feature importance:
  # "PredictionValuesChange" is fast and stable enough for fold stability analysis
  imp_vals <- catboost.get_feature_importance(
    model,
    pool = val_pool,
    type = "PredictionValuesChange"
  )
  
  cv_importance[[k]] <- tibble(
    fold = k,
    feature = colnames(X_train),
    importance = as.numeric(imp_vals)
  )
}

cv_metrics_df <- bind_rows(cv_metrics)
cv_imp_df     <- bind_rows(cv_importance)

write_csv(cv_metrics_df, "results/catboost/metrics_cv.csv")
write_csv(cv_imp_df,     "results/catboost/feature_importance_folds.csv")

# =========================
# Train final model (train + early stop on test pool)
# =========================

cat("Training final CatBoost...\n")

train_pool <- catboost.load_pool(data = X_train, label = y_train)
test_pool  <- catboost.load_pool(data = X_test,  label = y_test)

final_model <- catboost.train(
  learn_pool = train_pool,
  test_pool  = test_pool,
  params = params
)

catboost.save_model(final_model, "results/catboost/model.cbm")

# =========================
# Test set evaluation
# =========================

pred_test <- catboost.predict(final_model, test_pool)

rmse_test <- rmse_vec(y_test, pred_test)
mae_test  <- mae_vec(y_test, pred_test)
r2_test   <- rsq_vec(y_test, pred_test)

metrics_test <- tibble(
  Model = "CatBoost",
  RMSE  = rmse_test,
  MAE   = mae_test,
  R2    = r2_test
)

write_csv(metrics_test, "results/catboost/metrics_test.csv")

# =========================
# Predictions
# =========================

pred_df <- tibble(
  y_true = y_test,
  y_pred = as.numeric(pred_test)
)

write_csv(pred_df, "results/catboost/predictions_test.csv")

# =========================
# Feature importance (final)
# =========================

final_imp <- catboost.get_feature_importance(
  final_model,
  pool = test_pool,
  type = "PredictionValuesChange"
)

importance_df <- tibble(
  feature = colnames(X_train),
  importance = as.numeric(final_imp)
) %>%
  arrange(desc(importance))

write_csv(importance_df, "results/catboost/feature_importance.csv")

cat("CatBoost DONE\n")