source("/Users/vilhohelenius/Documents/Kouluhommat/Dippatyö/r_koodi/config.R")

library(xgboost)
library(dplyr)
library(readr)
library(tibble)
library(purrr)
library(yardstick)

dir.create("/Users/vilhohelenius/Documents/Kouluhommat/Dippatyö/r_koodi/results/xgboost", recursive = TRUE, showWarnings = FALSE)

# =========================
# Load data 
# =========================

train <- read_csv("export/train_microbiome.csv", show_col_types = FALSE)
test  <- read_csv("export/test_microbiome.csv",  show_col_types = FALSE)

y_train <- train$bmi
y_test  <- test$bmi

X_train <- train %>% select(-bmi) %>% as.matrix()
X_test  <- test  %>% select(-bmi) %>% as.matrix()

dtrain_full <- xgb.DMatrix(data = X_train, label = y_train)
dtest       <- xgb.DMatrix(data = X_test,  label = y_test)

# =========================
# Parameters
# =========================

params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.05,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# =========================
# Cross-validation
# =========================

set.seed(SEED)
folds <- sample(rep(1:CV_FOLDS, length.out = nrow(X_train)))

cv_metrics <- list()
cv_importance <- list()

cat("Running CV for XGBoost...\n")

for (k in 1:CV_FOLDS) {
  
  idx_train <- which(folds != k)
  idx_val   <- which(folds == k)
  
  dtrain <- xgb.DMatrix(X_train[idx_train, ], label = y_train[idx_train])
  dval   <- xgb.DMatrix(X_train[idx_val, ],   label = y_train[idx_val])
  
  model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = 500,
    watchlist = list(val = dval),
    early_stopping_rounds = 30,
    verbose = 0
  )
  
  preds <- predict(model, dval)
  
  rmse_val <- rmse_vec(y_train[idx_val], preds)
  mae_val  <- mae_vec(y_train[idx_val], preds)
  r2_val   <- rsq_vec(y_train[idx_val], preds)
  
  cv_metrics[[k]] <- tibble(
    fold = k,
    RMSE = rmse_val,
    MAE  = mae_val,
    R2   = r2_val
  )
  
  # Feature importance (gain)
  imp <- xgb.importance(model = model)
  
  cv_importance[[k]] <- tibble(
    fold = k,
    feature = imp$Feature,
    importance = imp$Gain
  )
}

cv_metrics_df <- bind_rows(cv_metrics)
cv_imp_df     <- bind_rows(cv_importance)

write_csv(cv_metrics_df, "results/xgboost/metrics_cv.csv")
write_csv(cv_imp_df,     "results/xgboost/feature_importance_folds.csv")

# =========================
# Train final model
# =========================

cat("Training final XGBoost...\n")

final_model <- xgb.train(
  params = params,
  data = dtrain_full,
  nrounds = 500,
  watchlist = list(train = dtrain_full),
  early_stopping_rounds = 30,
  verbose = 0
)

xgb.save(final_model, "results/xgboost/xgb_model.json")

# =========================
# Test set evaluation
# =========================

pred_test <- predict(final_model, dtest)

rmse_test <- rmse_vec(y_test, pred_test)
mae_test  <- mae_vec(y_test, pred_test)
r2_test   <- rsq_vec(y_test, pred_test)

metrics_test <- tibble(
  Model = "XGBoost",
  RMSE  = rmse_test,
  MAE   = mae_test,
  R2    = r2_test
)

write_csv(metrics_test, "results/xgboost/metrics_test.csv")

# =========================
# Predictions
# =========================

pred_df <- tibble(
  y_true = y_test,
  y_pred = pred_test
)

write_csv(pred_df, "results/xgboost/predictions_test.csv")

# =========================
# Feature importance (final)
# =========================

importance <- xgb.importance(model = final_model)

importance_df <- tibble(
  feature = importance$Feature,
  importance = importance$Gain
) %>%
  arrange(desc(importance))

write_csv(importance_df, "results/xgboost/feature_importance.csv")

cat("XGBoost DONE\n")