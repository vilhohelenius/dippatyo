source("/Users/vilhohelenius/Documents/Kouluhommat/Dippatyö/r_koodi/config.R")

library(glmnet)
library(dplyr)
library(readr)
library(tibble)
library(purrr)
library(yardstick)

dir.create("/Users/vilhohelenius/Documents/Kouluhommat/Dippatyö/r_koodi/results/lasso", recursive = TRUE, showWarnings = FALSE)

# =========================
# Load data
# =========================

train <- read_csv("export/train_microbiome.csv", show_col_types = FALSE)
test  <- read_csv("export/test_microbiome.csv",  show_col_types = FALSE)

y_train <- train$bmi
y_test  <- test$bmi

X_train <- train %>% select(-bmi) %>% as.matrix()
X_test  <- test  %>% select(-bmi) %>% as.matrix()

# =========================
# Cross-validation setup
# =========================

set.seed(SEED)

folds <- sample(rep(1:CV_FOLDS, length.out = nrow(X_train)))

cv_metrics <- list()
cv_importance <- list()

cat("Running CV for LASSO...\n")

for (k in 1:CV_FOLDS) {
  
  idx_train <- which(folds != k)
  idx_val   <- which(folds == k)
  
  X_tr <- X_train[idx_train, ]
  y_tr <- y_train[idx_train]
  
  X_val <- X_train[idx_val, ]
  y_val <- y_train[idx_val]
  
  cv_fit <- cv.glmnet(
    X_tr, y_tr,
    alpha = 1,
    nfolds = 5,
    standardize = TRUE
  )
  
  lambda_best <- cv_fit$lambda.min
  
  model <- glmnet(X_tr, y_tr, alpha = 1, lambda = lambda_best)
  
  preds <- predict(model, X_val) %>% as.numeric()
  
  rmse_val <- rmse_vec(y_val, preds)
  mae_val  <- mae_vec(y_val, preds)
  r2_val   <- rsq_vec(y_val, preds)
  
  cv_metrics[[k]] <- tibble(
    fold = k,
    RMSE = rmse_val,
    MAE  = mae_val,
    R2   = r2_val
  )
  
  # Feature importance = absolute coefficients
  coef_vec <- as.vector(coef(model))[-1]
  names(coef_vec) <- colnames(X_train)
  
  cv_importance[[k]] <- tibble(
    fold = k,
    feature = names(coef_vec),
    importance = abs(coef_vec)
  )
}

cv_metrics_df <- bind_rows(cv_metrics)
cv_imp_df     <- bind_rows(cv_importance)

write_csv(cv_metrics_df, "results/lasso/metrics_cv.csv")
write_csv(cv_imp_df,     "results/lasso/feature_importance_folds.csv")

# =========================
# Train final model on full training set
# =========================

cat("Training final LASSO...\n")

final_cv <- cv.glmnet(
  X_train, y_train,
  alpha = 1,
  nfolds = CV_FOLDS,
  standardize = TRUE
)

lambda_best <- final_cv$lambda.min

lasso_model <- glmnet(X_train, y_train, alpha = 1, lambda = lambda_best)

saveRDS(lasso_model, "results/lasso/model.rds")

# =========================
# Test set evaluation
# =========================

pred_test <- predict(lasso_model, X_test) %>% as.numeric()

rmse_test <- rmse_vec(y_test, pred_test)
mae_test  <- mae_vec(y_test, pred_test)
r2_test   <- rsq_vec(y_test, pred_test)

metrics_test <- tibble(
  Model = "LASSO",
  RMSE  = rmse_test,
  MAE   = mae_test,
  R2    = r2_test
)

write_csv(metrics_test, "results/lasso/metrics_test.csv")

# =========================
# Predictions
# =========================

pred_df <- tibble(
  y_true = y_test,
  y_pred = pred_test
)

write_csv(pred_df, "results/lasso/predictions_test.csv")

# =========================
# Feature importance (final)
# =========================

coef_vec <- as.vector(coef(lasso_model))[-1]
names(coef_vec) <- colnames(X_train)

importance_df <- tibble(
  feature = names(coef_vec),
  importance = abs(coef_vec)
) %>%
  arrange(desc(importance))

write_csv(importance_df, "results/lasso/feature_importance.csv")

cat("LASSO DONE\n")