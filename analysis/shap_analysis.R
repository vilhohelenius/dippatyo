###############################################################
# analysis/shap_analysis.R
# Compute SHAP importances on MICROBIOME dataset for:
# - XGBoost (predcontrib)
# - CatBoost (ShapValues)
# - TabPFN (read from Python outputs)
# Saves results to results/shap/
###############################################################

source("/Users/vilhohelenius/Documents/Kouluhommat/Dippatyö/r_koodi/config.R")

library(dplyr)
library(readr)
library(tibble)
library(xgboost)
library(catboost)

SHAP_DIR <- file.path(RESULTS_DIR, "shap")
dir.create(SHAP_DIR, showWarnings = FALSE, recursive = TRUE)

###############################################################
# Load MICROBIOME test features
###############################################################

test_micro <- read_csv(file.path(EXPORT_DIR, "test_microbiome.csv"), show_col_types = FALSE)
X_test <- test_micro %>% select(-bmi)
X_test_mat <- as.matrix(X_test)
feature_names <- colnames(X_test_mat)

###############################################################
# 1) XGBoost SHAP (predcontrib)
###############################################################

XGB_MODEL_PATH <- file.path(RESULTS_DIR, "xgboost", "xgb_model.json")

if (file.exists(XGB_MODEL_PATH)) {
  xgb_model <- xgb.load(XGB_MODEL_PATH)
  
  shap_vals <- predict(
    xgb_model,
    newdata = X_test_mat,
    predcontrib = TRUE
  )
  
  shap_df <- as.data.frame(shap_vals)
  shap_df <- shap_df[, -ncol(shap_df), drop = FALSE]  # drop BIAS term
  colnames(shap_df) <- feature_names
  
  xgb_imp <- tibble(
    feature = feature_names,
    mean_abs_shap = colMeans(abs(as.matrix(shap_df)))
  ) %>%
    arrange(desc(mean_abs_shap)) %>%
    mutate(model = "XGBoost")
  
  write_csv(xgb_imp, file.path(SHAP_DIR, "shap_importance_xgboost_microbiome.csv"))
  message("Saved XGBoost SHAP importance.")
} else {
  warning("XGBoost model not found: ", XGB_MODEL_PATH)
}

###############################################################
# 2) CatBoost SHAP (ShapValues)
###############################################################

CB_MODEL_PATH <- file.path(RESULTS_DIR, "catboost", "model.cbm")

if (file.exists(CB_MODEL_PATH)) {
  cb_model <- catboost.load_model(CB_MODEL_PATH)
  
  test_pool <- catboost.load_pool(data = X_test)
  
  shap_cb <- catboost.get_feature_importance(
    cb_model,
    pool = test_pool,
    type = "ShapValues"
  )
  
  shap_cb <- shap_cb[, -ncol(shap_cb), drop = FALSE]  # drop expected value column
  colnames(shap_cb) <- feature_names
  
  cb_imp <- tibble(
    feature = feature_names,
    mean_abs_shap = colMeans(abs(shap_cb))
  ) %>%
    arrange(desc(mean_abs_shap)) %>%
    mutate(model = "CatBoost")
  
  write_csv(cb_imp, file.path(SHAP_DIR, "shap_importance_catboost_microbiome.csv"))
  message("Saved CatBoost SHAP importance.")
} else {
  warning("CatBoost model not found: ", CB_MODEL_PATH)
}

###############################################################
# 3) TabPFN SHAP importance (read from Python)
###############################################################

TABPFN_SHAP_PATH <- file.path(RESULTS_DIR, "tabpfn", "shap_importance_microbiome.csv")

if (file.exists(TABPFN_SHAP_PATH)) {
  tabpfn_imp <- read_csv(TABPFN_SHAP_PATH, show_col_types = FALSE) %>%
    transmute(
      feature = feature,
      mean_abs_shap = mean_abs_shap,
      model = "TabPFN"
    ) %>%
    arrange(desc(mean_abs_shap))
  
  write_csv(tabpfn_imp, file.path(SHAP_DIR, "shap_importance_tabpfn_microbiome.csv"))
  message("Saved TabPFN SHAP importance (copied).")
} else {
  warning("TabPFN SHAP file not found: ", TABPFN_SHAP_PATH)
}

message("Done: shap_analysis.R")
